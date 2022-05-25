
from inspect import Parameter
from mira.topic_model.base import BaseModel, EarlyStopping, ModelParamError
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm.notebook import tqdm, trange
import numpy as np
import logging
from math import ceil
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
logger = logging.getLogger(__name__)
from mira.topic_model.mine import WassersteinDualFlat, WassersteinDual
from pyro import poutine
from functools import partial


class CovariateModelMixin(BaseModel):

    def __init__(self, endogenous_key = None,
            exogenous_key = None,
            counts_layer = None,
            covariates_keys = None,
            extra_features_keys = None,
            num_topics = 16,
            hidden = 128,
            num_layers = 3,
            num_epochs = 40,
            decoder_dropout = 0.2,
            encoder_dropout = 0.02,
            use_cuda = True,
            seed = 0,
            min_learning_rate = 1e-6,
            max_learning_rate = 1e-1,
            beta = 0.95,
            batch_size = 64,
            initial_pseudocounts = 50,
            nb_parameterize_logspace = True,
            embedding_size = None,
            kl_strategy = 'cyclic',
            reconstruction_weight = 1/2,
            dataset_loader_workers = 0,
            dependence_lr = 1e-4,
            dependence_beta = 1000,
            dependence_hidden = 64,
            dependence_model = WassersteinDual
            ):
        super().__init__()

        self.endogenous_key = endogenous_key
        self.exogenous_key = exogenous_key
        self.counts_layer = counts_layer
        self.covariates_keys = covariates_keys
        self.extra_features_keys = extra_features_keys
        self.num_topics = num_topics
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.use_cuda = use_cuda
        self.seed = seed
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.initial_pseudocounts = initial_pseudocounts
        self.nb_parameterize_logspace = nb_parameterize_logspace
        self.embedding_size = embedding_size
        self.kl_strategy = kl_strategy
        self.reconstruction_weight = reconstruction_weight
        self.dataset_loader_workers = dataset_loader_workers
        self.dependence_model = dependence_model
        self.dependence_lr = dependence_lr
        self.dependence_beta = dependence_beta
        self.dependence_hidden = dependence_hidden

    def _get_weights(self, on_gpu = True, inference_mode = False):
        super()._get_weights(on_gpu=on_gpu, inference_mode=inference_mode)

        self.dependence_network = self.dependence_model(
            self.dependence_model.get_statistics_network(
            2*self.num_exog_features, 
            self.dependence_hidden)
        ).to(self.device)


    @adi.wraps_modelfunc(tmi.fetch_features, adi.return_output,
        fill_kwargs=['dataset'])
    def get_dependence_loss(self, batch_size = 512, bar = False,*,dataset):
        return self._get_dependence_loss(batch_size=batch_size,
            dataset=dataset, bar = bar)

    
    def _get_dependence_loss(self, batch_size = 512, bar =False,*, dataset):

        def dependence_loss(model, guide, **batch):

            TraceMeanField_ELBO().loss(self.model, self.guide, **batch)

            dependence_loss = -self.dependence_network(
                self.decoder.biological_signal.detach(),
                self.decoder.covariate_signal.detach(),
            ).item()

            return dependence_loss * self.dependence_beta * batch_size


        return self._evaluate_vae_loss(
                self.model, dependence_loss,
                dataset=dataset, batch_size = batch_size, bar = bar
            )

    def get_loss_fn(self):
        return TraceMeanField_ELBO().differentiable_loss


    '''def _distortion_rate_loss(self, batch_size = 512, bar = False,*,dataset):
        
        distortion, rate, vae_loss = super()._distortion_rate_loss(batch_size = batch_size, bar = bar,
                dataset = dataset)

        augmented_loss = vae_loss*self.num_exog_features + \
                self._get_dependence_loss(batch_size=batch_size, bar = bar, dataset=dataset)

        return distortion, rate, augmented_loss/self.num_exog_features'''  


    def model_step(self, batch, opt, parameters, anneal_factor = 1,
        batch_size_adjustment = 1., disentanglement_coef = 1.):

        opt.zero_grad()

        bioloss = self.get_loss_fn()(self.model, self.guide, **batch,
            anneal_factor = anneal_factor, batch_size_adjustment = batch_size_adjustment)

        dependence_loss = -self.dependence_network(
            self.decoder.biological_signal,
            self.decoder.covariate_signal,
        )

        disentangle_multiplier = torch.tensor(
                    anneal_factor * self.dependence_beta * batch_size_adjustment, 
                    requires_grad = False
                )

        loss = bioloss + disentangle_multiplier * dependence_loss

        loss.backward()
        
        opt.step()

        return loss.item(), bioloss, dependence_loss


    def dependence_step(self, batch, opt, parameters):

        opt.zero_grad()
        loss = self.dependence_network(
            self.decoder.biological_signal.detach(),
            self.decoder.covariate_signal.detach(),
        )
        loss.backward()
        
        opt.step()

        return -loss.item()
    
    
    def _get_1cycle_scheduler(self, optimizer, n_batches_per_epoch):

        return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_learning_rate, 
            epochs=self.num_epochs, steps_per_epoch=n_batches_per_epoch, 
            anneal_strategy='cos', cycle_momentum=True, 
            div_factor= self.max_learning_rate/self.min_learning_rate, three_phase=False)


    @staticmethod
    def _get_step_disentanglement_coef(step_num, *, n_epochs, n_batches_per_epoch):
        
        total_steps = n_epochs * n_batches_per_epoch
        n_cycles = 3
        #tau = ((step_num+1) % (total_steps/n_cycles))/(total_steps/n_cycles)
        
        cyclenum = (step_num + 1)//(total_steps//n_cycles)

        return 10**(-n_cycles + cyclenum+1)


    def _get_monotonic_disentanglement_coef(step_num, *, n_epochs, n_batches_per_epoch):
        return 1.

    def _step(self, batch, 
            model_optimizer, dependence_optimizer, #objective_optimizer,
            model_parameters, dependence_parameters, # objective_parameters,
            anneal_factor = 1., batch_size_adjustment = 1.,
            disentanglement_coef = 1.):
        
        total_loss, bioloss, dependence_loss = self.model_step(batch, model_optimizer, model_parameters,
                 anneal_factor = anneal_factor, batch_size_adjustment=batch_size_adjustment,
                 disentanglement_coef = disentanglement_coef)

        self.dependence_step(batch, dependence_optimizer, dependence_parameters)

        return {
            'total_loss' : total_loss,
            'ELBO_loss' : bioloss,
            'disentanglement_loss' : dependence_loss,
            'anneal_factor' : anneal_factor,
        }


    def get_model_parameters(self, data_loader):

        batch = next(iter(self.transform_batch([next(iter(data_loader))], bar=False)))

        with poutine.trace(param_only=True) as param_capture:
            self.get_loss_fn()(self.model, self.guide, **batch)
            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}

        return params, self.dependence_network.parameters()#, self.objective_network.parameters()


    @adi.wraps_modelfunc(fetch = tmi.fit_adata, 
        fill_kwargs=['features','highly_variable','dataset'])
    def get_learning_rate_bounds(self, num_epochs = 6, eval_every = 10, 
        lower_bound_lr = 1e-6, upper_bound_lr = 1,*,
        features, highly_variable, dataset):
        
        self._instantiate_model(
            features = features, highly_variable = highly_variable
        )

        data_loader = self.get_dataloader(dataset, training=True,
            batch_size = self.batch_size)

        self._get_dataset_statistics(dataset)

        n_batches = len(dataset)//self.batch_size
        eval_steps = ceil((n_batches * num_epochs)/eval_every)

        learning_rates = np.exp(
                np.linspace(np.log(lower_bound_lr), 
                np.log(upper_bound_lr), 
                eval_steps+1)
            )

        self.learning_rates = learning_rates

        def lr_function(e):
            return learning_rates[e]/learning_rates[0]

        parameters = self.get_model_parameters(data_loader)

        model_optimizer = Adam(parameters[0], lr = learning_rates[0], betas = (0.90, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_function)

        dependence_optimizer = Adam(parameters[1], lr = self.dependence_lr)
        
        optimizers = (model_optimizer, dependence_optimizer)
        batches_complete, step_loss = 0,0
        learning_rate_losses = []
        
        try:

            t = trange(eval_steps-2, desc = 'Learning rate range test', leave = True)
            _t = iter(t)

            for epoch in range(num_epochs + 1):

                self.train()
                for batch in self.transform_batch(data_loader, bar = False):

                    step_loss += self._step(batch, *optimizers, *parameters,
                            anneal_factor = 1., batch_size_adjustment = 64/self.batch_size,
                            disentanglement_coef = 1.)['total_loss']

                    batches_complete+=1
                    
                    if batches_complete % eval_every == 0 and batches_complete > 0:
                        scheduler.step()
                        learning_rate_losses.append(step_loss/(self.batch_size * self.num_exog_features))
                        step_loss = 0.0
                        try:
                            next(_t)
                        except StopIteration:
                            break

        except ValueError as err:
            print(repr(err))
            logger.error(str(err) + '\nProbably gradient overflow from too high learning rate, stopping test early.')

        self.gradient_lr = np.array(learning_rates[:len(learning_rate_losses)])
        self.gradient_loss = np.array(learning_rate_losses)

        return self.trim_learning_rate_bounds()


    def _fit(self, writer = None, training_bar = True, reinit = True,*,
            dataset, features, highly_variable):
        
        if reinit:
            self._instantiate_model(
                features = features, highly_variable = highly_variable, 
            )

        self._get_dataset_statistics(dataset)

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4, convergence_check=False)

        n_batches = len(dataset)//self.batch_size
        n_observations = len(dataset)

        data_loader = self.get_dataloader(dataset, training=True,
            batch_size = self.batch_size)

        parameters = self.get_model_parameters(data_loader)

        model_optimizer = Adam(parameters[0], lr = self.min_learning_rate, betas = (self.beta, 0.999))
        scheduler = self._get_1cycle_scheduler(model_optimizer, n_batches)
        dependence_optimizer = Adam(parameters[1], lr = self.dependence_lr)

        optimizers = (model_optimizer, dependence_optimizer)

        self.training_loss = []
        
        anneal_fn = partial(self._get_cyclic_KL_factor if self.kl_strategy == 'cyclic' else self._get_monotonic_kl_factor, 
            n_epochs = self.num_epochs, n_batches_per_epoch = n_batches)

        disentangle_fn = partial(self._get_step_disentanglement_coef, 
            n_epochs = self.num_epochs, n_batches_per_epoch = n_batches)

        step_count = 0
        t = trange(self.num_epochs, desc = 'Epoch 0', leave = True) if training_bar else range(self.num_epochs)
        _t = iter(t)
        epoch = 0

        while True:
            
            self.train()
            running_loss = 0.0
            for batch in self.transform_batch(data_loader, bar = False):
                
                anneal_factor = anneal_fn(step_count)
                disentanglement_coef = disentangle_fn(step_count)

                try:

                    metrics = self._step(batch, *optimizers, *parameters, 
                        anneal_factor = anneal_factor, batch_size_adjustment = 64/self.batch_size,
                        disentanglement_coef = disentanglement_coef)

                    metrics['learning_rate'] = float(scheduler._last_lr[0])
                        
                    if not writer is None:
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, step_count)

                    running_loss+=metrics['total_loss']
                    step_count+=1

                except ValueError:
                    raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate. Try setting a lower learning rate.')

                if epoch < self.num_epochs:
                    scheduler.step()
            
            epoch_loss = running_loss/(n_observations * self.num_exog_features)
            self.training_loss.append(epoch_loss)
            recent_losses = self.training_loss[-5:]

            if training_bar:
                t.set_description("Epoch {} done. Recent losses: {}".format(
                    str(epoch + 1),
                    ' --> '.join('{:.3e}'.format(loss) for loss in recent_losses)
                ))

            try:
                next(_t)
            except StopIteration:
                pass

            if early_stopper(recent_losses[-1]) and epoch > self.num_epochs:
                break

            epoch+=1

            yield epoch, epoch_loss

        self.set_device('cpu')
        self.eval()
        return self

    def set_device(self, device):
        super().set_device(device)
        self.dependence_network = self.dependence_network.to(device)
