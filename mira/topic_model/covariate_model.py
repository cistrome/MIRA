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
from mira.topic_model.mine import WassersteinDual as DependencyModel
from pyro import poutine
from functools import partial


class CovariateModelMixin(BaseModel):


    def _get_weights(self, on_gpu = True, inference_mode = False):
        super()._get_weights(on_gpu=on_gpu, inference_mode=inference_mode)

        if self.covariate_compensation:
            self.dependence_network = DependencyModel(DependencyModel.get_statistics_network(
                2*self.num_exog_features, DependencyModel.hidden)
            ).to(self.device)


    def get_loss_fn(self):
        return TraceMeanField_ELBO().differentiable_loss


    def model_step(self, batch, opt, anneal_factor = 1):

        opt.zero_grad()

        bioloss = self.get_loss_fn()(self.model, self.guide, **batch)

        dependence_loss = -self.dependence_network(
            self.decoder.biological_signal,
            self.decoder.covariate_signal,
        )

        loss = bioloss + torch.tensor(anneal_factor, requires_grad = False) * DependencyModel.loss_beta * dependence_loss        
        loss.backward()

        opt.step()

        return loss.item()


    def dependence_step(self, batch, opt):

        opt.zero_grad()
        loss = self.dependence_network(
            self.decoder.biological_signal.detach(),
            self.decoder.covariate_signal.detach(),
        )
        loss.backward()

        return -loss.item()
    
    
    def _get_1cycle_scheduler(self, optimizer, n_batches_per_epoch):

        return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_learning_rate, 
            epochs=self.num_epochs, steps_per_epoch=n_batches_per_epoch, 
            anneal_strategy='cos', cycle_momentum=False, 
            div_factor= self.max_learning_rate/self.min_learning_rate, three_phase=False)


    def _step(self, batch, model_optimizer, dependence_optimizer, anneal_factor = 1.):
        
        return {
            'topic_model_loss' : self.model_step(batch, model_optimizer, anneal_factor = anneal_factor),
            'dependence_loss' : self.dependence_step(batch, dependence_optimizer),
            'anneal_factor' : anneal_factor,
        }


    def get_model_parameters(self, data_loader):

        batch = next(iter(data_loader))

        with poutine.trace(param_only=True) as param_capture:
            
            self.get_loss_fn()(self.model, self.guide, **batch)
                    
            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}

        return params, self.dependence_network.parameters()


    @adi.wraps_modelfunc(fetch = tmi.fit_adata, 
        fill_kwargs=['features','highly_variable','dataset'])
    def get_learning_rate_bounds(self, num_epochs = 6, eval_every = 10, 
        lower_bound_lr = 1e-6, upper_bound_lr = 1,*,
        features, highly_variable, dataset):
        
        self._instantiate_model(
            features = features, highly_variable = highly_variable
        )

        data_loader = self.get_dataloader(dataset, training=True)

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

        topic_model_params, dependence_model_params = self._get_model_parameters(data_loader)

        model_optimizer = Adam(topic_model_params, lr = learning_rates[0], betas = (0.90, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_function)

        dependence_optimizer = Adam(dependence_model_params, lr = DependencyModel.lr)

        batches_complete, step_loss = 0,0
        learning_rate_losses = []
        
        try:

            t = trange(eval_steps-2, desc = 'Learning rate range test', leave = True)
            _t = iter(t)

            for epoch in range(num_epochs + 1):

                self.train()
                for batch in self.transform_batch(data_loader, bar = False):

                    step_loss += self._step(batch, model_optimizer, dependence_optimizer, anneal_factor = 1.)['topic_model_loss']
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


    def _fit(self,*,training_bar = True, reinit = True,
            features, highly_variable, endog_features, exog_features, covariates,
            extra_features):
        
        if reinit:
            self._instantiate_model(
                features = features, highly_variable = highly_variable, 
                endog_features = endog_features, exog_features = exog_features,
                covariates = covariates, extra_features = extra_features,
            )

        self._get_dataset_statistics(endog_features, exog_features, 
            covariates, extra_features)

        n_observations = endog_features.shape[0]
        n_batches = self.get_num_batches(n_observations, self.batch_size)

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4, convergence_check=False)

        #scheduler = self._get_1cycle_scheduler(n_batches)
        #self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        self.training_loss, self.testing_loss, self.num_epochs_trained = [],[],0
        
        anneal_fn = partial(self._get_cyclic_KL_factor if self.kl_strategy == 'cyclic' else self._get_monotonic_kl_factor, n_epochs = self.num_epochs, 
            n_batches_per_epoch = n_batches)

        step_count = 0
        self.anneal_factors, self.dependence_losses = [],[]
        self.mine_norms, self.model_norms = [],[]
        try:

            t = trange(self.num_epochs, desc = 'Epoch 0', leave = True) if training_bar else range(self.num_epochs)
            _t = iter(t)
            epoch = 0
            while True:
                
                self.train()
                running_loss = 0.0
                for minibatch in self._iterate_batches(endog_features = endog_features, 
                    exog_features = exog_features, covariates = covariates, extra_features = extra_features,
                        batch_size = self.batch_size, bar = False):
                    
                    anneal_factor = anneal_fn(step_count)
                    self.anneal_factors.append(anneal_factor)
                    #if batch[0].shape[0] > 1:
                    try:
                        if step_count == 0:

                            with poutine.trace(param_only=True) as param_capture:
                                _ = self.get_loss_fn()(self.model, self.guide, **minibatch)
                        
                            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}

                            model_optimizer = Adam(params, lr = self.min_learning_rate, betas = (self.beta, 0.999))
                            scheduler = self._get_1cycle_scheduler(model_optimizer, n_batches)

                            if self.covariate_compensation:
                                mine_optimizer = Adam(
                                    self.dependence_network.parameters(), lr = DependencyModel.lr,
                                )

                        step_loss, step_norm = self.model_step(model_optimizer, params, **minibatch, anneal_factor = anneal_factor)
                        running_loss+=float(step_loss)
                        self.model_norms.append(step_norm)

                        dependence_loss, mine_norm = self.dependence_step(mine_optimizer)
                        self.dependence_losses.append(float(dependence_loss))
                        self.mine_norms.append(mine_norm)
                        step_count+=1
                    except ValueError:
                        raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate. Try setting a lower learning rate.')

                    if epoch < self.num_epochs:
                        scheduler.step()
                
                #epoch cleanup
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

        except KeyboardInterrupt:
            logger.warn('Interrupted training.')

        self.set_device('cpu')
        self.eval()
        return self

    def set_device(self, device):
        super().set_device(device)
        self.dependence_network = self.dependence_network.to(device)
