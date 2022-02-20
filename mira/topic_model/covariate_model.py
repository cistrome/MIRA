from mira.topic_model.base import BaseModel
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
from mira.topic_model.mine import Mine, get_statistics_network
from pyro import poutine


class CovariateModel(BaseModel):

    mine_lr = 1e-4
    mine_alpha = 0.01
    mine_hidden = 100
    MI_beta = 1

    def _get_weights(self, on_gpu = True, inference_mode = False):
        super()._get_weights(on_gpu=on_gpu, inference_mode=inference_mode)

        if self.self.covariate_compensation:
            self.mine_network = get_statistics_network(
                2*self.num_exog_features, self.mine_hidden)

    def get_loss_fn(self):
        return TraceMeanField_ELBO.differentiable_loss

    def get_MINE_loss(self):


    def model_step(self, opt, *args, **kwargs):

        opt.zero_grad()

        bioloss = self.get_loss_fn()(self.model, self.guide, *args, **kwargs)
        
        causal_MI = -self.mine_network(
            self.decoder.biological_effect,
            self.decoder.covariates_effect,
        )

        loss = bioloss + self.MI_beta * causal_MI
        loss.backward()
        opt.step()

        return loss.item()

    def mine_step(self, opt):

        opt.zero_grad()
        loss = self.min_network(
            self.decoder.biological_effect.detach(),
            self.decoder.covariates_effect.detach(),
        )
        loss.backward()
        opt.step()
        

    @adi.wraps_modelfunc(fetch = tmi.fit_adata, 
        fill_kwargs=['features','highly_variable','endog_features','exog_features', 
        'covariates','extra_features'])
    def get_learning_rate_bounds(self, num_epochs = 6, eval_every = 10, 
        lower_bound_lr = 1e-6, upper_bound_lr = 1,*,
        features, highly_variable, endog_features, exog_features, covariates,
        extra_features):

        self._instantiate_model(
            features = features, highly_variable = highly_variable, 
            endog_features = endog_features, exog_features = exog_features,
            covariates = covariates, extra_features = extra_features
        )

        self._get_dataset_statistics(endog_features, exog_features, covariates, extra_features)

        n_batches = self.get_num_batches(endog_features.shape[0], self.batch_size)

        eval_steps = ceil((n_batches * num_epochs)/eval_every)

        learning_rates = np.exp(
                np.linspace(np.log(lower_bound_lr), 
                np.log(upper_bound_lr), 
                eval_steps+1))

        self.learning_rates = learning_rates

        def lr_function(e):
            return learning_rates[e]/learning_rates[0]

        batches_complete, steps_complete, step_loss, samples_seen = 0,0,0,0
        learning_rate_losses = []
        
        try:
            t = trange(eval_steps-2, desc = 'Learning rate range test', leave = True)
            _t = iter(t)

            for epoch in range(num_epochs + 1):

                #train step
                self.train()
                for minibatch in self._iterate_batches(endog_features = endog_features, 
                        exog_features = exog_features, covariates = covariates, extra_features = extra_features,
                        batch_size = self.batch_size, bar = False):

                    if steps_complete == 0:
                        
                        with poutine.trace(param_only=True) as param_capture:
                            loss = self.get_loss_fn()(self.model, self.guide, **minibatch)
                    
                        params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}

                        model_optimizer = Adam(params, lr = learning_rates[0], betas = (0.90, 0.999))
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            model_optimizer, lr_function)

                        if self.covariate_compensation:
                            mine_optimizer = Adam(
                                self.mine_network.parameters(), lr = self.mine_lr,
                            )

                    step_loss += self.model_step(model_optimizer, **minibatch, anneal_factor = 1.)
                    self.mine_step(mine_optimizer)

                    batches_complete+=1
                    samples_seen += minibatch['endog_features'].shape[0]
                    
                    if batches_complete % eval_every == 0 and batches_complete > 0:
                        steps_complete+=1
                        scheduler.step()
                        learning_rate_losses.append(step_loss/(samples_seen * self.num_exog_features))
                        step_loss, samples_seen = 0.0, 0
                        try:
                            next(_t)
                        except StopIteration:
                            break

        except ValueError as err:
            logger.error(str(err) + '\nProbably gradient overflow from too high learning rate, stopping test early.')

        self.gradient_lr = np.array(learning_rates[:len(learning_rate_losses)])
        self.gradient_loss = np.array(learning_rate_losses)

        return self.trim_learning_rate_bounds()