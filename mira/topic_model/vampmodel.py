
import pyro.distributions as dist
import torch
from torch import nn
from pyro import poutine
from pyro.contrib.autoname import scope
import pyro
from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform
import torch.nn.functional as F
from mira.topic_model.ilr_tools import gram_schmidt_basis
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
import numpy as np
from functools import partial
from mira.topic_model.base import encoder_layer, get_fc_stack
from mira.topic_model.base import Decoder as LinearDecoder

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


class NonlinDecoder(LinearDecoder):
    
    def __init__(self, covariates_hidden = 32,
        covariates_dropout = 0.05, mask_dropout = 0.05,*,
        num_exog_features, num_topics, num_covariates, topics_dropout):
        super().__init__(
            covariates_hidden = covariates_hidden,
            covariates_dropout = covariates_dropout, 
            mask_dropout = mask_dropout,
            num_exog_features = num_exog_features, 
            num_topics = num_topics, 
            num_covariates = num_covariates, 
            topics_dropout = topics_dropout
        )

        self.beta = get_fc_stack(
            layer_dims = [num_topics, 128, num_exog_features],
            dropout = topics_dropout, skip_nonlin = True,
        )


class ExpressionVampModel:

    _decoder_model = NonlinDecoder
    
    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, covariates, read_depth, extra_features, 
        anneal_factor = 1., batch_size_adjustment = 1.):
        
        pyro.module("decoder", self.decoder)
        n_mix = 512
        gamma_a, gamma_b = 1.,1.

        with poutine.scale(None, batch_size_adjustment):

            dispersion = pyro.param('dispersion', read_depth.new_ones(self.num_exog_features).to(self.device) * 5., constraint = constraints.positive)
            dispersion = dispersion.to(self.device)

            alpha = pyro.sample('alpha', dist.Gamma(self._t(1.), self._t(1.)))
            #print(alpha)
            pi = pyro.sample(
                        "pi", dist.Beta(dispersion.new_ones(n_mix), 
                                            dispersion.new_ones(n_mix) * alpha).to_event(1)
                        )

            pi = mix_weights(pi)[:-1][:, None]

            pseudoinputs = pyro.param('pseudoinput', torch.randn(n_mix, self.hidden, device = self.device))

            q_pseudo_mu, q_pseudo_std = self.encoder.pseudoinputs(pseudoinputs) # n_mix, n_latent

            with pyro.plate("cells", endog_features.shape[0]):

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample('Z', dist.Normal(
                                (pi*q_pseudo_mu).sum(0), torch.sqrt((pi*q_pseudo_std**2).sum(0))
                                ).to_event(1)
                            )
                    
                    read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))
                    
                expr_rate = self.decoder(theta, covariates)
                
                logits = (read_scale * expr_rate).log() - (dispersion).log()
                X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, logits = logits).to_event(1), obs = exog_features)


    @scope(prefix= 'rna')
    def guide(self,*,endog_features, exog_features, covariates, read_depth, 
            extra_features, anneal_factor = 1., batch_size_adjustment = 1.):
        pyro.module("encoder", self.encoder)
        n_mix = 512
        gamma_a, gamma_b = 3.,3.

        with poutine.scale(None, batch_size_adjustment):
            
            alpha_a = pyro.param('alpha_a', torch.tensor(gamma_a, device = self.device), 
                                constraint=constraints.positive)
            alpha_b = pyro.param('alpha_b', torch.tensor(gamma_b, device = self.device), 
                                constraint=constraints.positive)

            alpha = pyro.sample('alpha', dist.Gamma(alpha_a, alpha_b))

            pi_mu = pyro.param('pi_mu', torch.randn(n_mix, device = self.device))
            pi_std = pyro.param('pi_std', torch.ones(n_mix, device = self.device),
                        constraint=constraints.positive)

            pi = pyro.sample(
                        "pi", dist.TransformedDistribution(
                            dist.Normal(pi_mu, pi_std), [SigmoidTransform()]
                        ).to_event(1)
                    )

            with pyro.plate("cells", endog_features.shape[0]):
                
                theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(endog_features, read_depth, covariates, extra_features)

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample(
                        "Z", dist.Normal(theta_loc, theta_scale).to_event(1)
                    )

                    read_depth = pyro.sample(
                        "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                    )

    def _t(self, val):
        return torch.tensor(val, requires_grad = False, device = self.device)


    @adi.wraps_modelfunc(tmi.fetch_features,
        fill_kwargs=['dataset'])
    def predict(self, batch_size = 512, bar = True,*, dataset):

        return self._run_encoder_fn(
                lambda *x : self.encoder.forward(*x)[0].detach().cpu().numpy(), 
                dataset, batch_size = batch_size, bar = bar)