from mira.topic_model.expression_model import ExpressionEncoder, ExpressionTopicModel
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
from mira.topic_model.base import encoder_layer


class DeepDecoder(nn.Module):
    
    def __init__(self, covar_channels = 32,*,
        num_exog_features, num_topics, num_covariates, dropout):
        super().__init__()
        #self.beta = nn.Linear(num_topics, num_exog_features, bias = False)
        #self.bn = nn.BatchNorm1d(num_exog_features)
        
        dropout_rate = 1 - np.sqrt(1-dropout)
        self.drop = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.num_topics = num_topics
        self.num_covariates = num_covariates

        if num_covariates > 0:

            self.batch_effect_model = nn.Sequential(
                encoder_layer(num_topics + num_covariates, covar_channels, 
                    dropout=dropout/2, nonlin=True),
                nn.Linear(covar_channels, num_exog_features),
                nn.BatchNorm1d(num_exog_features, affine = False),
            )
            if num_covariates > 0:
                self.batch_effect_gamma = nn.Parameter(
                    torch.zeros(num_exog_features)
                )

    def forward(self, theta, covariates, nullify_covariates = False):
        
        #self.theta = theta
        
        X = self.drop(theta)

        self.covariate_signal = self.get_batch_effect(X, covariates, 
            nullify_covariates = nullify_covariates)

        self.biological_signal = self.get_biological_effect(self.drop2(X))

        return F.softmax(self.biological_signal + self.covariate_signal, dim=1)


    def get_biological_effect(self, theta):
        return self.bn(self.beta(theta))


    def get_batch_effect(self, theta, covariates, nullify_covariates = False):
        
        if self.num_covariates == 0 or nullify_covariates: 
            batch_effect = theta.new_zeros(1)
            batch_effect.requires_grad = False
        else:
            batch_effect = self.batch_effect_gamma * self.batch_effect_model(
                    torch.hstack([theta, covariates])
                )

        return batch_effect


    def get_softmax_denom(self, theta, covariates):

        return (self.get_biological_effect(theta) + self.get_batch_effect(theta, covariates)).exp().sum(-1)


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


class DP_EncoderMixin:

    def topic_comps(self, X, read_depth, covariates, extra_features):

        alpha = self.forward(X, read_depth, covariates, extra_features)[0]
        vi = torch.sigmoid(alpha)

        theta = mix_weights(vi[:,:-1]).detach().cpu().numpy()

        return theta


class DP_ExpressionEncoder(ExpressionEncoder, DP_EncoderMixin):
    pass

    
class ExpressionDirichletProcessModel(ExpressionTopicModel):
    
    encoder_model = DP_ExpressionEncoder
    
    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, covariates, read_depth, extra_features, 
        anneal_factor = 1., batch_size_adjustment = 1.):
        pyro.module("decoder", self.decoder)
        
        with poutine.scale(None, batch_size_adjustment):

            alpha = pyro.sample('alpha', dist.Gamma(self._t(2.), self._t(0.5)))
            
            dispersion = pyro.param('dispersion', read_depth.new_ones(self.num_exog_features).to(self.device) * 5., constraint = constraints.positive)
            dispersion = dispersion.to(self.device)

            with pyro.plate("cells", endog_features.shape[0]):

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample(
                        "theta", dist.Beta(dispersion.new_ones(self.num_topics), 
                                            dispersion.new_ones(self.num_topics) * alpha).to_event(1))
                    
                    read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))
                    
                theta = mix_weights(theta[:,:-1])
                expr_rate = self.decoder(theta, covariates)
                
                if not self.nb_parameterize_logspace:
                    mu = torch.multiply(read_scale, expr_rate)
                    probs = mu/(mu + dispersion)
                    X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, probs = probs).to_event(1), obs = exog_features)
                else:
                    logits = (read_scale * expr_rate).log() - (dispersion).log()
                    X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, logits = logits).to_event(1), obs = exog_features)


    @scope(prefix= 'rna')
    def guide(self,*,endog_features, exog_features, covariates, read_depth, 
            extra_features, anneal_factor = 1., batch_size_adjustment = 1.):
        pyro.module("encoder", self.encoder)

        with poutine.scale(None, batch_size_adjustment):
        
            alpha_a = pyro.param('alpha_a', torch.tensor(2., device = self.device), 
                                constraint=constraints.positive)
            alpha_b = pyro.param('alpha_b', torch.tensor(0.5, device = self.device), 
                                constraint=constraints.positive)
            alpha = pyro.sample('alpha', dist.Gamma(alpha_a, alpha_b))

            with pyro.plate("cells", endog_features.shape[0]):
                
                theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(endog_features, read_depth, covariates, extra_features)

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample(
                        "theta", dist.TransformedDistribution(
                            dist.Normal(theta_loc, theta_scale), [SigmoidTransform()]
                        ).to_event(1)
                    )

                    read_depth = pyro.sample(
                        "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                    )
                
                
    @staticmethod
    def _get_monotonic_kl_factor(step_num, *, n_epochs, n_batches_per_epoch):
        
        total_steps = n_epochs * n_batches_per_epoch
        return min(1., (step_num + 1)/(total_steps * 1/3 + 1))
    

    @property
    def stick_len(self):
        alpha = pyro.get_param_store()['alpha_a']/pyro.get_param_store()['alpha_b']
        return (1 - 1/(1+alpha)).cpu().detach().numpy()
    
    
    @staticmethod
    def boxcox(x, a):
        return ( x**a - 1)/a
    
    
    @adi.wraps_modelfunc(tmi.fetch_topic_comps, partial(adi.add_obsm, add_key = 'X_umap_features'),
        fill_kwargs=['topic_compositions', 'covariates','extra_features'])
    def get_umap_features(self, num_topics = 20, box_cox = 0.5,*, 
            topic_compositions, covariates, extra_features):
        
        basis = gram_schmidt_basis(num_topics)

        topic_compositions = topic_compositions[:, :num_topics]
        transformed = topic_compositions/np.power(self.stick_len, np.arange(num_topics))[np.newaxis, :]

        return self.boxcox(transformed, box_cox).dot(basis)
    
    def _t(self, val):
        return torch.tensor(val, requires_grad = False, device = self.device)