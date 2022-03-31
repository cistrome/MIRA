from mira.topic_model.expression_model import ExpressionEncoder, ExpressionTopicModel
import pyro.distributions as dist
import torch
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
    def model(self,*,endog_features, exog_features, covariates, read_depth, extra_features, anneal_factor = 1.):
        pyro.module("decoder", self.decoder)
        
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
            extra_features, anneal_factor = 1.):
        pyro.module("encoder", self.encoder)
        
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