from mira.topic_model.modality_mixins.expression_model \
    import ExpressionEncoder
from mira.topic_model.modality_mixins.accessibility_model import DANEncoder, DANSkipEncoder, LSIEncoder, \
    ZeroPaddedBinaryMultinomial, ZeroPaddedMultinomial

#from mira.topic_model.generative_models.lda_generative \
#    import ExpressionDirichletModel, AccessibilityDirichletModel

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
from mira.topic_model.base import encoder_layer, logger


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


class DP_EncoderMixin:

    def topic_comps(self, X, read_depth, covariates, extra_features):

        alpha = self.forward(X, read_depth, covariates, extra_features)[0]
        vi = torch.sigmoid(alpha)

        theta = mix_weights(vi[:,:-1]).detach().cpu().numpy()

        return theta


    def sample_posterior(self, X, read_depth, covariates, extra_features,
            n_samples = 100):

        theta_loc, theta_scale = self.forward(X, read_depth, covariates, extra_features)[:2]
        # theta = z*std + mu
        alpha = torch.randn(*theta_loc.shape, n_samples)*theta_scale[:,:,None] + theta_loc[:,:,None]
        vi = torch.sigmoid(alpha)
        theta = mix_weights(torch.transpose(vi, -2,-1)[:,:,:-1])

        theta = torch.transpose(theta, -2, -1)
        
        return theta.detach().cpu().numpy().mean(-1)


class DPModel:

    def _get_save_data(self):
        save_data = super()._get_save_data()
        save_data['weights']['stick_len'] = self.stick_len

        return save_data


    def _set_weights(self, fit_params, weights):
        
        stick_len = weights.pop('stick_len')
        super()._set_weights(fit_params, weights)

        self._stick_len = stick_len


    @adi.wraps_modelfunc(tmi.fetch_topic_comps, 
        partial(adi.add_obsm, add_key = 'X_umap_features'),
        fill_kwargs=['topic_compositions', 'covariates','extra_features'])
    def predict_num_topics(self, min_contribution = 0.05,*, 
            topic_compositions, covariates, extra_features):

        return np.sum(
            topic_compositions.max(0) > min_contribution
        )


    @property
    def stick_len(self):
        try:
            return self._stick_len
        except AttributeError:
            alpha = pyro.get_param_store()['alpha_a']/pyro.get_param_store()['alpha_b']
            self._stick_len = (1 - 1/(1+alpha)).cpu().detach().numpy()

        return self._stick_len
    

    def _recommend_num_topics(self, n_samples):
        #boxcox transform of the number of samples
        return max(int(
                self.boxcox(n_samples, 0.3)
            ), 50)
    

    @staticmethod
    def boxcox(x, a):
        return ( x**a - 1)/a


    @staticmethod
    def get_active_topics(topic_compositions, min_contribution = 0.05):
        dead_topics = topic_compositions.max(0) < min_contribution

        return ~dead_topics
    

    @adi.wraps_modelfunc(tmi.fetch_topic_comps, 
        partial(adi.add_obsm, add_key = 'X_umap_features'),
        fill_kwargs=['topic_compositions', 'covariates','extra_features'])
    def get_umap_features(self, box_cox = 0.5, min_contribution = 0.05,*, 
            topic_compositions, covariates, extra_features):

        active_topics = self.get_active_topics(topic_compositions, min_contribution)
        num_topics = active_topics.sum()

        logger.info('Found {} topics from the data.'.format(num_topics))

        basis = gram_schmidt_basis(num_topics)

        topic_compositions = topic_compositions[:, active_topics]
        transformed = topic_compositions/np.power(self.stick_len, np.arange(num_topics))[np.newaxis, :]

        return self.boxcox(transformed, box_cox).dot(basis)

    def _t(self, val):
        return torch.tensor(val, requires_grad = False, device = self.device)



class ExpressionDirichletProcessModel(DPModel):
    
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


    @scope(prefix='rna')
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

        self.stick_len

    
class AccessibilityDirichletProcessModel(DPModel):
    
    @property
    def encoder_model(self):
        base_encoder_class = super().encoder_model

        return  type(
                'DP_' + base_encoder_class.__name__,
                (base_encoder_class, DP_EncoderMixin),
                {}
            )

    @scope(prefix= 'atac')
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
                    
                    
                theta = mix_weights(theta[:,:-1])
                peak_probs = self.decoder(theta, covariates)
                
                if self.count_model == 'binary':
                    pyro.sample(
                        'obs', ZeroPaddedBinaryMultinomial(total_count = 1, probs = peak_probs), obs = exog_features,
                    )
                else:
                    pyro.sample(
                        'obs', ZeroPaddedMultinomial(probs = peak_probs, validate_args = False), obs = (exog_features, endog_features),
                    )



    @scope(prefix= 'atac')
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
                
                theta_loc, theta_scale = self.encoder(endog_features, read_depth, covariates, extra_features)

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample(
                        "theta", dist.TransformedDistribution(
                            dist.Normal(theta_loc, theta_scale), [SigmoidTransform()]
                        ).to_event(1)
                    )

        self.stick_len