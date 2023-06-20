
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import numpy as np
import torch
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
from mira.topic_model.modality_mixins.accessibility_model \
    import ZeroPaddedBinaryMultinomial

from mira.topic_model.generative_models.dirichlet_process \
    import ExpressionDirichletProcessModel, AccessibilityDirichletProcessModel



class DirichletMarginals:

    def _get_dp_model(self):

        if isinstance(self, ExpressionDirichletModel):
            generative_model = ExpressionDirichletProcessModel
        else:
            generative_model = AccessibilityDirichletProcessModel

        return self._spawn_submodel(generative_model)
    

    def get_topic_model(self):
        return self


    def model(self):
        
        pyro.module("decoder", self.decoder)

        _alpha, _beta = self._get_gamma_parameters(self.initial_pseudocounts, self.num_topics)
        with pyro.plate("topics", self.num_topics):
            initial_counts = pyro.sample("a", dist.Gamma(self._to_tensor(_alpha), self._to_tensor(_beta)))

        theta_loc = self._get_prior_mu(initial_counts, self.K)
        theta_scale = self._get_prior_std(initial_counts, self.K)

        return theta_loc.to(self.device), theta_scale.to(self.device)


    def guide(self):

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(
            *self._get_gamma_moments(self.initial_pseudocounts, self.num_topics)
        )

        pseudocount_mu = pyro.param('pseudocount_mu', _counts_mu * torch.ones((self.num_topics,)).to(self.device))

        pseudocount_std = pyro.param('pseudocount_std', np.sqrt(_counts_var) * torch.ones((self.num_topics,)).to(self.device), 
                constraint = constraints.positive)

        pyro.module("encoder", self.encoder)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(pseudocount_mu, pseudocount_std))


    @staticmethod
    def _get_gamma_parameters(I, K):
        return 2., 2*K/I

    @staticmethod
    def _get_gamma_moments(I,K):
        return I/K, 0.5 * (I/K)**2

    @staticmethod
    def _get_lognormal_parameters_from_moments(m, v):
        m_squared = m**2
        mu = np.log(m_squared / np.sqrt(v + m_squared))
        var = np.log(v/m_squared + 1)

        return mu, var

    @staticmethod
    def _get_prior_mu(a, K):
        return a.log() - 1/K * torch.sum(a.log())

    @staticmethod
    def _get_prior_std(a, K):
        return torch.sqrt(1/a * (1-2/K) + 1/(K * a))


class ExpressionDirichletModel(DirichletMarginals):

    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, covariates, read_depth, extra_features, 
        anneal_factor = 1., batch_size_adjustment = 1.):

        with poutine.scale(None, batch_size_adjustment):
            theta_loc, theta_scale = super().model()
            pyro.module("decoder", self.decoder)

            dispersion = pyro.param('dispersion', read_depth.new_ones(self.num_exog_features).to(self.device) * 5., constraint = constraints.positive)
            dispersion = dispersion.to(self.device)

            with pyro.plate("cells", endog_features.shape[0]):

                with poutine.scale(None, anneal_factor):
                    
                    theta = pyro.sample(
                        "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                    )

                    read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))

                theta = theta/theta.sum(-1, keepdim = True)
                expr_rate = self.decoder(theta, covariates)
                
                if not self.nb_parameterize_logspace:
                    mu = torch.multiply(read_scale, expr_rate)
                    probs = mu/(mu + dispersion)
                    X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, probs = probs).to_event(1), obs = exog_features)
                else:
                    logits = (read_scale * expr_rate).log() - (dispersion).log()
                    X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, logits = logits).to_event(1), obs = exog_features)


    @scope(prefix= 'rna')
    def guide(self,*,endog_features, exog_features, covariates, 
            read_depth, extra_features, anneal_factor = 1., batch_size_adjustment = 1.):

        with poutine.scale(None, batch_size_adjustment):
            super().guide()

            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(endog_features, read_depth, covariates, extra_features)
            
            with pyro.plate("cells", endog_features.shape[0]):

                with poutine.scale(None, anneal_factor):

                    theta = pyro.sample(
                        "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                    )

                    read_depth = pyro.sample(
                        "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                    )


class AccessibilityDirichletModel(DirichletMarginals):

    @scope(prefix='atac')
    def model(self,*, endog_features, exog_features, 
        read_depth, covariates, extra_features, anneal_factor = 1.,
        batch_size_adjustment = 1.):

        with poutine.scale(None, batch_size_adjustment):

            theta_loc, theta_scale = super().model()
            
            with pyro.plate("cells", endog_features.shape[0]):

                with poutine.scale(None, anneal_factor):
                    theta = pyro.sample(
                        "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                    )

                theta = theta/theta.sum(-1, keepdim = True)            
                peak_probs = self.decoder(theta, covariates)
                
                #if self.count_model == 'binary':
                #    #print('this')
                pyro.sample(
                    'obs', ZeroPaddedBinaryMultinomial(total_count = 1, probs = peak_probs), obs = exog_features,
                )
                #else:
                    #print('here')
                #    pyro.sample(
                #        'obs', ZeroPaddedMultinomial(probs = peak_probs, validate_args = False), obs = (exog_features, endog_features),
               #     )

    @scope(prefix = 'atac')
    def guide(self, *, endog_features, exog_features, read_depth, covariates, 
        extra_features, anneal_factor = 1.,
        batch_size_adjustment = 1.):

        with poutine.scale(None, batch_size_adjustment):
            super().guide()
        
            with pyro.plate("cells", endog_features.shape[0]):
                
                theta_loc, theta_scale = self.encoder(endog_features, read_depth, covariates, extra_features)

                with poutine.scale(None, anneal_factor):
                        
                    theta = pyro.sample(
                        "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                    )