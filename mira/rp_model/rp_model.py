from functools import partial
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_mean, init_to_value
from pyro import poutine
import numpy as np
import torch.distributions.constraints as constraints
import logging
from math import ceil
import time
from pyro.contrib.autoname import scope
from sklearn.base import BaseEstimator
from mira.topic_model.base import EarlyStopping
#from kladiv2.core.adata_interface import *
import warnings
from mira.rp_model.optim import LBFGS as stochastic_LBFGS
import torch.nn.functional as F
import gc
import argparse
from scipy.stats import nbinom
from scipy.sparse import isspmatrix
from mira.adata_interface.rp_model import wraps_rp_func, add_isd_results
from mira.adata_interface.core import add_layer
from collections.abc import Sequence

logger = logging.getLogger(__name__)

class SaveCallback:

    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, model):
        if model.was_fit:
            model.save(self.prefix)

def mean_default_init_to_value(
    site = None,
    values = {},
    *,
    fallback = init_to_mean,
):
    if site is None:
        return partial(mean_default_init_to_value, values=values, fallback=fallback)

    if site["name"] in values:
        return values[site["name"]]
    if fallback is not None:
        return fallback(site)
    raise ValueError(f"No init strategy specified for site {repr(site['name'])}")


class BaseModel:

    @classmethod
    def _make(cls, expr_model, accessibility_model, counts_layer, models, learning_rate, use_NITE_features):
        self = BaseModel.__new__(cls)
        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.learning_rate = learning_rate
        self.use_NITE_features = use_NITE_features
        self.counts_layer = counts_layer
        self.models = models

        return self

    def __init__(self,*,
        expr_model, 
        accessibility_model, 
        genes,
        learning_rate = 1,
        use_NITE_features = False,
        counts_layer = None,
        initialization_model = None):

        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.learning_rate = learning_rate
        self.use_NITE_features = use_NITE_features
        self.counts_layer = counts_layer

        if use_NITE_features and initialization_model is None:
            logger.warn('Training global regulation models without providing pre-trained LITE models may cause divergence in statistical testing.')

        self.models = []
        for gene in genes:

            init_params = None
            try:
                init_params = initialization_model.get_model(gene).posterior_map
            except (IndexError, AttributeError):
                if use_NITE_features:
                    logger.warn('No initialization model found for gene {}.'.format(gene))

            self.models.append(
                GeneModel(
                    gene = gene, 
                    learning_rate = learning_rate, 
                    use_NITE_features = use_NITE_features,
                    init_params= init_params
                )
            )

    def subset(self, genes):
        assert(isinstance(genes, (list, np.ndarray)))
        for gene in genes:
            if not gene in self.genes:
                raise ValueError('Gene {} is not in RP model'.format(str(gene)))        

        return self._make(
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model, counts_layer=self.counts_layer, 
            learning_rate = self.learning_rate, use_NITE_features = self.use_NITE_features,
            models = [model for model in self.models if model.gene in genes]
        )

    def join(self, rp_model):

        assert(isinstance(rp_model, BaseModel))
        assert(rp_model.use_NITE_features == self.use_NITE_features), 'Cannot join LITE model with NITE model'

        add_models = np.setdiff1d(rp_model.genes, self.genes)

        for add_gene in add_models:
            self.models.append(
                rp_model.get_model(add_gene)
            )
        
        return self

    @property
    def genes(self):
        return np.array([model.gene for model in self.models])

    @property
    def features(self):
        return self.genes

    @property
    def model_type(self):
        if self.use_NITE_features:
            return 'NITE'
        else:
            return 'LITE'

    def _get_masks(self, tss_distance):
        promoter_mask = np.abs(tss_distance) <= 1500
        upstream_mask = np.logical_and(tss_distance < 0, ~promoter_mask)
        downstream_mask = np.logical_and(tss_distance > 0, ~promoter_mask)

        return promoter_mask, upstream_mask, downstream_mask


    def _get_region_weights(self, NITE_features, softmax_denom, idx):
        
        model = self.accessibility_model

        def bn(x):
            return (x - model._get_bn_mean()[idx])/np.sqrt(model._get_bn_var()[idx] + model.decoder.bn.eps)

        rate = model._get_gamma()[idx] * bn(NITE_features.dot(model._get_beta()[:, idx])) + model._get_bias()[idx]

        region_probabilities = np.exp(rate)/softmax_denom[:, np.newaxis]

        return region_probabilities

    
    def _get_features_for_model(self,*, gene_expr, read_depth, expr_softmax_denom, NITE_features, atac_softmax_denom, 
        upstream_idx, downstream_idx, promoter_idx, upstream_distances, downstream_distances, include_factor_data = False):

        features = dict(
            gene_expr = gene_expr,
            read_depth = read_depth,
            softmax_denom = expr_softmax_denom,
            NITE_features = NITE_features,
            upstream_distances = upstream_distances,
            downstream_distances = downstream_distances,
        )
        
        if include_factor_data:
            features.update(dict(
                promoter_idx = promoter_idx,
                upstream_idx = upstream_idx,
                downstream_idx = downstream_idx
            ))

        for k, idx in zip(['upstream_weights', 'downstream_weights', 'promoter_weights'],
                    [upstream_idx, downstream_idx, promoter_idx]):

            features[k] = self._get_region_weights(NITE_features, atac_softmax_denom, idx) * 1e4
        return features


    def save(self, prefix):
        for model in self.models:
            model.save(prefix)


    def get_model(self, gene):
        try:
            return self.models[np.argwhere(self.genes == gene)[0,0]]
        except IndexError:
            raise IndexError('Model for gene {} does not exist'.format(gene))


    def load(self, prefix):

        genes = self.genes
        self.models = []
        for gene in genes:
            try:
                model = GeneModel(gene = gene, use_NITE_features = self.use_NITE_features)
                model.load(prefix)
                self.models.append(model)
            except FileNotFoundError:
                logger.warn('Cannot load {} model. File not found.'.format(gene))

        return self

    def subset_fit_models(self, was_fit):
        self.models = [model for fit, model in zip(was_fit, self.models) if fit]
        return self

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs : self.subset_fit_models(output), bar_desc = 'Fitting models')
    def fit(self, model, features, callback = None):
        try:
            model.fit(features)
        except ValueError:
            logger.warn('{} model failed to fit.'.format(model.gene))
            pass

        if not callback is None:
            callback(model)

        return model.was_fit

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: np.array(output).sum(), bar_desc = 'Scoring')
    def score(self, model, features):
        return model.score(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: \
        add_layer(expr_adata, (self.features, np.hstack(output)), add_layer = self.model_type + '_prediction', sparse = True),
        bar_desc = 'Predicting expression')
    def predict(self, model, features):
        return model.predict(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: \
        add_layer(expr_adata, (self.features, np.hstack(output)), add_layer = self.model_type + '_logp', sparse = True),
        bar_desc = 'Getting logp(Data)')
    def get_logp(self, model, features):
        return model.get_logp(features)

    '''@wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: \
        add_layer(expr_adata, (self.features, np.hstack(output)), add_layer = self.model_type + '_samples', sparse = True)
    )
    def _sample_posterior(self, model, features, site = 'prediction'):
        return model.to_numpy(model.get_posterior_sample(features, site))[:, np.newaxis]'''

    @wraps_rp_func(lambda self, expr_adata, atac_adata, output, **kwargs : output, bar_desc = 'Formatting features')
    def get_features(self, model, features):
        return features

    @wraps_rp_func(lambda self, expr_adata, atac_adata, output, **kwargs : output, 
        bar_desc = 'Formatting features', include_factor_data = True)
    def get_isd_features(self, model, features,*,hits_matrix, metadata):
        return features

    @wraps_rp_func(add_isd_results, 
        bar_desc = 'Predicting TF influence', include_factor_data = True)
    def probabilistic_isd(self, model, features,*,hits_matrix, metadata):
        return model.probabilistic_isd(features, hits_matrix)


class LITE_Model(BaseModel):

    def __init__(self,*, expr_model, accessibility_model, genes, learning_rate = 1, counts_layer = None):
        super().__init__(
            expr_model = expr_model, 
            accessibility_model = accessibility_model, 
            genes = genes,
            learning_rate = learning_rate,
            use_NITE_features = False, 
            initialization_model = None,
            counts_layer=counts_layer,
        )

class NITE_Model(BaseModel):

    def __init__(self,*, expr_model, accessibility_model, genes, learning_rate = 1, 
        counts_layer = None, initialization_model = None):
        super().__init__(
            expr_model = expr_model, 
            accessibility_model = accessibility_model, 
            genes = genes,
            learning_rate = learning_rate,
            use_NITE_features = True, 
            initialization_model = initialization_model,
            counts_layer=counts_layer,
        )


class GeneModel:

    def __init__(self,*,
        gene, 
        learning_rate = 1.,
        use_NITE_features = False,
        init_params = None
    ):
        self.gene = gene
        self.learning_rate = learning_rate
        self.use_NITE_features = use_NITE_features
        self.was_fit = False
        self.init_params = init_params

    def _get_weights(self):
        pyro.clear_param_store()
        self.bn = torch.nn.BatchNorm1d(1, momentum = 1.0, affine = False)

        if self.init_params is None:
            self.guide = AutoDelta(self.model, init_loc_fn = init_to_mean)
        else:
            self.seed_params = {self.prefix + '/' + k.split('/')[-1] : v.detach().clone() for k,v in self.init_params.items()}
            self.guide = AutoDelta(self.model, init_loc_fn = mean_default_init_to_value(values = self.seed_params))

    def get_prefix(self):
        return ('NITE' if self.use_NITE_features else 'cis') + '_' + self.gene

    def RP(self, weights, distances, d):
        return (weights * torch.pow(0.5, distances/(1e3 * d))).sum(-1)

    def model(self, 
        gene_expr, 
        softmax_denom,
        read_depth,
        upstream_weights,
        upstream_distances,
        downstream_weights,
        downstream_distances,
        promoter_weights,
        NITE_features):

        with scope(prefix = self.get_prefix()):

            with pyro.plate("spans", 3):
                a = pyro.sample("a", dist.HalfNormal(1.))

            with pyro.plate("upstream-downstream", 2):
                d = pyro.sample('logdistance', dist.LogNormal(np.log(15), 1.2))

            if self.use_NITE_features and hasattr(self, 'seed_params'):
                theta = self.seed_params[self.prefix + '/theta']
                theta.requires_grad = False
            else:
                theta = pyro.sample('theta', dist.Gamma(2., 0.5))
            
            gamma = pyro.sample('gamma', dist.LogNormal(0., 0.5))
            bias = pyro.sample('bias', dist.Normal(0, 5.))

            if self.use_NITE_features:
                with pyro.plate('NITE_coefs', NITE_features.shape[-1]):
                    a_NITE = pyro.sample('a_NITE', dist.Normal(0.,1.))

            with pyro.plate('data', len(upstream_weights)):

                f_Z = a[0] * self.RP(upstream_weights, upstream_distances, d[0])\
                    + a[1] * self.RP(downstream_weights, downstream_distances, d[1]) \
                    + a[2] * promoter_weights.sum(-1)

                if self.use_NITE_features:
                    f_Z = f_Z + torch.matmul(NITE_features, torch.unsqueeze(a_NITE, 0).T).reshape(-1)

                prediction = self.bn(f_Z.reshape((-1,1)).float()).reshape(-1)
                pyro.deterministic('unnormalized_prediction', prediction)
                independent_rate = (gamma * prediction + bias).exp()

                rate =  independent_rate/softmax_denom
                pyro.deterministic('prediction', rate)
                mu = read_depth.exp() * rate

                p = mu / (mu + theta)

                pyro.deterministic('prob_success', p)
                NB = dist.NegativeBinomial(total_count = theta, probs = p)
                pyro.sample('obs', NB, obs = gene_expr)


    def _t(self, X):
        return torch.tensor(X, requires_grad=False)

    @staticmethod
    def get_loss_fn():
        return TraceMeanField_ELBO().differentiable_loss

    def get_optimizer(self, params):
        #return torch.optim.LBFGS(params, lr=self.learning_rate, line_search_fn = 'strong_wolfe')
        return stochastic_LBFGS(params, lr = self.learning_rate, history_size = 5,
            line_search = 'Armijo')


    def get_loss_and_grads(self, optimizer, features):
        
        optimizer.zero_grad()

        loss = self.get_loss_fn()(self.model, self.guide, **features)
        loss.backward()

        grads = optimizer._gather_flat_grad()

        return loss, grads

    def armijo_step(self, optimizer, features, update_curvature = True):

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_fn()(self.model, self.guide, **features)
            return loss

        obj_loss, grad = self.get_loss_and_grads(optimizer, features)

        # compute initial gradient and objective
        p = optimizer.two_loop_recursion(-grad)
        p/=torch.norm(p)
    
        # perform line search step
        options = {'closure': closure, 'current_loss': obj_loss, 'interpolate': True}
        obj_loss, lr, _, _, _, _ = optimizer.step(p, grad, options=options)

        # compute gradient
        obj_loss.backward()
        grad = optimizer._gather_flat_grad()

        # curvature update
        if update_curvature:
            optimizer.curvature_update(grad, eps=0.2, damping=True)

        return obj_loss.detach().item()

    def fit(self, features):

        features = {k : self._t(v) for k, v in features.items()}

        self._get_weights()
        N = len(features['upstream_weights'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with poutine.trace(param_only=True) as param_capture:
                loss = self.get_loss_fn()(self.model, self.guide, **features)

            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}
            optimizer = self.get_optimizer(params)
            early_stopper = EarlyStopping(patience = 3, tolerance = 1e-4)
            update_curvature = False

            self.loss = []
            self.bn.train()
            for i in range(100):

                self.loss.append(
                    float(self.armijo_step(optimizer, features, update_curvature = update_curvature)/N)
                )
                update_curvature = not update_curvature

                if early_stopper(self.loss[-1]):
                    break

        self.was_fit = True
        self.posterior_map = self.guide()

        if self.use_NITE_features and hasattr(self, 'seed_params'):
            theta_name = self.prefix + '/theta'
            self.posterior_map[theta_name] = self.seed_params[theta_name]

        del optimizer
        del features
        del self.guide

        return self


    def get_posterior_sample(self, features, site):

        features = {k : self._t(v) for k, v in features.items()}
        self.bn.eval()

        guide = AutoDelta(self.model, init_loc_fn = init_to_value(values = self.posterior_map))
        guide_trace = poutine.trace(guide).get_trace(**features)
        #print(guide_trace)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace))\
            .get_trace(**features)

        return model_trace.nodes[self.prefix + '/' + site]['value']

    @property
    def prefix(self):
        return self.get_prefix()


    def predict(self, features):
        return self.to_numpy(self.get_posterior_sample(features, 'prediction'))[:, np.newaxis]


    def score(self, features):
        return self.get_logp(features).sum()


    def get_logp(self, features):

        features = {k : self._t(v) for k, v in features.items()}

        p = self.get_posterior_sample(features, 'prob_success')
        theta = self.posterior_map[self.prefix + '/theta']

        logp = dist.NegativeBinomial(total_count = theta, probs = p).log_prob(features['gene_expr'])
        return self.to_numpy(logp)[:, np.newaxis]

    @staticmethod
    def to_numpy(X):
        return X.clone().detach().cpu().numpy()

    def get_savename(self, prefix):
        return prefix + self.prefix + '.pth'

    def _get_save_data(self):
        return dict(bn = self.bn.state_dict(), guide = self.posterior_map)

    def _load_save_data(self, state):
        self._get_weights()
        self.bn.load_state_dict(state['bn'])
        self.posterior_map = state['guide']

    def save(self, prefix):
        torch.save(self._get_save_data(), self.get_savename(prefix))

    def load(self, prefix):
        state = torch.load(self.get_savename(prefix))
        self._load_save_data(state)


    def get_normalized_params(self):
        d = {
            k[len(self.prefix) + 1:] : v.detach().cpu().numpy()
            for k, v in self.posterior_map.items()
        }
        d['bn_mean'] = self.to_numpy(self.bn.running_mean)
        d['bn_var'] = self.to_numpy(self.bn.running_var)
        d['bn_eps'] = self.bn.eps
        return d

    @staticmethod
    def _select_informative_samples(expression, n_bins = 20, n_samples = 1500, seed = 2556):
        '''
        Bin based on contribution to overall expression, then take stratified sample to get most informative cells.
        '''
        np.random.seed(seed)

        expression = np.ravel(expression)
        assert(np.all(expression >= 0))

        expression = np.log1p(expression)
        expression += np.mean(expression)

        sort_order = np.argsort(-expression)

        cummulative_counts = np.cumsum(expression[sort_order])
        counts_per_bin = expression.sum()/(n_bins - 1)
        
        samples_per_bin = n_samples//n_bins
        bin_num = cummulative_counts//counts_per_bin
        
        differential = 0
        informative_samples = []
        samples_taken = 0
        for _bin, _count in zip(*np.unique(bin_num, return_counts = True)):
            
            if _bin == n_bins - 1:
                take_samples = n_samples - samples_taken
            else:
                take_samples = samples_per_bin + differential

            if _count < take_samples:
                informative_samples.append(
                    sort_order[bin_num == _bin]
                )
                differential = take_samples - _count
                samples_taken += _count

            else:
                differential = 0
                samples_taken += take_samples
                informative_samples.append(
                    np.random.choice(sort_order[bin_num == _bin], size = take_samples, replace = False)
                )

        return np.concatenate(informative_samples)


    @staticmethod
    def _prob_ISD(hits_matrix,*, upstream_weights, downstream_weights, 
        promoter_weights, upstream_idx, promoter_idx, downstream_idx,
        upstream_distances, downstream_distances, read_depth, 
        softmax_denom, gene_expr, NITE_features, params, bn_eps):

        assert(isspmatrix(hits_matrix))
        assert(len(hits_matrix.shape) == 2)
        num_factors = hits_matrix.shape[0]

        def tile(x):
            x = np.expand_dims(x, -1)
            return np.tile(x, num_factors+1).transpose((0,2,1))

        def delete_regions(weights, region_mask):
            
            num_regions = len(region_mask)
            hits = 1 - hits_matrix[:, region_mask].toarray().astype(int) #1, factors, regions
            hits = np.vstack([np.ones((1, num_regions)), hits])
            hits = hits[np.newaxis, :, :].astype(int)

            return np.multiply(weights, hits)

        upstream_weights = delete_regions(tile(upstream_weights), upstream_idx) #cells, factors, regions
        promoter_weights = delete_regions(tile(promoter_weights), promoter_idx)
        downstream_weights = delete_regions(tile(downstream_weights), downstream_idx)

        read_depth = read_depth[:, np.newaxis]
        softmax_denom = softmax_denom[:, np.newaxis]

        upstream_distances = upstream_distances[np.newaxis, np.newaxis, :]
        downstream_distances = downstream_distances[np.newaxis,np.newaxis, :]
        expression = gene_expr[:, np.newaxis]

        def RP(weights, distances, d):
            return (weights * np.power(0.5, distances/(1e3 * d))).sum(-1)

        f_Z = params['a'][0] * RP(upstream_weights, upstream_distances, params['logdistance'][0]) \
        + params['a'][1] * RP(downstream_weights, downstream_distances, params['logdistance'][1]) \
        + params['a'][2] * promoter_weights.sum(-1) # cells, factors

        original_data = f_Z[:,0]
        sorted_first_col = np.sort(original_data).reshape(-1)
        quantiles = np.argsort(f_Z, axis = 0).argsort(0)

        f_Z = sorted_first_col[quantiles]
        f_Z[:,0] = original_data

        #f_Z = (f_Z - f_Z[:,0].mean(0,keepdims = True))/np.sqrt(f_Z[:, 0].var(0, keepdims = True) + bn_eps)
        f_Z = (f_Z - params['bn_mean'])/np.sqrt(params['bn_var'] + bn_eps)

        indep_rate = np.exp(params['gamma'] * f_Z + params['bias'])
        compositional_rate = indep_rate/softmax_denom

        mu = np.exp(read_depth) * compositional_rate

        p = mu / (mu + params['theta'])

        logp_data = nbinom(params['theta'], 1 - p).logpmf(expression)
        logp_summary = logp_data.sum(0)
        return logp_summary[0] - logp_summary[1:], f_Z, expression, logp_data


    def probabilistic_isd(self, features, hits_matrix, n_samples = 1500, n_bins = 20):
        
        np.random.seed(2556)
        N = len(features['gene_expr'])
        informative_samples = self._select_informative_samples(features['gene_expr'], 
            n_bins = n_bins, n_samples = n_samples)
        

        '''informative_samples = np.random.choice(
            N, size = 1500, replace = False,
        )'''
        
        for k in 'gene_expr,upstream_weights,downstream_weights,promoter_weights,softmax_denom,read_depth,NITE_features'.split(','):
            features[k] = features[k][informative_samples]

        samples_mask = np.zeros(N)
        samples_mask[informative_samples] = 1
        samples_mask = samples_mask.astype(bool)
        
        return (*self._prob_ISD(
            hits_matrix, **features, 
            params = self.get_normalized_params(), 
            bn_eps= self.bn.eps
        ), samples_mask)