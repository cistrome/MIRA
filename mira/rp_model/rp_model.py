from functools import partial
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.autoguide.initialization import init_to_mean, init_to_value, init_to_sample
from pyro import poutine
import numpy as np
import logging
from pyro.contrib.autoname import scope
from mira.topic_model.base import EarlyStopping
import warnings
from mira.rp_model.optim import LBFGS as stochastic_LBFGS
from scipy.stats import nbinom
from scipy.sparse import isspmatrix
from mira.adata_interface.rp_model import wraps_rp_func, add_isd_results, add_predictions, fetch_TSS_from_adata
import mira.adata_interface.rp_model as rpi
from mira.adata_interface.core import add_layer, wraps_modelfunc
import mira.adata_interface.core as adi
import h5py as h5
from tqdm.auto import tqdm
import os
import glob

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
    def load_dir(cls,counts_layer = None,*,expr_model, accessibility_model, prefix):
        '''
        Load directory of RP models. Adds all available RP models into a container.

        Parameters
        ----------
        expr_model: mira.topics.ExpressionTopicModel
            Trained MIRA expression topic model.
        accessibility_model : mira.topics.AccessibilityTopicModel
            Trained MIRA accessibility topic model.
        counts_layer : str, default=None
            Layer in AnnData that countains raw counts for modeling.
        prefix : str
            Prefix under which RP models were saved.

        Examples
        --------

        .. code-block :: python

            >>> litemodel = mira.rp.LITE_Model.load_dir(
            ...     counts_layer = 'counts',
            ...     expr_model = rna_model, 
            ...     accessibility_model = atac_model,
            ...     prefix = 'path/to/rpmodels/'
            ... )

        '''

        paths = glob.glob(prefix + cls.prefix + '*.pth')

        if len(paths) == 0:
            if len(glob.glob(prefix + cls.old_prefix + '*.pth')) > 0:
                logger.error('''
    Cannot load models, but found a models using older file conventions. 
    Please use "mira.rp.{}Model.convert_models(<prefix>) to convert old 
    models to the new format.
                '''.format(cls.prefix))

            raise ValueError('No models found at {}'.format(str(prefix)))

        genes = [os.path.basename(x.split('_')[-1].split('.')[0]) 
                for x in paths]

        model = cls(expr_model = expr_model, accessibility_model = accessibility_model,
                counts_layer = counts_layer, genes = genes).load(prefix)

        return model

    @classmethod
    def convert_models(cls, prefix):

        paths = glob.glob(prefix + cls.old_prefix + '*.pth')

        if len(paths) == 0:
            raise ValueError('No models found at {}'.format(str(prefix)))

        for path in tqdm(paths, desc = 'Reformatting models'):
            old_model = torch.load(path)
            old_model['guide'] = {
                old_key.replace(cls.old_prefix, cls.prefix).replace('logdistance','distance') : v
                for old_key,v in old_model['guide'].items()
            }

            gene = os.path.basename(path).split('_')[-1].split('.')[0]

            torch.save(old_model, 
                os.path.join(prefix, 
                '{}{}.pth'.format(cls.prefix, gene))
            )

            
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
        counts_layer = None,
        search_reps = 1,
        initialization_model = None):
        '''
        Parameters
        ----------

        expr_model: mira.topics.ExpressionTopicModel
            Trained MIRA expression topic model.
        accessibility_model : mira.topics.AccessibilityTopicModel
            Trained MIRA accessibility topic model.
        genes : np.ndarray[str], list[str]
            List of genes for which to learn RP models.
        learning_rate : float>0
            Learning rate for L-BGFS optimizer.
        counts_layer : str, default=None
            Layer in AnnData that countains raw counts for modeling.
        initialization_model : mira.rp.LITE_Model, mira.rp.NITE_Model, None
            Initialize parameters of RP model using the provided model before
            further optimization with L-BGFS. This is used when training the NITE
            model, which is initialized with the LITE model parameters learned 
            for the same genes, then retrained to optimized the NITE model's 
            extra parameters. This procedure speeds training.

        Attributes
        ----------
        genes : np.ndarray[str]
            Array of gene names for models
        features : np.ndarray[str]
            Array of gene names for models
        models : list[mira.rp.GeneModel]
            List of trained RP models
        model_type : {"NITE", "LITE"}
        
        Examples
        --------

        Setup requires RNA and ATAC AnnData objects with shared cell barcodes
        and trained topic models for both modes:

        .. code-block:: python
            
            >>> rp_args = dict(expr_adata = rna_data, atac_adata = atac_data)
        
        Instantiating a LITE model (local chromatin accessibility only):

        .. code-block:: python

            >>> litemodel = mira.rp.LITE_Model(
            ...     expr_model = rna_model, 
            ...     accessibility_model = atac_model,
            ...     counts_layer = 'counts',
            ...     genes = ['LEF1','WNT3','EDA','NOTCH1'],
            ... )
            >>> litemodel.fit(**rp_args)
        
        Instantiating a NITE model (local chromatin accessibility only):

            >>> nitemodel = mira.rp.NITE_Model(
            ...     expr_model = rna_model, 
            ...     accessibility_model = atac_model,
            ...     counts_layer = 'counts',
            ...     genes = litemodel.genes,
            ...     instantiation_model = litemodel
            ... )
            >>> nitemodel.fit(**rp_args)
        
        '''

        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.learning_rate = learning_rate
        self.counts_layer = counts_layer

        assert(isinstance(search_reps, int) and search_reps > 0)
        self.search_reps = search_reps

        self.models = []
        for gene in genes:

            init_params = None
            try:
                init_params = initialization_model.get_model(gene).posterior_map
            except (IndexError, AttributeError):
                pass

            self.models.append(
                GeneModel(
                    gene = gene, 
                    learning_rate = learning_rate, 
                    use_NITE_features = self.use_NITE_features,
                    init_params= init_params,
                    search_reps = search_reps,
                )
            )

    def subset(self, genes):
        '''
        Return a subset container of RP models.

        Parameters
        ----------

        genes : np.ndarray[str], list[str]
            List of genes to subset from RP model

        Examples
        --------

        .. code-block :: python

            >>> less_models = litemodel.subset(['LEF1','WNT3'])

        
        '''
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
        '''
        Merge RP models from two model containers.

        Parameters
        ----------

        rp_model : mira.rp.LITE_Model, mira.rp.NITE_Model
            RP model container from which to append new RP models

        Examples
        --------

        .. code-block :: python

            >>> model1.genes
            ... ['LEF1','WNT3']
            >>> model2.genes
            ... ['CTSC','EDAR']
            >>> merged_model = model1.join(model2)
            >>> merged_model.genes
            ... ['LEF1','WNT3','CTSC','EDAR']

        '''

        assert(isinstance(rp_model, BaseModel))
        assert(rp_model.use_NITE_features == self.use_NITE_features), 'Cannot join LITE model with NITE model'

        add_models = np.setdiff1d(rp_model.genes, self.genes)

        for add_gene in add_models:
            self.models.append(
                rp_model.get_model(add_gene)
            )
        
        return self

    def __getitem__(self, gene):
        '''
        Alias for `get_model(gene)`.

        Examples
        --------

        >>> rp_model["LEF1"]
        ... <mira.rp_model.rp_model.GeneModel at 0x7fa07af1cf10>

        '''
        return self.get_model(gene)

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
        '''
        Save RP models.

        Parameters
        ----------

        prefix : str
            Prefix under which to save RP models. May be filename prefix
            or directory. RP models will save with format:
            **{prefix}_{LITE/NITE}_{gene}.pth**

        '''
        for model in self.models:
            model.save(prefix)


    def get_model(self, gene):
        '''
        Gets model for gene

        Parameters
        ----------

        gene : str
            Fetch RP model for this gene

        '''
        try:
            return self.models[np.argwhere(self.genes == gene)[0,0]]
        except IndexError:
            raise IndexError('Model for gene {} does not exist'.format(gene))


    def load(self, prefix):
        '''
        Load RP models saved with *prefix*.

        Parameters
        ----------

        prefix : str
            Prefix under which RP models were saved.

        '''

        genes = self.genes
        self.models = []
        for gene in genes:
            try:
                model = GeneModel(gene = gene, use_NITE_features = self.use_NITE_features)
                model.load(prefix)
                self.models.append(model)
            except FileNotFoundError:
                old_filename = prefix + self.old_prefix + gene + '.pth'
                if os.path.isfile(old_filename):
                    logger.warn('''
    Cannot load {} model, but found a model using older file conventions: {}. 
    Please use "mira.rp.{}_Model.convert_models(<prefix>) to convert old 
    models to the new format.
                    '''.format(
                        gene, old_filename, self.model_type
                    ))
                else:
                    logger.warn('Cannot load {} model. File not found.'.format(gene))

        if len(self.models) == 0:
            raise ValueError('No models loaded.')

        return self

    def subset_fit_models(self, models):

        self.models = []
        for model in models:
            if not model.was_fit:
                logger.warn('{} model failed to fit.'.format(model.gene))
            else:
                self.models.append(model)

        return self

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs : self.subset_fit_models(output), bar_desc = 'Fitting models')
    def fit(self, model, features, callback = None):
        '''
        Optimize parameters of RP models to learn *cis*-regulatory relationships.

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with 
            mira.tl.get_distance_to_TSS.

        Returns
        -------

        rp_model : mira.rp.LITE_Model, mira.rp.NITE_Model
            RP model with optimized parameters
 
        '''
        try:
            model.fit(features)
        except ValueError:
            pass

        if not callback is None:
            callback(model)

        return model

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: np.array(output).sum(), bar_desc = 'Scoring')
    def score(self, model, features):
        return model.score(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: \
        add_predictions(expr_adata, (self.features, output), model_type = self.model_type, sparse = True),
        bar_desc = 'Predicting expression')
    def predict(self, model, features):
        '''
        Predicts the expression of genes given their *cis*-accessibility state.
        Also evaluates the probability of that prediction for LITE/NITE evaluation.

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with 
            mira.tl.get_distance_to_TSS.

        Returns
        -------

        anndata.AnnData
            `.layers['LITE_prediction']` or `.layers['NITE_prediction']`: np.ndarray[float] of shape (n_cells, n_features)
                Predicted relative frequencies of features using LITE or NITE model, respectively
            `.layers['LTIE_logp']` or `.layers['NITE_logp']` : np.ndarray[float] of shape (n_cells, n_features)
                Probability of observed expression given posterior predictive estimate of LITE or
                NITE model, respectively.
        
        '''
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
    def probabilistic_isd(self, model, features, n_samples = 1500, checkpoint = None,
        *,hits_matrix, metadata):
        '''
        For each gene, calcuate association scores with each transcription factor.
        Association scores detect when a TF binds within *cis*-regulatory
        elements (CREs) that are influential to expression predictions for that gene.
        CREs that influence the RP model expression prediction are nearby a 
        gene's TSS and have accessibility that correlates with expression. This
        model assumes these attributes indicate a factor is more likely to 
        regulate a gene. 

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with TSS and factor
            binding data using mira.tl.get_distance_to_TSS **and** 
            mira.tl.get_motif_hits_in_peaks/mira.tl.get_CHIP_hits_in_peaks.
        n_samples : int>0, default=1500
            Downsample cells to this amount for calculations. Speeds up computation
            time. Cells are sampled by stratifying over expression levels.
        checkpoint : str, default = None
            Path to checkpoint h5 file. pISD calculations can be slow, and saving
            a checkpoint ensures progress is not lost if calculations are 
            interrupted. To resume from a checkpoint, just pass the path to the h5.

        Returns
        -------

        anndata.AnnData
            `.varm['motifs-prob_deletion']` or `.varm['chip-prob_deletion']`: np.ndarray[float] of shape (n_genes, n_factors)
                Association scores for each gene-TF combination. Higher scores indicate
                greater predicted association/regulatory influence.

        '''

        already_calculated = False
        if not checkpoint is None:
            if not os.path.isfile(checkpoint):
                h5.File(checkpoint, 'w').close()

            with h5.File(checkpoint, 'r') as h:
                try:
                    h[model.gene]
                    already_calculated = True
                except KeyError:
                    pass

        if checkpoint is None or not already_calculated:
            result = model.probabilistic_isd(features, hits_matrix, n_samples = n_samples)

            if not checkpoint is None:
                with h5.File(checkpoint, 'a') as h:
                    g = h.create_group(model.gene)
                    g.create_dataset('samples_mask', data = result[1])
                    g.create_dataset('isd', data = result[0])

            return result
        else:
            with h5.File(checkpoint, 'r') as h:
                g = h[model.gene]
                result = g['isd'][...], g['samples_mask'][...]

            return result

    @property
    def parameters_(self):
        '''
        Returns parameters of all contained RP models.
        '''
        return {
            gene : self[gene].parameters_
            for gene in self.features
        }


class LITE_Model(BaseModel):

    use_NITE_features = False
    prefix = 'LITE_'
    old_prefix = 'cis_'

    def __init__(self,*, expr_model, accessibility_model, genes, learning_rate = 1, 
        counts_layer = None, initialization_model = None, search_reps = 1):
        '''
        Container for multiple regulatory potential (RP) LITE models. LITE models
        learn a relationship between a gene's expression and accessibility in 
        nearby *cis*-regulatory elements (CRE). The MIRA model assumes the regulatory
        influence of a CRE on a gene decays with respect to distance from that
        gene. MIRA learns this distance using variational Bayesian inference. 

        With a trained RP model, one may assess the 

        * LITE/NITE characteristics of a gene: whether that gene's expression is decoupled from changes in local chromatin.
        * Chromatin differential: the relative levels of nearby accessibility versus gene expression.
        * *Insilico*-deletion: predicts transcription factor regulators based on a model of nearby binding in influential CREs, as determined by the RP model.
        
        '''
        
        super().__init__(
            expr_model = expr_model, 
            accessibility_model = accessibility_model, 
            genes = genes,
            learning_rate = learning_rate,
            initialization_model = initialization_model,
            counts_layer=counts_layer,
            search_reps = search_reps,
        )

    def spawn_NITE_model(self):
        '''
        Returns a NITE model seeded with the LITE model's
        parameters.
        '''
        return NITE_Model(
            expr_model= self.expr_model,
            accessibility_model=self.accessibility_model,
            genes = self.genes,
            learning_rate=self.learning_rate,
            counts_layer=self.counts_layer,
            initialization_model=self,
            search_reps=self.search_reps
        )

class NITE_Model(BaseModel):

    use_NITE_features = True
    prefix = 'NITE_'
    old_prefix = 'trans_'

    def __init__(self,*, expr_model, accessibility_model, genes, learning_rate = 1, 
        counts_layer = None, initialization_model = None, search_reps = 1):
        '''
        Container for multiple regulatory potential (RP) NITE models. NITE models
        learn a relationship between a gene's expression and accessibility in 
        nearby *cis*-regulatory elements (CRE), **and** the cell-wide chromatin landscape. 

        The predictive capacity of local vs. cell-wide chromatin in predicting a gene's
        expression state determines a gene's *NITE Score*, and edulicates whether that
        gene is primarily regulated by local or nonlocal mechanisms.

        '''

        super().__init__(
            expr_model = expr_model, 
            accessibility_model = accessibility_model, 
            genes = genes,
            learning_rate = learning_rate,
            initialization_model = initialization_model,
            counts_layer=counts_layer,
            search_reps = search_reps,
        )
        


class GeneModel:
    '''
    Gene-level RP model object.
    '''

    def __init__(self,*,
        gene, 
        learning_rate = 1.,
        use_NITE_features = False,
        init_params = None,
        search_reps = 1,
    ):
        self.gene = gene
        self.learning_rate = learning_rate
        self.use_NITE_features = use_NITE_features
        self.was_fit = False
        self.search_reps= search_reps
        self.init_params = init_params

    def _get_weights(self, loading = False, seed = None):

        if not seed is None:
            pyro.set_rng_seed(seed)

        pyro.clear_param_store()
        self.bn = torch.nn.BatchNorm1d(1, momentum = 1.0, affine = False)

        if self.init_params is None:
            if self.use_NITE_features and not loading:
                logger.warn('\nTraining NITE regulation model for {} without providing pre-trained LITE models may cause divergence in statistical testing.'\
                    .format(self.gene))

            if seed is None:
                self.guide = AutoDelta(self.model, init_loc_fn = init_to_mean)
            else:
                self.guide = AutoDelta(self.model, init_loc_fn = init_to_sample)
        else:
            self.seed_params = {self.prefix + '/' + k.split('/')[-1] : v.detach().clone() for k,v in self.init_params.items()}
            self.guide = AutoDelta(self.model, init_loc_fn = mean_default_init_to_value(values = self.seed_params))


    def get_prefix(self):
        return ('NITE' if self.use_NITE_features else 'LITE') + '_' + self.gene

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
                d = pyro.sample('distance', dist.LogNormal(np.log(15), 1.2))

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

                pyro.deterministic('mu', mu)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            def find_init_point(seed):
                
                self._get_weights(seed = seed)

                with poutine.trace(param_only=True) as param_capture:
                    loss = self.get_loss_fn()(self.model, self.guide, **features)

                return seed, loss

            best_seed = None
            if self.search_reps > 1:
                best_seed = sorted(map(find_init_point, [None, *range(self.search_reps - 1)]), key = lambda x : x[1])[0][0]

            self._get_weights(seed = best_seed)

            N = len(features['upstream_weights'])

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


    def get_posterior_sample(self, features):

        features = {k : self._t(v) for k, v in features.items()}
        self.bn.eval()

        guide = AutoDelta(self.model, init_loc_fn = init_to_value(values = self.posterior_map))
        guide_trace = poutine.trace(guide).get_trace(**features)
        #print(guide_trace)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace))\
            .get_trace(**features)

        return model_trace


    @property
    def prefix(self):
        return self.get_prefix()


    def __getitem__(self, gene):
         return self.get_model(gene)


    def predict(self, features):

        trace = self.get_posterior_sample(features)
        
        expression_prediction = self.to_numpy(
            trace.nodes[self.prefix + '/prediction']['value'])[:, np.newaxis]

        logp_data = self._get_logp(features['gene_expr'], trace)

        return expression_prediction, logp_data


    def score(self, features):

        trace = self.get_posterior_sample(features)

        return self._get_logp(features['gene_expr'], trace).sum()


    def _get_logp(self, gene_expr, trace):

        p = trace.nodes[self.prefix + '/prob_success']['value']
        theta = self.posterior_map[self.prefix + '/theta']

        logp = dist.NegativeBinomial(total_count = theta, probs = p)\
                .log_prob(self._t(gene_expr))
        logp_data = self.to_numpy(logp)[:, np.newaxis]

        return logp_data


    def get_logp(self, features):
        raise DeprecationWarning('As of MIRA 0.2.0, "get_logp" is deprecated and no longer necessary.The "predict" method now returns logp(data) information')
        

    @staticmethod
    def to_numpy(X):
        return X.clone().detach().cpu().numpy()

    def get_savename(self, prefix):
        return prefix + self.prefix + '.pth'

    def _get_save_data(self):
        return dict(bn = self.bn.state_dict(), guide = self.posterior_map)

    def _load_save_data(self, state):
        self._get_weights(loading = True)
        self.bn.load_state_dict(state['bn'])
        self.posterior_map = state['guide']

    def save(self, prefix):
        torch.save(self._get_save_data(), self.get_savename(prefix))

    def load(self, prefix):
        state = torch.load(self.get_savename(prefix))
        self._load_save_data(state)

    def _get_normalized_params(self):
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

        f_Z = params['a'][0] * RP(upstream_weights, upstream_distances, params['distance'][0]) \
        + params['a'][1] * RP(downstream_weights, downstream_distances, params['distance'][1]) \
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
        return logp_summary[0] - logp_summary[1:]#, f_Z, expression, logp_data


    def probabilistic_isd(self, features, hits_matrix, n_samples = 1500, n_bins = 20):
        
        np.random.seed(2556)
        N = len(features['gene_expr'])
        informative_samples = self._select_informative_samples(features['gene_expr'], 
            n_bins = n_bins, n_samples = n_samples)
        
        
        for k in 'gene_expr,upstream_weights,downstream_weights,promoter_weights,softmax_denom,read_depth,NITE_features'.split(','):
            features[k] = features[k][informative_samples]

        samples_mask = np.zeros(N)
        samples_mask[informative_samples] = 1
        samples_mask = samples_mask.astype(bool)
        
        return self._prob_ISD(
            hits_matrix, **features, 
            params = self._get_normalized_params(), 
            bn_eps= self.bn.eps
        ), samples_mask


    def _get_RP_model_coordinates(self, scale_height = False, bin_size = 50,
        decay_periods = 20, promoter_width = 3000, *,
        gene_chrom, gene_start, gene_end, gene_strand):

        assert(isinstance(promoter_width, int) and promoter_width > 0)
        assert(isinstance(decay_periods, int) and decay_periods > 0)
        assert(isinstance(bin_size, int) and bin_size > 0)
        assert(scale_height in [True, False])

        rp_params = self._get_normalized_params()
            
        upstream, downstream = 1e3 * rp_params['distance']

        left_decay, right_decay, start_pos = upstream, downstream, gene_start
        left_a, promoter_a, right_a = rp_params['a']

        if gene_strand == '-':
            left_decay, right_decay, start_pos = downstream, upstream, gene_end
            right_a, promoter_a, left_a = rp_params['a']
        
        left_extent = int(decay_periods*left_decay)
        left_x = np.linspace(1, left_extent, left_extent//bin_size).astype(int)
        left_y = 0.5**(left_x / left_decay) * (left_a if scale_height else 1.)

        right_extent = int(decay_periods*right_decay)
        right_x = np.linspace(0, right_extent, right_extent//bin_size).astype(int)
        right_y = 0.5**(right_x / right_decay) * (right_a if scale_height else 1.)

        left_x = -left_x[::-1] - promoter_width//2 + start_pos
        right_x = right_x + promoter_width//2 + start_pos
        promoter_x = [-promoter_width//2 + start_pos]
        promoter_y = [promoter_a if scale_height else 1.]

        x = np.concatenate([left_x, promoter_x, right_x])
        y = np.concatenate([left_y[::-1], promoter_y, right_y])

        return x, y

    @property
    def parameters_(self):
        '''
        Returns maximum a posteriori estimate of RP model parameters
        as dictionary dict[str : parameter, float : value].

        '''
        norm_params = self._get_normalized_params()

        params = {
            k : np.atleast_1d(v)[0]
            for k, v in norm_params.items() if len(np.atleast_1d(v)) == 1
        }

        params['a_upstream'] = norm_params['a'][0]
        params['a_promoter'] = norm_params['a'][1]
        params['a_downstream'] = norm_params['a'][2]

        params['distance_upstream'] = norm_params['distance'][0]
        params['distance_downstream'] = norm_params['distance'][1]

        if self.use_NITE_features:
            params.update({
                'a-NITE_' + str(i) : v
                for i, v in enumerate(norm_params['a_NITE'])
            })

        return params

    @adi.wraps_modelfunc(fetch_TSS_from_adata, 
        fill_kwargs = ['gene_chrom','gene_start','gene_end','gene_strand'])
    def write_bedgraph(self, scale_height = False, bin_size = 50,
        decay_periods = 20, promoter_width = 3000,*, save_name,
        gene_chrom, gene_start, gene_end, gene_strand):
        '''
        Write bedgraph of RP model coverage. Useful for visualization with 
        Bedtools.

        Parameters
        ----------

        adata : anndata.AnnData
            AnnData object with TSS data annotated by `mira.tl.get_distance_to_TSS`.
        save_name : str
            Path to saved bedgraph file.
        scale_height : boolean, default = False
            Write RP model tails proportional in height to their respective
            multiplicative coeffecient. Useful for evaluating not only the distance
            of predicted regulatory influence, but the weighted importance of regions 
            in terms of predicting expression.
        decay_periods : int>0, default = 10
            Number of decay periods to write.
        promoter_width : int>0, default = 0
            Width of flat region at promoter of gene in base pairs (bp). MIRA default is 3000 bp.

        Returns
        -------

        None

        '''

        coord, value = self._get_RP_model_coordinates(scale_height = scale_height, bin_size = bin_size,
            decay_periods = decay_periods, promoter_width = promoter_width,
            gene_chrom = gene_chrom, gene_start = gene_start, gene_end = gene_end, gene_strand = gene_strand)

        with open(save_name, 'w') as f:
            for start, end, val in zip(coord[:-1], coord[1:], value):
                print(gene_chrom, start, end, val, sep = '\t', end = '\n', file = f)


    @adi.wraps_modelfunc(rpi.fetch_get_influential_local_peaks, rpi.return_peaks_by_idx,
        fill_kwargs=['peak_idx','tss_distance'])
    def get_influential_local_peaks(self, peak_idx, tss_distance, decay_periods = 5):
        '''
        Returns the `.var` field of the adata, but subset for only peaks within 
        the local chromatin neighborhood of a gene. The local chromatin neighborhood
        is defined by the decay distance parameter for that gene's RP model.

        Parameters
        ----------

        adata : anndata.AnnData
            AnnData object with ATAC features and TSS annotations.
        decay_periods : int > 0, default = 5
            Return peaks that are within `decay_periods*upstream_decay_distance` upstream
            of gene and `decay_periods*downstream_decay_distance` downstream of gene,
            where upstream and downstream decay distances are given by the parameters
            of the RP model.

        Returns
        -------

        pd.DataFrame : 
            
            subset from `adata.var` to include only features/peaks within
            the gene's local chromatin neighborhood. This function adds two columns:

            `distance_to_TSS` : int
                Distance, in base pairs, from the gene's TSS
            `is_upstream` : boolean
                If peak is upstream or downstream of gene

        '''

        assert isinstance(decay_periods, (int, float)) and decay_periods > 0

        downstream_mask = (tss_distance >= 0) \
                & (tss_distance < (decay_periods * 1e3 * self.parameters_['distance_downstream']))

        upstream_mask = (tss_distance < 0) \
                & (np.abs(tss_distance) < (decay_periods * 1e3 * self.parameters_['distance_upstream']))

        combined_mask = upstream_mask | downstream_mask

        return peak_idx[combined_mask], tss_distance[combined_mask]