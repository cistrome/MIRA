
import torch
from torch._C import Value
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from pyro.nn import PyroParam
import pyro.distributions as dist
from mira.topic_model.base import BaseModel, get_fc_stack
from pyro.contrib.autoname import scope
import pyro
import numpy as np
import warnings
from scipy.sparse import isspmatrix
from functools import partial
from pyro import poutine
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
import mira.tools.enrichr_enrichments as enrichr
from mira.plots.enrichment_plot import plot_enrichments as mira_plot_enrichments
from torch.distributions.utils import broadcast_all
from pyro.distributions.torch_distribution import ExpandedDistribution
import logging
logger = logging.getLogger(__name__)


'''class WeightedNegativeBinomial(pyro.distributions.NegativeBinomial):

    def __init__(self,*,total_count, weights, probs=None, logits=None, validate_args=None):
        super().__init__(total_count=total_count, probs = probs, logits = logits, validate_args=validate_args)
        _, self.weights = broadcast_all(self.total_count, weights)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = (self.total_count * F.logsigmoid(-self.logits) +
                                 value * F.logsigmoid(self.logits))

        log_normalization = (-torch.lgamma(self.total_count + value) + torch.lgamma(1. + value) +
                             torch.lgamma(self.total_count))

        unweighted_probs = log_unnormalized_prob - log_normalization
        return unweighted_probs * self.weights

    def expand(self, batch_shape, _instance=None):
        return ExpandedDistribution(self, batch_shape)'''


class ExpressionEncoder(torch.nn.Module):


    def __init__(self,embedding_size=None,*,num_endog_features, num_topics, hidden, dropout, num_layers):
        super().__init__()

        if embedding_size is None:
            embedding_size = hidden

        output_batchnorm_size = 2*num_topics + 2
        self.num_topics = num_topics
        self.fc_layers = get_fc_stack(
            layer_dims = [num_endog_features + 1, embedding_size, *[hidden]*(num_layers-2), output_batchnorm_size],
            dropout = dropout, skip_nonlin = True
        )
        
    def forward(self, X, read_depth):

        X = torch.hstack([X, torch.log(read_depth)])

        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])# + 1e-5

        rd_loc = X[:,-2].reshape((-1,1))
        rd_scale = F.softplus(X[:,-1]).reshape((-1,1))# + 1e-5

        return theta_loc, theta_scale, rd_loc, rd_scale

    def topic_comps(self, X, read_depth):

        theta = self.forward(X, read_depth)[0]
        theta = theta.exp()/theta.exp().sum(-1, keepdim = True)
        
        return theta.detach().cpu().numpy()

    def read_depth(self, X, read_depth):
        return self.forward(X, read_depth)[2].detach().cpu().numpy()


class ExpressionTopicModel(BaseModel):

    encoder_model = ExpressionEncoder

    @property
    def genes(self):
        return self.features

    @staticmethod
    def _residual_transform(y_ij, pi_j_hat):
        
        n_i = y_ij.sum(axis = 1, keepdims = True)

        mu_ij_hat = n_i * pi_j_hat[np.newaxis, :]

        count_dif = n_i - y_ij
        expected_count_dif = n_i - mu_ij_hat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            r_ij = np.multiply(
                np.sign(y_ij - mu_ij_hat), 
                np.sqrt(
                np.where(y_ij > 0, 2 * np.multiply(y_ij, np.log(y_ij / mu_ij_hat)), 0) + \
                2 * np.multiply(count_dif, np.log(count_dif / expected_count_dif))
                )
            )

        return np.clip(np.nan_to_num(r_ij), -10, 10)

    def _get_obs_weight(self):

        weights = self.highly_variable.astype(int) + 1
        weights = weights * self.num_exog_features/weights.sum()
        #weights = torch.tensor(self._get_obs_weight(), requires_grad = False).to(self.device)

        return weights

    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, read_depth, anneal_factor = 1.):
        theta_loc, theta_scale = super().model()
        pyro.module("decoder", self.decoder)

        dispersion = pyro.param('dispersion', read_depth.new_ones(self.num_exog_features).to(self.device) * 5., constraint = constraints.positive)
        dispersion = dispersion.to(self.device)

        with pyro.plate("cells", endog_features.shape[0]):
            with poutine.scale(None, anneal_factor):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
                theta = theta/theta.sum(-1, keepdim = True)
                
                expr_rate = self.decoder(theta)

                read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))
            
            if not self.nb_parameterize_logspace:
                mu = torch.multiply(read_scale, expr_rate)
                probs = mu/(mu + dispersion)
                X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, probs = probs).to_event(1), obs = exog_features)
            else:
                logits = (read_scale * expr_rate).log() - (dispersion).log()
                X = pyro.sample('obs', dist.NegativeBinomial(total_count = dispersion, logits = logits).to_event(1), obs = exog_features)


    @scope(prefix= 'rna')
    def guide(self,*,endog_features, exog_features, read_depth, anneal_factor = 1.):
        super().guide()

        with pyro.plate("cells", endog_features.shape[0]):
            
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(endog_features, read_depth)

            with poutine.scale(None, anneal_factor):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                )

                read_depth = pyro.sample(
                    "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                )

    def _get_dataset_statistics(self, endog_features, exog_features):
        super()._get_dataset_statistics(endog_features, exog_features)

        self.residual_pi = np.array(endog_features.sum(axis = 0)).reshape(-1)/endog_features.sum()


    @adi.wraps_modelfunc(tmi.fetch_features, partial(adi.add_obs_col, colname = 'model_read_scale'),
        ['endog_features','exog_features'])
    def _get_read_depth(self, *, endog_features, exog_features, batch_size = 512):

        return self._run_encoder_fn(self.encoder.read_depth, endog_features = endog_features, exog_features = exog_features, 
            batch_size =batch_size, bar = False, desc = 'Calculating reads scale')
        

    def _preprocess_endog(self, X, read_depth):
        
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if isspmatrix(X):
            X = X.toarray()

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_endog_features)
        
        assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        X = self._residual_transform(X.astype(np.float32), self.residual_pi)

        return torch.tensor(X, requires_grad = False).to(self.device)


    def _preprocess_exog(self, X):
        
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if isspmatrix(X):
            X = X.toarray()

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_exog_features)
        
        assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'
        return torch.tensor(X.astype(np.float32), requires_grad = False).to(self.device)

    def _get_save_data(self):
        data = super()._get_save_data()
        data['fit_params']['residual_pi'] = self.residual_pi

        return data
    
    def rank_genes(self, topic_num):
        '''
        Ranks genes according to their activation in module ``topic_num``. Sorted from most suppressed to most activated.

        Args:
            topic_num (int): For which module to rank genes

        Returns:
            np.ndarray: sorted array of gene names in order from most suppressed to most activated given the specified module
        '''
        assert(isinstance(topic_num, int) and topic_num < self.num_topics and topic_num >= 0)

        return self.genes[np.argsort(self._score_features()[topic_num, :])]

    def rank_modules(self, gene):
        '''
        For a gene, rank how much its expression is activated by each module

        Args:
            gene (str): name of gene
        
        Raises:
            AssertionError: if ``gene`` is not in self.genes
        
        Returns:
            (list): of format [(topic_num, activation), ...]
        '''
        
        assert(gene in self.genes)

        gene_idx = np.argwhere(self.genes == gene)[0]
        return list(sorted(zip(range(self.num_topics), self._score_features()[:, gene_idx].reshape(-1)), key = lambda x : -x[1]))


    def get_top_genes(self, topic_num, top_n = None, min_genes = 200, max_genes = 600):
        '''
        For a module, return the top n genes that are most activated.

        Args:
            topic_num (int): For which module to return most activated genes
            top_n (int): number of genes to return

        Returns
            (np.ndarray): Names of top n genes, sorted from least to most activated
        '''

        if top_n is None:
            gene_scores = self._score_features()[topic_num,:]
            top_genes_mask = gene_scores - np.maximum(gene_scores.mean(),0) > 2

            genes_found = top_genes_mask.sum() 

            if genes_found > min_genes:
                if genes_found > max_genes:
                    logger.warn('Topic {} enriched for too many ({}) genes, truncating to {}.'.format(str(topic_num), str(genes_found), str(max_genes)))
                    return self.rank_genes(topic_num)[-max_genes:]
                else:
                    return self.genes[top_genes_mask]
            else:
                logger.warn('Topic {} enriched for too few ({}) genes, taking top {}.'.format(str(topic_num), str(genes_found), str(min_genes)))
                return self.rank_genes(topic_num)[-min_genes : ]

        else:

            assert(isinstance(top_n, int) and top_n > 0)
            return self.rank_genes(topic_num)[-top_n : ]


    def post_topic(self, topic_num, top_n = None, min_genes = 200, max_genes = 600):

        list_id = enrichr.post_genelist(
            self.get_top_genes(topic_num, top_n = top_n, min_genes = min_genes, max_genes = max_genes)
        )

        self.enrichments[topic_num] = dict(
            list_id = list_id,
            results = {}
        )

    def post_topics(self, top_n = None, min_genes = 200, max_genes = 600):

        for i in range(self.num_topics):
            self.post_topic(i, top_n = top_n, min_genes = min_genes, max_genes = max_genes)


    def fetch_topic_enrichments(self, topic_num, ontologies = enrichr.LEGACY_ONTOLOGIES):

        try:
            self.enrichments
        except AttributeError:
            raise AttributeError('User must run "post_topic" or "post_topics" before getting enrichments')

        try:
            list_id = self.enrichments[topic_num]['list_id']
        except (KeyError, IndentationError):
            raise KeyError('User has not posted topic yet, run "post_topic" first.')

        self.enrichments[topic_num]['results'].update(
            enrichr.fetch_ontologies(list_id, ontologies = ontologies)
        )
        #for ontology in ontologies:
            #self.enrichments[topic_num]['results'].update(enrichr.get_ontology(list_id, ontology=ontology))


    def fetch_enrichments(self,  ontologies = enrichr.LEGACY_ONTOLOGIES):
        
        for i in range(self.num_topics):
            self.fetch_topic_enrichments(i, ontologies = ontologies)


    def get_enrichments(self, topic_num):
        
        try:
            return self.enrichments[topic_num]['results']
        except (KeyError, IndentationError):
            raise KeyError('User has not posted topic yet, run "post_topic" first.')

    
    def plot_enrichments(self, topic_num, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', plots_per_row = 2, height = 4, aspect = 2.5, max_genes = 15, pval_threshold = 1e-5,
        color_by_adj = True, palette = 'Reds', gene_fontsize=10):

        '''
        Make plot of geneset enrichments given results from ``get_ontology`` or ``get_enrichments``.

        Example:

            post_id = expr_model.post_genelist(0) #post top 250 module 0 genes
            enrichments = expr_model.get_enrichments(post_id)
            expr_model.plot_enrichments(enrichments)

        Args:
            enrichment_results (dict): output from ``get_ontology`` or ``get_enrichments``
            show_genes (bool): overlay gene names on top of bars
            show_top (int): plot top n enrichment results
            barcolor (color): color of barplot bars
            text_color (text_color): color of text on barplot bars
            return_fig (bool): return fig and axes objects
            enrichments_per_row (int): number of plots per row
            height (float): height of each plot
            aspect (float): multiplier for width of each plot, width = aspect * height
            max_genes (int): maximum number of genes to display on bar

        Returns (if return_fig is True):
            matplotlib.figure, matplotlib.axes.Axes

        '''

        try:
            self.enrichments
        except AttributeError:
            raise AttributeError('User must run "post_topic" or "post_topics" before getting enrichments')

        try:
            results = self.enrichments[topic_num]['results']
        except (KeyError, IndentationError):
            raise KeyError('User has not posted topic yet, run "post_topic" first.')

        if len(results) == 0:
            raise Exception('No results for this topic, user must run "get_topic_enrichments" or "get_enrichments" before plotting.')

        return mira_plot_enrichments(results, text_color = text_color, label_genes = label_genes,
            show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes,
            enrichments_per_row = plots_per_row, height = height, aspect = aspect, pval_threshold = pval_threshold,
            palette = palette, color_by_adj = color_by_adj, gene_fontsize = gene_fontsize)