
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
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
from pyro.distributions.torch_distribution import ExpandedDistribution
import logging
logger = logging.getLogger(__name__)



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


    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, read_depth, anneal_factor = 1.):
        theta_loc, theta_scale = super().model()
        pyro.module("decoder", self.decoder)

        dispersion = pyro.param('dispersion', read_depth.new_ones(self.num_exog_features).to(self.device) * 5., constraint = constraints.positive)
        dispersion = dispersion.to(self.device)

        with pyro.plate("cells", endog_features.shape[0]):
            with poutine.scale(None, anneal_factor/self.reconstruction_weight):
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

            with poutine.scale(None, anneal_factor/self.reconstruction_weight):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                )

                read_depth = pyro.sample(
                    "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                )

    def _get_dataset_statistics(self, dataset):
        super()._get_dataset_statistics(dataset)

        skim_endog = dataset[0][0]
        cummulative_counts = np.zeros(skim_endog.shape[1])

        for i in range(len(dataset)):
            cummulative_counts = cummulative_counts + dataset[i][0].toarray().reshape(-1)

        self.residual_pi = cummulative_counts/cummulative_counts.sum()


    @adi.wraps_modelfunc(tmi.fetch_features, partial(adi.add_obs_col, colname = 'model_read_scale'),
        ['dataset'])
    def _get_read_depth(self, *, dataset, batch_size = 512):

        return self._run_encoder_fn(self.encoder.read_depth, dataset, 
            batch_size =batch_size, bar = False, desc = 'Calculating reads scale')


    def get_endog_fn(self):

        def preprocess_endog(X):
        
            assert(isinstance(X, np.ndarray) or isspmatrix(X))
            
            if isspmatrix(X):
                X = X.toarray()

            assert(len(X.shape) == 2)
            assert(X.shape[1] == self.num_endog_features)
            
            assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

            X = self._residual_transform(X, self.residual_pi).astype(np.float32)

            return X

        return preprocess_endog


    def get_exog_fn(self):
        
        def preprocess_exog(X):

            assert(isinstance(X, np.ndarray) or isspmatrix(X))
            if isspmatrix(X):
                X = X.toarray()

            assert(len(X.shape) == 2)
            assert(X.shape[1] == self.num_exog_features)
            
            assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

            return X.astype(np.float32)

        return preprocess_exog


    def _get_save_data(self):
        data = super()._get_save_data()
        data['fit_params']['residual_pi'] = self.residual_pi

        return data
    
    def rank_genes(self, topic_num):
        '''
        Ranks genes according to their activation in module `topic_num`. Sorted from least to most activated.

        Parameters
        ----------
        topic_num : int
            For which module to rank genes

        Returns
        -------
        np.ndarray: sorted array of gene names in order from most suppressed to most activated given the specified module

        Examples
        --------

        Genes are ranked from least to most activated. To get the top genes:

        .. code-block:: python

            >>> rna_model.rank_genes(0)[-10:]
            array(['ESRRG', 'APIP', 'RPGRIP1L', 'TM4SF4', 'DSCAM', 'NRAD1', 'ST3GAL1',
            'LEPR', 'EXOC6', 'SLC44A5'], dtype=object)

        '''
        assert(isinstance(topic_num, int) and topic_num < self.num_topics and topic_num >= 0)

        return self.genes[np.argsort(self._score_features()[topic_num, :])]

    def rank_modules(self, gene):
        '''
        For a gene, rank how much its expression is activated by each module

        Parameters
        ----------
        gene : str
            Name of gene
    
        Raises
        ------
        AssertionError: if **gene** is not in self.genes
        
        Returns
        -------
        list : of format [(topic_num, activation), ...]

        Examples
        --------

        To see the top 5 modules associated with gene "GHRL":

        .. code-block:: python

            >>> rna_model.rank_modules('GHRL')[:5]
            [(14, 3.375548), (22, 2.321417), (1, 2.3068447), (0, 1.780294), (9, 1.3936363)]

        '''
        
        assert(gene in self.genes)

        gene_idx = np.argwhere(self.genes == gene)[0]
        return list(sorted(zip(range(self.num_topics), self._score_features()[:, gene_idx].reshape(-1)), key = lambda x : -x[1]))


    def get_top_genes(self, topic_num, top_n = None, min_genes = 200, max_genes = 600):
        '''
        For a module, return the top n genes that are most activated.

        Parameters
        ----------
        topic_num : int
            For which module to return most activated genes
        top_n : int > 0
            Number of genes to return
        min_genes : int > 0
            If top_n is None, all activations (distributed standard normal) 
            greater than 3 will be posted. If this is less than **min_genes**,
            then **min_genes** will be posted.
        max_genes : int > 0
            If top_n is None, a maximum of **max_genes** will be posted.

        Returns
        -------
        np.ndarray : Names of top n genes, sorted from least to most activated
        '''

        if top_n is None:
            gene_scores = self._score_features()[topic_num,:]
            top_genes_mask = (gene_scores - np.maximum(gene_scores.mean(),0)) > 3

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


    def post_topic(self, topic_num, top_n = 500, min_genes = 200, max_genes = 600):
        '''
        Post the top genes from topic to Enrichr for geneset enrichment analysis.

        Parameters
        ----------
        topic_num : int
            Topic number to post geneset
        top_n : int, default = 500
            Number of genes to post
        min_genes : int > 0
            If top_n is None, all activations (distributed standard normal) 
            greater than 3 will be posted. If this is less than **min_genes**,
            then **min_genes** will be posted.
        max_genes : int > 0
            If top_n is None, a maximum of **max_genes** will be posted.

        Examples
        --------

        .. code-block:: python

            >>> rna_model.post_topic(10, top_n = 500)
            >>> rna_model.post_topic(10)

        '''

        list_id = enrichr.post_genelist(
            self.get_top_genes(topic_num, top_n = top_n, min_genes = min_genes, max_genes = max_genes)
        )

        self.enrichments[topic_num] = dict(
            list_id = list_id,
            results = {}
        )

    def post_topics(self, top_n = 500, min_genes = 200, max_genes = 600):
        '''
        Iterate through all topics and post top genes to Enrichr.

        Parameters
        ----------
        top_n : int, default = 500
            Number of genes to post
        min_genes : int > 0
            If top_n is None, all activations (distributed standard normal) 
            greater than 3 will be posted. If this is less than **min_genes**,
            then **min_genes** will be posted.
        max_genes : int > 0
            If top_n is None, a maximum of **max_genes** will be posted.

        Examples
        --------

        .. code-block:: python

            >>> rna_model.post_topics()

        '''

        for i in range(self.num_topics):
            self.post_topic(i, top_n = top_n, min_genes = min_genes, max_genes = max_genes)


    def fetch_topic_enrichments(self, topic_num, ontologies = enrichr.LEGACY_ONTOLOGIES):
        '''
        Fetch Enrichr enrichments for a topic. Will return results for the ontologies listed.

        Parameters
        ----------
        topic_num : int
            Topic number to fetch enrichments
        ontologies : list[str], default = mira.tools.enrichr_enrichments.LEGACY_ONTOLOGIES
            List of ontology names from which to retrieve results. May provide
            a list of any onlogies hosted on Enrichr.

        Examples
        --------

        .. code-block:: python

            >>> rna_model.fetch_topic_enrichments(10, ontologies = ['WikiPathways_2019_Mouse'])        
        '''

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
        '''
        Iterate through all topics and fetch enrichments.

        Parameters
        ----------
        ontologies : list[str], default = mira.tl.LEGACY_ONTOLOGIES
            List of ontology names from which to retrieve results. May provide
            a list of any onlogies hosted on Enrichr.

        Examples
        --------

        .. code-block:: python

            >>> rna_model.fetch_enrichments(ontologies = ['WikiPathways_2019_Mouse'])
        '''
        
        for i in range(self.num_topics):
            self.fetch_topic_enrichments(i, ontologies = ontologies)


    def get_enrichments(self, topic_num):
        '''
        Return the enrichment results for a  given topic.

        Paramters
        ---------
        topic_num : int
            Topic for which to return enrichment results
        
        Returns
        -------
        enrichments : dict
            Dictionary with schema:

            .. code-block::
                
                {
                    <ontology> : {
                        [
                            {'rank' : <rank>,
                            'term' : <term>,
                            'pvalue' : <pval>,
                            'zscore': <zscore>,
                            'combined_score': <combined_score>,
                            'genes': [<gene1>, ..., <geneN>],
                            'adj_pvalue': <adj_pval>},
                            ...,
                        ]
                    }
                }   
                
        '''
        
        try:
            return self.enrichments[topic_num]['results']
        except (KeyError, IndentationError):
            raise KeyError('User has not posted topic yet, run "post_topic" first.')

    
    def plot_enrichments(self, topic_num, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', plots_per_row = 2, height = 4, aspect = 2.5, max_genes = 15, pval_threshold = 1e-5,
        color_by_adj = True, palette = 'Reds', gene_fontsize=10):

        '''
        Make plot of geneset enrichments results.

        Parameters
        ----------
        topic_num : int
            Topic for which to plot results
        show_genes : boolean, default = True
            Whether to show gene names on enrichment barplot bars
        show_top : int > 0, default = 10
            Plot this many top terms for each ontology
        barcolor : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
            Color of barplot bars
        label_genes : list[str] or np.ndarray[str]
            Add an asterisc by the gene name of genes in this list. Useful for
            finding transcription factors or signaling factors of interest in
            enrichment results.
        text_color : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
            Color of text on plot
        plots_per_row : int > 0, default = 2
            Number of onotology plots per row in figure
        height : float > 0, default = 4
            Height of each ontology plot
        aspect : float > 0, default = 2.5
            Aspect ratio of ontology plot
        max_genes : int > 0, default = 15
            Maximum number of genes to plot on each term bar
        pval_threshold : float (0, 1), default = 1e-5
            Upper bound on color map for adjusted p-value coloring of bar
            outlines.
        color_by_adj : boolean, default = True
            Whether to outline term bars with adjusted p-value
        palette : str
            Color palette for adjusted p-value
        gene_fontsize : float > 0, default = 10
            Fontsize of gene names on term bars

        Returns
        -------
        ax : matplotlib.pyplot.axes

        Examples
        --------

        .. code-block:: python

            >>> rna_model.post_topic(13, 500)
            >>> rna_model.fetch_topic_enrichments(13, 
            ... ontologies=['WikiPathways_2019_Mouse','BioPlanet_2019'])
            >>> rna_model.plot_enrichments(13, height=4, show_top=6, max_genes=10, 
            ... aspect=2.5, plots_per_row=1)

        .. image:: /_static/mira.topics.ExpressionTopicModel.plot_enrichments.svg
            :width: 1200

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
            plots_per_row = plots_per_row, height = height, aspect = aspect, pval_threshold = pval_threshold,
            palette = palette, color_by_adj = color_by_adj, gene_fontsize = gene_fontsize)