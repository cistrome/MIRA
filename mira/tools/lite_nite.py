from numpy.lib.function_base import median
import mira.adata_interface.core as adi
import mira.adata_interface.lite_nite as lni
from scipy.stats import chi2, mannwhitneyu
import numpy as np
from scipy.sparse import isspmatrix
from functools import partial
from mira.plots.chromatin_differential_plot import plot_chromatin_differential


def _get_NITE_score(gene_expr, lite_logp, nite_logp, median_nonzero_expression = None, axis = 0):

    assert(gene_expr.shape == lite_logp.shape == nite_logp.shape)

    num_nonzero = np.ravel( np.array((gene_expr > 0).sum(axis)) )
    if median_nonzero_expression is None:
        median_nonzero = np.median(num_nonzero)
    else:
        assert(isinstance(median_nonzero_expression, int) and median_nonzero_expression > 0)
        median_nonzero = median_nonzero_expression

    delta = -2 * (lite_logp.sum(axis).reshape(-1) - nite_logp.sum(axis).reshape(-1))
    effective_sample_size = median_nonzero/(median_nonzero + num_nonzero)

    nite_score = effective_sample_size * delta

    return nite_score, num_nonzero


@adi.wraps_functional(
    lni.fetch_logp_data, lni.add_NITE_score_gene,
    ['genes','gene_expr','lite_logp','nite_logp']
)
def get_NITE_score_genes(median_nonzero_expression = None,*, genes, gene_expr, lite_logp, nite_logp):

    return (genes, *_get_NITE_score(gene_expr, lite_logp, nite_logp, 
        median_nonzero_expression = median_nonzero_expression, axis = 0))


@adi.wraps_functional(
    lni.fetch_logp_data, lni.add_NITE_score_cell,
    ['genes','gene_expr','lite_logp','nite_logp']
)
def get_NITE_score_cells(median_nonzero_expression = None, *, genes, gene_expr, lite_logp, nite_logp):

    return _get_NITE_score(gene_expr, lite_logp, nite_logp, 
        median_nonzero_expression = median_nonzero_expression, axis = 1)


@adi.wraps_functional(
    lni.fetch_lite_nite_prediction, 
    partial(adi.add_layer, add_layer = 'chromatin_differential', sparse = True),
    ['lite_prediction','nite_prediction','genes']
)
def get_chromatin_differential(*,lite_prediction, nite_prediction, genes):
    return genes, np.log2(lite_prediction) - np.log2(nite_prediction)


'''def _global_geneset_test(*,genes, test_statistic, test_gene_group):

    group_mask = np.isin(genes, test_gene_group)

    in_group = test_statistic[group_mask]
    out_group = test_statistic[~group_mask]

    return mannwhitneyu(in_group, out_group, alternative = 'greater')'''

'''@adi.wraps_functional(
    adata_extractor = adi.fetch_global_test_statistic, adata_adder = adi.return_output,
    del_kwargs = ['genes','test_statistic']
)
def global_geneset_test(*,genes, test_statistic, test_gene_group):

    return _global_geneset_test(
        genes = genes, 
        test_statistic = test_statistic,
        test_gene_group = test_gene_group
    )'''

'''@adi.wraps_functional(
    adata_extractor = adi.fetch_global_test_statistic,
    del_kwargs = ['genes','test_statistic']
)
def plot_global_regulation_statistic(*,genes, test_statistic, test_gene_group):
    pass


@adi.wraps_functional(
    adata_extractor = adi.fetch_global_test_statistic, adata_adder = adi.return_output,
    del_kwargs = ['genes','test_statistic']
)
def global_ontology_term_test(test_top_n = 10,*, genes, test_statistic, enrichments):

    results = []
    for ontology, terms in enrichments.items():
        for term in terms[:test_top_n]:

            test_genes = term['genes']
            in_group_mask = np.isin(genes, test_genes)

            if in_group_mask.sum() > 0:

                test_genes = list(zip(*sorted(zip(genes[in_group_mask], 
                                test_statistic[in_group_mask]), key = lambda x : x[1])))[0]

                num_genes_tested = len(test_genes)

                results.append(
                    {'ontology' : ontology,
                    'term' : term['term'],
                    'num_genes_tested' : num_genes_tested,
                    'global_pval' : _global_geneset_test(genes = genes, 
                                                            test_statistic = test_statistic, 
                                                            test_gene_group = test_genes),
                    'genes' : test_genes,
                    }
                )

    return sorted(results, key = lambda x : x['global_pval'])'''