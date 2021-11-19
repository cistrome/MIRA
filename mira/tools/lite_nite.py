import kladiv2.core.adata_interface as adi
from scipy.stats import chi2, mannwhitneyu
import numpy as np
from scipy.sparse import isspmatrix
from kladiv2.plots.chromatin_differential_plot import plot_chromatin_differential

@adi.wraps_functional(
    adata_extractor = adi.fetch_logp_data, adata_adder = adi.add_global_test_statistic,
    del_kwargs = ['genes','gene_expr','cis_logp','trans_logp']
)
def globally_regulated_gene_test(*, degrees_of_freedom, genes, gene_expr, cis_logp, trans_logp):

    assert(gene_expr.shape == cis_logp.shape == trans_logp.shape)

    num_nonzero = np.ravel( np.array((gene_expr > 0).sum(0)) )
    delta = -2 * (cis_logp.sum(0).reshape(-1) - trans_logp.sum(0).reshape(-1))

    median_nonzero = np.median(num_nonzero)
    effective_sample_size = median_nonzero/(median_nonzero + num_nonzero)

    adjusted_test_stat = effective_sample_size * delta

    pval = 1 - chi2(degrees_of_freedom).cdf(adjusted_test_stat)

    return genes, adjusted_test_stat, pval, num_nonzero


def _global_geneset_test(*,genes, test_statistic, test_gene_group):

    group_mask = np.isin(genes, test_gene_group)

    in_group = test_statistic[group_mask]
    out_group = test_statistic[~group_mask]

    return mannwhitneyu(in_group, out_group, alternative = 'greater')[1]


@adi.wraps_functional(
    adata_extractor = adi.fetch_global_test_statistic, adata_adder = adi.return_output,
    del_kwargs = ['genes','test_statistic']
)
def global_geneset_test(*,genes, test_statistic, test_gene_group):

    return _global_geneset_test(
        genes = genes, 
        test_statistic = test_statistic,
        test_gene_group = test_gene_group
    )


@adi.wraps_functional(
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

    return sorted(results, key = lambda x : x['global_pval'])

@adi.wraps_functional(
    adata_extractor = adi.fetch_cis_trans_prediction, adata_adder = adi.add_chromatin_differential,
    del_kwargs = ['cis_prediction','trans_prediction']
)
def get_chromatin_differential(*,cis_prediction, trans_prediction):
    return np.log2(cis_prediction) - np.log2(trans_prediction)