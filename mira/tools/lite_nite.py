import mira.adata_interface.core as adi
import mira.adata_interface.lite_nite as lni
import numpy as np
from functools import partial


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
    '''
    Calculates the NITE score (Non-locally Influence Transcriptional Expression) for each **gene**. The NITE
    score quantifies how well changes in local chromatin accessibility explain changes in gene expression.

    Parameters
    ----------
    adata : anndata.AnnData
        Adata of expression features per cell. This data must first be annotated with "LITE_logp" and
        "NITE_logp" using LITE and NITE RP models\' `get_logp` function.
    median_nonzero_expression : int > 0 or None, default = None
        The NITE score is normalized for nonzero counts per gene, which means the test
        is dependent on the genome-wide distribution of nonzero counts per gene. If you are not testing
        a large quantity of genes simultaneously, then the median of the distribution of 
        nonzero counts will be noisy. You may provide your own value for the median number 
        of nonzero counts per cell in this instance.

    Returns
    -------
    adata : anndata.AnnData
        `.var["NITE_score"]` : np.ndarray[float] of shape (n_genes,)
            Gene NITE score. Genes that were not tested will be assigned np.nan.

    Raises
    ------
    KeyError : if adata is missing "LITE_logp" or "NITE_logp"

    Examples
    --------

    .. code-block:: python

        >>> rp_args = dict(expr_adata = atac_data, expr_adata = rna_data)
        >>> litemodel.predict(**rp_args)
        >>> nitemodel.predict(**rp_args)
        >>> mira.tl.get_NITE_score_genes(rna_data)
        
    '''

    return (genes, *_get_NITE_score(gene_expr, lite_logp, nite_logp, 
        median_nonzero_expression = median_nonzero_expression, axis = 0))


@adi.wraps_functional(
    lni.fetch_logp_data, lni.add_NITE_score_cell,
    ['genes','gene_expr','lite_logp','nite_logp']
)
def get_NITE_score_cells(median_nonzero_expression = None, *, genes, gene_expr, lite_logp, nite_logp):
    '''
    Calculates the NITE score (Non-locally Influence Transcriptional Expression) for each **cell**. The NITE
    score quantifies how well changes in local chromatin accessibility explain changes in gene expression
    in that cell.

    Parameters
    ----------
    adata : anndata.AnnData
        Adata of expression features per cell. This data must first be annotated with "LITE_logp" and
        "NITE_logp" using LITE and NITE RP models\' `get_logp` function.

    Returns
    -------
    adata : anndata.AnnData
        `.obs["NITE_score"]` : np.ndarray[float] of shape (n_cells,)
            Cell NITE score.

    Raises
    ------
    KeyError : if adata is missing "LITE_logp" or "NITE_logp"

    Examples
    --------
    >>> rp_args = dict(expr_adata = atac_data, expr_adata = rna_data)
    >>> litemodel.predict(**rp_args)
    >>> nitemodel.predict(**rp_args)
    >>> mira.tl.get_NITE_score_cells(rna_data)
    '''

    return _get_NITE_score(gene_expr, lite_logp, nite_logp, 
        median_nonzero_expression = median_nonzero_expression, axis = 1)


@adi.wraps_functional(
    lni.fetch_lite_nite_prediction, 
    partial(adi.add_layer, add_layer = 'chromatin_differential', sparse = True),
    ['lite_prediction','nite_prediction','genes']
)
def get_chromatin_differential(*,lite_prediction, nite_prediction, genes):
    '''
    The per-cell difference in predictions between LITE and NITE models of gene
    is called "chromatin differential", and reflects the over or under-
    estimation of expression levels by local chromatin. Positive chromatin 
    differential means local chromatin over-estimates expression in that cell,
    negative chromatin differential means lcoal chromatin under-estimates
    expression.

    Parameters
    ----------
    adata : anndata.AnnData
        Adata of expression features per cell. This data must first be annotated with "LITE_prediction" and
        "NITE_prediction" using LITE and NITE RP models\' `predict` function.

    Returns
    -------
    adata : anndata.AnnData
        `.layers["chromatin_differential"]` : scipy.spmatrix of shape (n_cells, n_genes)
            Chromatin differential matrix. Genes that were not modeled are left empty.

    Raises
    ------
    KeyError : if adata is missing "LITE_prediction" or "NITE_prediction".

    Examples
    --------
    >>> rp_args = dict(expr_adata = atac_data, expr_adata = rna_data)
    >>> litemodel.predict(**rp_args)
    >>> nitemodel.predict(**rp_args)
    >>> mira.tl.get_chromatin_differential(rna_data)
    '''
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