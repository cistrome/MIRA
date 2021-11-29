import numpy as np
import mira.adata_interface.core as adi
import mira.adata_interface.regulators as ri
from scipy.stats import mannwhitneyu
import tqdm
from functools import partial
import logging

logger = logging.getLogger(__name__)

def _driver_TF_test(background = None, alt_hypothesis = 'greater',*,
    geneset, isd_matrix, genes, factors):

    assert((~np.isnan(isd_matrix)).all()), 'Some factor/gene combinations have not been tested. Rerun the pISD test.'
    assert(isinstance(geneset, (list, np.ndarray)))

    if not background is None:
        assert(isinstance(background, (list, np.ndarray)))
        background = np.array(background)
    else:
        background = genes

    geneset = np.intersect1d(geneset, genes)
    background = np.intersect1d(background, genes)

    background = np.setdiff1d(background, geneset)

    query_mask = np.isin(genes, geneset)
    background_mask = np.isin(genes, background)

    assert(query_mask.sum() > 0 and background_mask.sum() > 0), 'No pISD-tested genes are in the query or background'

    logger.info('Testing with {} query genes and {} background genes, against {} factors'.format(
        str(query_mask.sum()), str(background_mask.sum()), str(isd_matrix.shape[-1])
    ))

    results = []
    for factor_scores in tqdm.tqdm(isd_matrix.T, desc = 'Testing factors'):
        results.append(
            mannwhitneyu(factor_scores[query_mask], factor_scores[background_mask], 
                alternative = alt_hypothesis)
        )

    return [
        dict(**meta.copy(), pval = pval, test_statistic = test_stat)
        for meta, test_stat, pval in zip(factors, *list(zip(*results)))
    ]

@adi.wraps_functional(
    ri.fetch_driver_TF_test, adi.return_output,
    ['isd_matrix','genes','factors']
)
def driver_TF_test(background = None, alt_hypothesis = 'greater',*,
    geneset, isd_matrix, genes, factors):

    return _driver_TF_test(background = background, alt_hypothesis = alt_hypothesis,
        geneset = geneset, isd_matrix = isd_matrix, genes = genes, factors = factors)