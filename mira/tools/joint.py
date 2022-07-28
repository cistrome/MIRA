
import numpy as np
import mira.adata_interface.core as adi
import mira.adata_interface.joint as ji
from functools import partial
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances


def _get_total_MI(x, y):

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert len(x) == len(y)

    x_marg = x.mean(0,keepdims = True)
    y_marg = y.mean(0, keepdims = True)

    joint = (x[:, np.newaxis, :] * y[:,:, np.newaxis]).mean(axis = 0)
    marg = (x_marg * y_marg.T)[np.newaxis, :,:]
    
    mutual_information = np.sum(joint*np.log(joint/marg))

    return mutual_information


def _get_pointwise_MI(x, y):

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert len(x) == len(y)

    x_marg = x.mean(0,keepdims = True)
    y_marg = y.mean(0, keepdims = True)

    joint = x[:, np.newaxis, :] * y[:,:, np.newaxis]
    marg = (x_marg * y_marg.T)[np.newaxis, :,:]
    
    mutual_information = np.sum(joint*np.log(joint/marg), axis = (-2,-1))

    return mutual_information

@adi.wraps_functional(
    partial(ji.fetch_obsms, key = 'X_topic_compositions'),
    partial(adi.add_obs_col, colname = 'pointwise_mutual_information'),
    fill_kwargs= ['x','y'],
    joint = True,
)
def get_cell_pointwise_mutual_information(x, y):
    '''
    For each cell, calculate the pointwise mutual information between 
    RNA and ATAC topic compositions. This compares the joint distribution
    of topic compositions against the marginal distributions over all cells.
    High values for pointwise mutual information suggest that the topic
    compositions in one mode statistically support the compositions in the other.

    Parameters
    ----------

    expr_adata : anndata.AnnData
        AnnData object with expression features, must have "X_topic_compositions" in `.obsm`.
    atac_adata : anndata.AnnData
        AnnData object with accessibility features, must have "X_topic_compositions" in `.obsm`.

    Returns
    -------

    adata : anndata.AnnData
        `.obs['pointwise_mutual_information']` : np.ndarray[float] of shape (n_cells,)
            Pointwise mutual information between expression and accessibility topoics
            for each cell
    
    '''
    return _get_pointwise_MI(x, y)


@adi.wraps_functional(
    partial(ji.fetch_obsms, key = 'X_topic_compositions'),
    fill_kwargs= ['x','y'],
    joint = True,
)
def summarize_mutual_information(x, y):
    '''
    Calculate the total mutual information between expression and accessibility
    topics. A value of 0 indicates low correspondance between modes,
    while 0.5 indicates high correspondance. Good models for cell systems
    should have high mutual information.

    Parameters
    ----------

    expr_adata : anndata.AnnData
        AnnData object with expression features, must have "X_topic_compositions" in `.obsm`.
    atac_adata : anndata.AnnData
        AnnData object with accessibility features, must have "X_topic_compositions" in `.obsm`.

    Returns
    -------

    mutual information : float
        Total mutual information between expression and accessibility topics. Use this 
        metric to evaluate the concordance between models across modes.

    '''
    return _get_total_MI(x, y)


@adi.wraps_functional(
    partial(ji.fetch_obsms, key = 'X_umap_features'),
    partial(adi.add_obs_col, colname = 'relative_mode_weights'),
    fill_kwargs= ['x','y'],
    joint = True,
)
def get_relative_norms(x,y):
    '''
    One may assume that the influence of the two modalities on the joint representation
    is driven by the relative magnitude of the norm of these modalities' embeddings. This 
    function calculates the relative norm of the embeddings so that one can determine which
    model is principally driving joint UMAP geometry.
    
    Parameters
    ----------

    expr_adata : anndata.AnnData
        AnnData object with expression features, must have "X_umap_features" in `.obsm`.
    atac_adata : anndata.AnnData
        AnnData object with accessibility features, must have "X_umap_features" in `.obsm`.

    Returns
    -------

    adata : anndata.AnnData
        `.obs['relative_mode_weights']` : np.ndarray[float] of shape (n_cells,)
            log2 ratio of expression embedding norms over accessibility embedding
            norms. This per-cell metric will thus be positive if the joint
            representation for that cell is primarily driven by expression topics,
            and negative for accessibility.
    
    '''

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert len(x) == len(y)

    return np.log2(
        np.linalg.norm(x, axis = -1)/np.linalg.norm(y, axis = -1)
    )


@adi.wraps_functional(
    partial(ji.fetch_obsms, key = 'X_topic_compositions'),
    partial(ji.format_corr_dataframe, adata1_name = 'RNA', adata2_name = 'ATAC'),
    fill_kwargs= ['x','y'],
    joint = True,
)
def get_topic_cross_correlation(x, y):
    '''
    Get DataFrame of pearson cross-correlation between expression and accessibility
    topics.

    Parameters
    ----------

    expr_adata : anndata.AnnData
        AnnData object with expression features, must have "X_topic_compositions" in `.obsm`.
    atac_adata : anndata.AnnData
        AnnData object with accessibility features, must have "X_topic_compositions" in `.obsm`.

    Returns
    -------

    cross correlations : pd.DataFrame of shape (n_expr_topics, n_accessibility_topics)
        Pearson cross correlation between expression and accessibility topics across
        all cells.        

    '''
    return pairwise_distances(
        x.T, y.T,
        metric=lambda x,y : pearsonr(x,y)[0]
    )