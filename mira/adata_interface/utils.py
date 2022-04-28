
import anndata
import numpy as np
import pandas as pd
import logging
import mira.adata_interface.regulators as ri
import mira.adata_interface.rp_model as rpi
logger = logging.getLogger(__name__)


def wide_view():
    '''
    Makes Jupyter notebooks take up whole screen.
    '''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))

def pretty_sderr():
    '''
    Changes stderr color to blue in Jupyter notebooks.
    '''
    from IPython.core.display import HTML
    return HTML("""
<style>
div.output_stderr {
    background: #e6e7ed;
}
</style>
    """)

def show_gif(path):
    '''
    Display GIF in Jupyter notebook.
    '''
    from IPython.display import Image, display
    with open(path,'rb') as f:
        display(Image(data=f.read(), format='png'))


def make_joint_representation(
    adata1, adata2,
    adata1_key = 'X_umap_features',
    adata2_key = 'X_umap_features',
    key_added = 'X_joint_umap_features',
):
    '''
    Finds common cells between two dataframes and concatenates features
    to form the joint representation. 

    Parameters
    ----------
    adata1, adata2 : anndata.AnnData
        Two AnnData objects from which to construct joint representation.
        Order (ATAC or RNA) does not matter. 
    adata1_key : str, default='X_umap_features'
        Which key in `.obsm` to find ILR-transformed topic embeddings
        in adata1.
    adata2_key : str, default='X_umap_features'
        Which key in `.obsm` to find ILR-transformed topic embeddings
        in adata2.
    key_added : str, default='X_joint_umap_features'
        Key to add to both adatas' `.obsm` containing the joint representation.

    Returns
    -------

    adata1, adata2 : anndata.AnnData
        Adata objects returned in the order provided. New adata objects
        contain only cells common between both input adatas, and have the
        same ordering. Both adatas have a new field: the joint representation,
        stored in `.obsm[<key_added>]`.

    '''

    obs_1, obs_2 = adata1.obs_names.values, adata2.obs_names.values

    shared_cells = np.intersect1d(obs_1, obs_2)

    num_shared_cells = len(shared_cells)
    if num_shared_cells == 0:
        raise ValueError('No cells/obs are shared between these two datasets. Make sure .obs_names is formatted identically between datasets.')

    if num_shared_cells < len(obs_1) or num_shared_cells < len(obs_2):
        logger.warn('Some cells are not shared between views. Returned adatas will be subset copies')

    total_cells = num_shared_cells + len(obs_1) + len(obs_2) - 2*num_shared_cells

    logger.info('{} out of {} cells shared between datasets ({}%).'.format(
        str(num_shared_cells), str(total_cells), str(int(num_shared_cells/total_cells * 100))
    ))

    adata1 = adata1[shared_cells].copy()
    adata2 = adata2[shared_cells].copy()

    joint_representation = np.hstack([
        adata1.obsm[adata1_key], adata2.obsm[adata2_key]
    ])

    adata1.obsm[key_added] = joint_representation
    adata2.obsm[key_added] = joint_representation
    
    logger.info('Key added to obsm: {}'.format(key_added))

    return adata1, adata2


def subset_factors(atac_adata,*, use_factors, factor_type = 'motifs'):
    '''
    Subset which transcription factor binding annotations are used
    in downstream analysis. This function marks annotations if the factor
    is in the list provided to *use_factors*, but does not erase
    out-of-list factors' information. Thus, a new subset may be applied 
    without re-scanning for motifs.

    **Important: we do not suggest subsetting to transcription factors
    that have high or highly dispersed expression in multiomics analyses. 
    Many transcription factors may have potent regulatory effects without
    showing a great change in expression.**

    Parameters
    ----------
    
    atac_adata : anndata.AnnData
        AnnData object of ATAC features
    use_factors : np.ndarray[str], list[str]
        List of transcription factor names to use for downstream analysis.
    factor_type : {'motifs','chip'}, default='motifs' 
        Which factor type to filter.

    Returns
    -------

    anndata.AnnData

    '''
    
    metadata, _ = ri.fetch_factor_meta(None, atac_adata, 
        factor_type = factor_type, mask_factors = False)
    
    assert(isinstance(use_factors, (list, np.ndarray, pd.Index))),'Must supply list of factors for either "user_factors" or "hide_factors".'

    factor_mask = [
        factor['parsed_name'] in use_factors
        for factor in metadata
    ]

    ri.add_factor_mask(atac_adata, factor_mask, factor_type = factor_type)
    
    logger.info('Found {} factors in expression data.'.format(str(np.array(factor_mask).sum())))


def fetch_factor_meta(atac_adata, factor_type = 'motifs', mask_factors = False):
    '''
    Fetch metadata associated with transcription factor binding annotations.
    Returns "id", "name", and "parsed_name" fields. "parsed_name" is used
    to look up TFs in expression data.

    Parameters
    ----------
    
    atac_adata : anndata.AnnData
        AnnData object of ATAC features
    mask_factors : boolean, default = False
        Whether to subset the list of TFs returned to those flagged by 
        "subset_factors".
    factor_type : {'motifs','chip'}, default='motifs' 
        Which factor type to filter.

    Returns
    -------

    pd.DataFrame

    '''

    return pd.DataFrame(
            ri.fetch_factor_meta(None, atac_adata, factor_type = factor_type, 
            mask_factors = mask_factors)[0]
    )


def fetch_factor_hits(atac_adata, factor_type = 'motifs', mask_factors = False):
    '''
    Returns AnnData object of transcription factor binding annotations.
    
    Parameters
    ----------
    
    atac_adata : anndata.AnnData
        AnnData object of ATAC features
    mask_factors : boolean, default = False
        Whether to subset the list of TFs returned to those flagged by 
        "subset_factors".
    factor_type : {'motifs','chip'}, default='motifs' 
        Which factor type to filter.

    Returns
    -------

    anndata.AnnData:
        `.obs` : pd.DataFrame
            TF annotation metadata.
        `.var` : pd.DataFrame
            Peak metadata taken from *atac_adata*.
        `X` : scipy.sparsematrix
            TF binding predictions. For motifs, values show MOODS3 "Match Score", 
            with higher values indicating a better match between a peak sequence and motif PWM. 
            For ChIP-seq samples, values are binary, with 1 indicating overlap with a 
            peak in a Cistrome ChIP-seq sample.
    
    '''

    metadata, mask = ri.fetch_factor_meta(None, atac_adata, 
        factor_type = factor_type, mask_factors = mask_factors)

    try:
        hits_matrix = atac_adata.varm[factor_type + '_hits'].T.tocsr()
    except KeyError:
        raise KeyError('Factor binding predictions for {} not yet calculated.'.format(factor_type))

    hits_matrix = hits_matrix[mask, :]

    return anndata.AnnData(
        obs = pd.DataFrame(metadata),
        var = atac_adata.var,
        X = hits_matrix,
    )

def fetch_gene_TSS_distances(atac_adata):
    '''
    Returns matrix of distances between gene transcription
    start sites and peaks.
    '''
    try:
        atac_adata.varm['distance_to_TSS']
    except KeyError:
        raise KeyError('TSS annotations not found. Run "mira.tl.get_distance_to_TSS" before running this function.')

    return anndata.AnnData(
                X = atac_adata.varm['distance_to_TSS'].T,
                var = atac_adata.var,
                obs = pd.DataFrame(
                    atac_adata.uns['TSS_metadata'], 
                    index = atac_adata.uns['distance_to_TSS_genes'])
               )


def fetch_binding_sites(atac_adata, factor_type = 'motifs',*, id):
    '''
    Returns `.var` field of `atac_adata`, but subset to only contain
    peaks which are predicted to bind a certain transcription factor.

    Parameters
    ----------

    atac_adata : anndata.AnnData
        AnnData object with accessibility features, annotated with
        factor binding predictions
    factor_type : {"motifs", "chip"}, default = "motifs"
        Which type of factor to look for `id`.
    id : str
        Unique identifier for transcription factor binding. JASPAR
        ID in the case of motifs, or cistrome ID for ChIP samples.

    Returns
    -------

    pd.DataFrame : subset of `atac_adata.var`
    
    '''

    factor_hits = fetch_factor_hits(atac_adata, 
            factor_type = factor_type, 
            mask_factors = False)

    factor_hits.obs = factor_hits.obs.set_index('id')
    
    try:
        factor_hits = factor_hits[id]
    except KeyError:
        raise KeyError('Factor id {} not found in dataset. To see factor meta and find valid IDs, use "mira.utils.fetch_factor_meta".'.format(id))

    return factor_hits[:, factor_hits.X.tocsr().indices].var


def fetch_TSS_data(adata):
    '''
    Returns TSS metadata from `mira.tl.get_distance_to_TSS`.
    '''
    return rpi.fetch_TSS_data(None, adata)


def fetch_ISD_matrix(expr_adata, factor_type = 'motifs', mask_factors = True, 
        mask_untested_genes = True, id_column = 'name'):

    results = ri.fetch_ISD_results(None, expr_adata, factor_type, mask_factors=mask_factors, 
        mask_untested_genes = mask_untested_genes)
    
    try:
        return pd.DataFrame(
            results['isd_matrix'], index = results['genes'], 
            columns = [meta[id_column] for meta in results['factors']]
        )
    except KeyError:
        raise KeyError('{} column is not associated with {} factor metadata')


