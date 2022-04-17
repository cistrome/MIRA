from mira.tools.motif_scan import validate_peaks, _parse_motif_name
from lisa import FromRegions
import mira.adata_interface.core as adi
import mira.adata_interface.regulators as ri
from functools import partial
import numpy as np

@adi.wraps_functional(ri.fetch_peaks, partial(ri.add_factor_hits_data, factor_type = 'chip'), ['peaks'])
def get_ChIP_hits_in_peaks(species = 'mm10', *,peaks):
    '''
    Find ChIP hits that overlap with accessible regions using CistromeDB's 
    catalogue of publically-available datasets. 

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData of accessibility features
    species : {"hg38", "mm10"}
        Organism. CistromeDB's catalogue contains samples for hg38 and mm10.
    chrom : str, default = "chr"
        The column in `adata.var` corresponding to the chromosome of peaks
    start : str, defualt = "start"
        The column in `adata.var` corresponding to the start coordinate of peaks
    end : str, default = "end"
        The column in `adata.var` corresponding to the end coordinate of peaks

    Returns
    -------
    adata : anndata.AnnData
        `.varm["chip_hits"]` : scipy.spmatrix[float] of shape (n_motifs, n_peaks)
            Called ChIP hits for each peak. Non-significant hits are left empty in the sparse matrix.
        `.uns['chip']` : dict of type {str : list}
            Dictionary of metadata for ChIP samples. Each key is an attribute. Attributes 
            recorded for each motif are the ID, name, parsed factor name (for lookup
            in expression data), and whether expression data exists for that factor. The
            columns are labeled id, name, parsed_name, and in_expr_data, respectively. 

    .. note::

        To retrieve the metadata for ChIP, one may use the method 
        `mira.utils.fetch_factor_meta(adata, factor_type = "chip")`.
        Methods that interact with binding site data always have a `factor_type` parameter.
        This parameter defaults to "motifs", so when using ChIP data, specify
        `factory_type` = "chip".

    Examples
    --------

    .. code-block:: python

        >>> atac_data.var
        ...                       chr   start     end
        ...    chr1:9778-10670     chr1    9778   10670
        ...    chr1:180631-181281  chr1  180631  181281
        ...    chr1:183970-184795  chr1  183970  184795
        ...    chr1:190991-191935  chr1  190991  191935
        >>> mira.tl.get_ChIP_hits_in_peaks(atac_data, 
        ...    chrom = "chr", start = "start", end = "end",
        ...    species = "hg38")
        ...    Grabbing hg38 data (~15 minutes):
        ...       Downloading from database    
        ...       Done
        ...    Loading gene info ...
        ...    Validating user-provided regions ...
        ...    WARNING: 71 regions encounted from unknown chromsomes: KI270728.1,GL000194.1,GL000205.2,GL000195.1,GL000219.1,KI270734.1,GL000218.1,KI270721.1,KI270726.1,KI270711.1,KI270713.1
        ...    INFO:mira.adata_interface.regulators:Added key to varm: chip_hits
        ...    INFO:mira.adata_interface.regulators:Added key to uns: chip

    '''

    peaks = validate_peaks(peaks)

    regions_test = FromRegions(species, peaks)

    chip_hits, sample_ids, metadata = regions_test._load_factor_binding_data()

    bin_map = np.hstack([np.arange(chip_hits.shape[0])[:,np.newaxis], regions_test.region_score_map[:, np.newaxis]])

    new_hits = regions_test.data_interface.project_sparse_matrix(chip_hits, bin_map, num_bins=len(peaks))

    hits_matrix = new_hits.T.tocsr()
    factors = metadata['factor']
    parsed_factor_names = list(map(_parse_motif_name, factors))

    return sample_ids, factors, parsed_factor_names, hits_matrix