
from lisa.core import genome_tools
from lisa.core.genome_tools import Region, RegionSet, Genome
from collections import Counter
import logging
import numpy as np
from scipy.sparse import coo_matrix
from mira.adata_interface.core import wraps_functional
from mira.adata_interface.rp_model import get_peak_and_tss_data, add_peak_gene_distances

logger = logging.getLogger(__name__)

class StrandedRegion(Region):

    def __init__(self, chromosome, start, end, strand = "+", annotation = None):
        super().__init__(chromosome, start, end, annotation = annotation)
        self.strand = strand
        

def project_sparse_matrix(input_hits, bin_map, num_bins, binarize = False):

    index_converted = input_hits.tocsc()[bin_map[:,0], :].tocoo()

    input_hits = coo_matrix(
        (index_converted.data, (bin_map[index_converted.row, 1], index_converted.col)), 
        shape = (num_bins, input_hits.shape[1]) if not num_bins is None else None 
    ).tocsr()

    if binarize:
        input_hits.data = np.ones_like(input_hits.data)

    return input_hits

def check_region_specification(regions, genome):

        invalid_chroms = Counter()
        valid_regions = []
        for i, region in enumerate(regions):
            assert(isinstance(region, (tuple, list)) and len(region) == 3), 'Error at region #{}: Each region passed must be in format (string \"chr\",int start, int end'\
                .format(str(i))
            
            try:
                new_region = Region(*region, annotation = i)
                genome.check_region(new_region)
                valid_regions.append(new_region)
            except ValueError:
                raise AssertionError('Error at region #{}: Could not coerce positions into integers'.format(str(i)))
            except genome_tools.NotInGenomeError as err:
                invalid_chroms[region[0]]+=1
                #raise AssertionError('Error at region #{}: '.format(str(i+1)) + str(err) + '\nOnly main chromosomes (chr[1-22,X,Y] for hg38, and chr[1-19,X,Y] for mm10) are permissible for LISA test.')
            except genome_tools.BadRegionError as err:
                raise AssertionError('Error at region #{}: '.format(str(i+1)) + str(err))
        
        if len(invalid_chroms) > 0 :
            logger.warn('{} regions encounted from unknown chromsomes: {}'.format(
                str(sum(invalid_chroms.values())), str(','.join(invalid_chroms.keys()))
            ))

        return valid_regions


def get_masked_distances(promoter_set, region_set, max_distance):

    def stranded_distance(r1, r2):
        if r1.chromosome == r2.chromosome:
            distance = (-1 if r2.strand == '-' else 1) * (r1.get_center() - r2.get_center())

            if distance == 0: #cannot set to zero b/c sparse
                distance = 1

            return distance
        else:
            return 0


    logger.info('Finding peak intersections with promoters ...')
    region_promoter_map = region_set.map_intersects(
                                promoter_set, 
                                distance_function = lambda x,y : x.overlaps(y, min_overlap_proportion=0.3), 
                                slop_distance=0
                            ).tocsr() #REGIONS X GENES

    not_promoter_region = (1 - region_promoter_map.sum(axis = 1) > 0) # REGIONS X 1

    logger.info('Calculating distances between peaks and TSS ...')
    distance_matrix = region_set.map_intersects(
                                promoter_set,
                                distance_function = stranded_distance, #ensure zero distance is not masked out by sparse matrix
                                slop_distance= 0.5 * max_distance
                        ).tocsr() #REGIONS X GENES

    logger.info('Masking other genes\' promoters ...')
    distance_matrix.multiply(not_promoter_region) + distance_matrix.multiply(region_promoter_map)

    distance_matrix.eliminate_zeros()

    return distance_matrix


@wraps_functional(get_peak_and_tss_data, add_peak_gene_distances, ['peaks','gene_id','chrom','start','end','strand'])
def get_distance_to_TSS(max_distance = 6e5, promoter_width = 3000,*, 
    peaks, gene_id, chrom, start, end, strand, genome_file):
    '''
    Given TSS data for genes, find the distance between the TSS of each gene
    and the center of each accessible site measured in the data. This distance
    is used to train RP Models.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object of chromatin accessibility. Peak locations located in
        `.var` with columns corresponding to the chromosome, start, and end
        coordinates given by the `peak_chrom`, `peak_start` and `peak_end`
        parameters, respectively. 
    tss_data : pd.DataFrame or str
        DataFrame of TSS locations for each gene. TSS information must include
        the chromosome, start, end, strand, and symbol of the gene. May pass
        either an in-memory dataframe or path to that dataframe on disk.
    sep : str, default = "\t"
        If loading `tss_data` from disk, use this separator character.

    peak_chrom : str, default = "chr"
        The column in `adata.var` corresponding to the chromosome of peaks
    peak_start : str, defualt = "start"
        The column in `adata.var` corresponding to the start coordinate of peaks
    peak_end : str, default = "end"
        The column in `adata.var` corresponding to the end coordinate of peaks
    
    gene_chrom : str, default = "chrom"
        The column in `tss_data` corresponding to the chromosome of genes
    gene_start : str, default = "txStart"
        The column in `tss_data` corresponding to the start index of a transcript.
        For plus-strand genes, this will be the TSS location.
    gene_end : str, default = "txEnd"
        The column in `tss_data` corresponding to the end of a transcript.
        For minus-strand genes, this will be the TSS location.
    gene_strand : str, defualt = "strand"
        The column in `tss_data` corresponding to the trandedness of the 
        gene.
    gene_id : str, default = "geneSymbol"
        The column in `tss_data` corresponding to the symbol of the gene.
        This will be used to refer to specific genes and to connect the loci
        to observed expression for that gene. Make sure to
        use identical symbology in TSS labeling as in the expression counts
        data of your multiome expriment. If multiple loci have the same symbol,
        or a gene has muliple loci, only the first encountered will be used. To
        disambiguate symbol-loci mapping, use a single canonical splice variant for each
        gene.

    max_distance : float > 0, default = 6e5
        Maximum distance to give a distance between a peak and a gene. All distances
        exceeding this threshold will be set to infinity.
    promoter_width : Width of the "promoter" region around each TSS, in base pairs. 
        The distance between a gene and a peak inside another gene's promoter region 
        is set to infinity. For PR modeling, this masks the effect of other 
        genes' promoter accessibility on the RP model.
    genome_file : str
        String, file location of chromosome lengths for you organism. For example:

        chr1	248956422
        chr2	242193529
        chr3	198295559
        chr4	190214555

    Returns
    -------
    adata : anndata.AnnData
        `.varm["distance_to_TSS"] : scipy.spmatrix[float] of shape (n_genes x n_peaks)
            Distance between genes' TSS and and peaks. 
        `.uns["distance_to_TSS_genes"] : np.ndarray[str] of shape (n_genes,)
            Gene symbols corresponding to rows in the `distance_to_TSS` matrix.

    Examples
    --------

    One can download mm10 or hg38 TSS annotations via:

    .. code-block :: python

        >>> mira.datasets.mm10_tss_data() # or mira.datasets.hg38_tss_data()
        ...   INFO:mira.datasets.datasets:Dataset contents:
        ...       * mira-datasets/mm10_tss_data.bed12


    Then, to annotate the ATAC peaks:

    .. code-block:: python

        >>> atac_data.var
        ...                        chr   start     end
        ...    chr1:9778-10670     chr1    9778   10670
        ...    chr1:180631-181281  chr1  180631  181281
        ...    chr1:183970-184795  chr1  183970  184795
        ...    chr1:190991-191935  chr1  190991  191935
        >>> mira.tl.get_distance_to_TSS(atac_data, 
        ...                        tss_data = "mira-datasets/mm10_tss_data.bed12", 
        ...                        gene_chrom='chrom', 
        ...                        gene_strand='strand', 
        ...                        gene_start='chromStart',
        ...                        gene_end='chromEnd',
        ...                        genome_file = '~/genomes/hg38/hg38.genome'
        ...                    )
        ...    WARNING:mira.tools.connect_genes_peaks:71 regions encounted from unknown chromsomes: KI270728.1,GL000194.1,GL000205.2,GL000195.1,GL000219.1,KI270734.1,GL000218.1,KI270721.1,KI270726.1,KI270711.1,KI270713.1
        ...    INFO:mira.tools.connect_genes_peaks:Finding peak intersections with promoters ...
        ...    INFO:mira.tools.connect_genes_peaks:Calculating distances between peaks and TSS ...
        ...    INFO:mira.tools.connect_genes_peaks:Masking other genes' promoters ...
        ...    INFO:mira.adata_interface.rp_model:Added key to var: distance_to_TSS
        ...    INFO:mira.adata_interface.rp_model:Added key to uns: distance_to_TSS_genes

    '''

    #load genome
    genome = Genome.from_file(genome_file)

    for c in np.unique(chrom):
        assert (c in genome.chromosomes), 'Chromosome {} from TSS data not found in genome file.'.format(str(c))

    #get gene promoters
    promoter_width/=2

    if not len(np.unique(gene_id)) == len(gene_id):
        logger.warn('Gene IDs are not unique! When searching for peak annotations for a duplicated ID, the first found will be used.')

    promoters = []
    for chrom, gene_id, start, end, strand in zip(chrom, gene_id, start, end, strand):

        start, end = int(start), int(end)
        
        tss = start, start + 1
        if strand == "-":
            tss = end - 1, end

        newregion = StrandedRegion(
                            chrom, 
                            max(0, tss[0] - promoter_width), 
                            min(tss[1] + promoter_width, genome.get_chromlen(chrom)),
                            strand = strand, 
                            annotation = gene_id
                            )
                        
        newregion._tx = (start, end)
        promoters.append(newregion)
    
    promoter_set = RegionSet(promoters, genome)

    #get regions
    peak_regions = check_region_specification(peaks, genome)
    peak_set = RegionSet(peak_regions, genome)

    sort_map = np.array([r.annotation for r in peak_set.regions])

    distance_matrix = get_masked_distances(promoter_set, peak_set, max_distance)

    unsorted_distance_matrix = project_sparse_matrix(
        distance_matrix,
        np.hstack([np.arange(len(sort_map))[:, np.newaxis], sort_map[:, np.newaxis]]),
        len(peaks),
    )

    gene_meta = [
        (r.annotation, r.chromosome, r._tx[0], r._tx[1], r.strand)
        for r in promoter_set.regions
    ]

    return (unsorted_distance_matrix, *list(zip(*gene_meta)))
