
from lisa.core import genome_tools
from lisa.core.genome_tools import Region, RegionSet, Genome
from collections import Counter
import logging
import numpy as np
from scipy.sparse import spdiags, csr_matrix, coo_matrix
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

    #load genome
    genome = Genome.from_file(genome_file)

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
            
        promoters.append(StrandedRegion(
                            chrom, 
                            max(0, tss[0] - promoter_width), 
                            min(tss[1] + promoter_width, genome.get_chromlen(chrom)),
                            strand = strand, 
                            annotation = gene_id
                            )
                        )
    
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

    return unsorted_distance_matrix, [r.annotation for r in promoter_set.regions]