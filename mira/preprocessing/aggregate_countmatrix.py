import subprocess
import logging
logger = logging.getLogger('mira.peakcount')
logger.setLevel(logging.INFO)
from lisa.core import genome_tools as gt
from scipy import sparse
import anndata
import pandas as pd

def _check(genome, region):
    try:
        genome.check_region(region)
        return True
    except gt.NotInGenomeError:
        return False

def count_peaks(*,fragment_file, peaks_file, genome_file):
    
    command = ' '.join([
        'bedtools','intersect',
        '-a', fragment_file,
        '-b', peaks_file, 
        '-wa','-wb','-sorted', '-loj'
    ])
    
    logger.warning('Command: ' + command)
    
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        shell=True, 
        stderr=subprocess.PIPE,
    )

    barcode_dict = {}
    
    genome = gt.Genome.from_file(genome_file)
    peaks = gt.Region.read_bedfile(peaks_file)
    peaks = [peak for peak in peaks if _check(genome, peak)]
    peaks = gt.RegionSet(peaks, genome)
    peak_dict = {r.to_tuple() : i for i,r in enumerate(peaks.regions)}
    peak_dict['background'] = len(peak_dict)
    
    peak_indices, barcode_indices, counts = [],[],[]
    
    i = 0
    while process.stdout.readable():
        line = process.stdout.readline()

        if not line:
            break
        else:
            i+=1
            
            line = line.decode().strip().split('\t')

            frag_chrom, frag_start, frag_end, barcode, count, \
                peak_chrom, peak_start, peak_end = line[:8]
            
            if peak_chrom == '.' or peak_start == '-1':
                peak_idx = peak_dict['background']
            else:
                peak_idx = peak_dict[(peak_chrom, int(peak_start), int(peak_end))]
                
            if barcode in barcode_dict:
                barcode_idx = barcode_dict[barcode]
            else:
                barcode_idx = len(barcode_dict)
                barcode_dict[barcode] = len(barcode_dict)
            
            peak_indices.append(peak_idx)
            barcode_indices.append(barcode_idx)
            counts.append(int(count))
            
            if i%5000000 == 0:
                logger.warning('Processed {} million fragments ...'.format(str(i//1e6)))

    if not process.poll() == 0:
        raise Exception('Error while scanning for motifs: ' + process.stderr.read().decode())
    
    logger.warning('Done reading fragments.')
    logger.warning('Formatting counts matrix ...')
    return sparse.coo_matrix((counts, (barcode_indices, peak_indices)), 
        shape = (len(barcode_dict), len(peaks) + 1)).tocsr(), list(barcode_dict.keys()), list(peak_dict.keys()) 


def format_adata(mtx, barcodes, peaks):

    data = anndata.AnnData(
        X = mtx,
        obs = pd.DataFrame(index = barcodes),
        var = pd.DataFrame(index = 
        ['{}:{}-{}'.format(*map(str, peak)) if not peak is 'background' else peak 
         for peak in peaks]),
    )

    data.obs['n_background_peaks'] = data.obs_vector('background')
    data = data[:, data.var_names != 'background']

    return data

def add_arguments(parser):
    #fragment_file, peak_file, genome_file
    parser.add_argument('--fragments','-f', type = str, required=True,
        help = 'Fragment file, may be gzipped')
    parser.add_argument('--peaks','-p', type = str, required=True,
        help = 'File of peaks from which to assemble count matrix. Just need (chr, start, end) columns.')
    parser.add_argument('--genome-file','-g', type = str, 
        help = 'Genome file (or chromlengths file).', required = True)
    parser.add_argument('--outfile','-o',required=True, type = str,
        help = 'Output filename for adata object.')

def main(args):

    data = format_adata(
        *count_peaks(
            fragment_file=args.fragments,
            peaks_file = args.peaks,
            genome_file=args.genome_file,
        )
    )

    data.write_h5ad(args.outfile)