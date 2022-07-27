import argparse
import logging
logger = logging.getLogger('mira.peakcount')
logger.setLevel(logging.INFO)
from scipy import sparse
import anndata
import pandas as pd
import sys

def count_peaks(*, input_stream, genome_file, 
    num_fragment_file_columns = 5):
    
    assert(num_fragment_file_columns>=5)

    barcode_dict = {}
    
    peak_dict = {'background' : 0}
    
    peak_indices, barcode_indices, counts = [],[],[]
    
    i = 0
    for i, line in enumerate(input_stream):
            
        line = line.strip().split('\t')

        frag_chrom, frag_start, frag_end, barcode, count = line[:5]
        peak_chrom, peak_start, peak_end = line[num_fragment_file_columns:num_fragment_file_columns+3]
        
        if peak_chrom == '.' or peak_start == '-1':
            peak_idx = peak_dict['background']
        else:

            peak_key = (peak_chrom, int(peak_start), int(peak_end))

            if peak_key in peak_dict:
                peak_idx = peak_dict[peak_key]
            else:
                peak_idx = len(peak_dict)
                peak_dict[peak_key] = peak_idx
            
        if barcode in barcode_dict:
            barcode_idx = barcode_dict[barcode]
        else:
            barcode_idx = len(barcode_dict)
            barcode_dict[barcode] = barcode_idx
        
        peak_indices.append(peak_idx)
        barcode_indices.append(barcode_idx)
        counts.append(int(count))
        
        if i%5000000 == 0:
            pass
            #logger.warning('Processed {} million fragments ...'.format(str(i//1e6)))
    
    logger.info('Done reading fragments.')
    logger.info('Formatting counts matrix ...')
    return sparse.coo_matrix((counts, (barcode_indices, peak_indices)), 
        shape = (len(barcode_dict), len(peak_dict))).tocsr(), list(barcode_dict.keys()), list(peak_dict.keys()) 


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
    parser.add_argument('input', type = argparse.FileType('r'), default = sys.stdin)
    parser.add_argument('--genome-file','-g', type = str, 
        help = 'Genome file (or chromlengths file).', required = True)
    parser.add_argument('--outfile','-o',required=True, type = str,
        help = 'Output filename for adata object.')


def main(args):

    data = format_adata(
        *count_peaks(
            input_stream=args.input,
            genome_file=args.genome_file,
        )
    )

    data.write_h5ad(args.outfile)