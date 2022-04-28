import numpy as np
from lisa.core.genome_tools import Region
import tqdm

def _plot_rp_model_tails(ax, start_pos, left_decay, right_decay, color = 'lightgrey', linecolor = 'black',
        linewidth = 1, alpha = 0.25, bin_size = 50):

    left_extent = int(10*left_decay)
    left_x = np.arange(0, left_extent, left_extent//bin_size)
    left_y = 0.5**(left_x / left_decay)

    right_extent = int(10*right_decay)
    right_x = np.arange(0, right_extent, right_extent//bin_size)
    right_y = 0.5**(right_x / right_decay)

    x, y = np.concatenate([-left_x[::-1] - 1500, right_x + 1500]) + start_pos, np.concatenate([left_y[::-1], right_y])

    ax.fill_between(x, y, y2 = 0, color = color, alpha = alpha)
    ax.plot(x, y, color = linecolor, linewidth = linewidth)
    

def _plot_rp_models(ax, color = 'lightgrey', linecolor = 'black', 
        linewidth = 1, alpha = 0.25, bin_size = 50, *, 
        interval_chrom, interval_start, interval_end, rp_models, 
        gene_id, chrom, start, end, strand):
    
    TSS_data = {
        gene : tuple(data)
        for gene, data in zip(gene_id, zip(chrom, start, end, strand))
    }
    
    interval = Region(interval_chrom, interval_start, interval_end)
                      
    for model in tqdm.tqdm(rp_models.models, desc = 'Intersecting RP models with interval'):
        if model.gene in TSS_data.keys():
            
            gene_chrom, gene_start, gene_end, gene_strand = TSS_data[model.gene]
            rp_params = model._get_normalized_params()
            
            upstream, downstream = 1e3 * rp_params['distance']

            left, right, start_pos = upstream, downstream, gene_start
            if gene_strand == '-':
                left, right, start_pos = downstream, upstream, gene_end

            
            gene_bounds = Region(gene_chrom, start_pos - 10*left - 1500, start_pos + 10*right + 1500)
            
            if gene_bounds.overlaps(interval):
                _plot_rp_model_tails(ax, start_pos, left, right, 
                    color = color, alpha = alpha, linecolor = linecolor, linewidth = linewidth, 
                    bin_size = bin_size)