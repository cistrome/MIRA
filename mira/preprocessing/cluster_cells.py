import scanpy as sc
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

def get_cluster_barcodes(
    min_peaks = 1000,
    max_counts = 1e5,
    min_frip = 0.5,
    resolution = 0.1,
    min_fragments = 4e6,
    num_components = 15,*,
    data, outdir, report_dir
    ):

    data = anndata.read_h5ad(data)

    sc.pp.filter_cells(data, min_genes=100)
    sc.pp.calculate_qc_metrics(data, log1p=False, inplace=True)
    data.obs['FRIP'] = data.obs.total_counts/(data.obs.n_background_peaks + data.obs.total_counts)

    ax = sns.scatterplot(data = data.obs, s = 0.2, color = 'black', palette='magma_r',
                        x='total_counts', y = 'n_genes_by_counts', hue = 'FRIP')
    ax.set(yscale = 'log', xscale = 'log')
    ax.vlines(max_counts, ymin = 0, ymax = 5e4, color = 'black')
    ax.hlines(min_peaks, xmin = 0, xmax = 1e5, color = 'black')
    sns.despine()
    plt.savefig(os.path.join(report_dir, 'readdepth_distribution.png'))

    del ax
    ax = sns.kdeplot(data = data.obs, x = 'FRIP')
    ax.vlines(min_frip, ymin = 0, ymax = 1, color = 'black')
    sns.despine()
    plt.savefig(os.path.join(report_dir, 'FRIP.png'))
    del ax

    data = data[(data.obs.n_genes_by_counts > min_peaks) \
                & (data.obs.total_counts < max_counts) \
                & (data.obs.FRIP > min_frip)
            ]

    data.layers['binary'] = data.X.copy()
    data.layers['binary'].data = np.ones_like(data.layers['binary'].data)

    lsi = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=50)),
        ('scaler', StandardScaler())
    ]).fit(data.layers['binary'])

    sns.lineplot(
        y = np.cumsum(lsi.steps[1][1].explained_variance_),
        x = np.arange(50)
    )
    sns.despine()
    os.path.join(report_dir, 'component_explained_variance.png')

    data.obsm['X_pca'] = lsi.transform(data.layers['binary'])[:, :num_components]

    sc.pp.neighbors(data)
    sc.tl.umap(data, min_dist=0.1)
    sc.tl.leiden(data, resolution = resolution)

    sc.pl.umap(data, color = 'leiden', show = False, frameon = False)
    plt.savefig(os.path.join(report_dir, 'leiden_UMAP.png'))

    groupsize = data.obs.groupby('leiden')['total_counts'].sum()
    groups = groupsize[groupsize >= min_fragments].index.values

    print('Found {} clusters.'.format(len(groups)))
    for group in groups:
        with open(os.path.join(outdir, '{}.barcodes.txt'.format(str(group))), 'w') as f:
            print(*data.obs_names[data.obs.leiden == group], sep = '\n', file = f)


def add_arguments(parser):

    parser.add_argument('--report-dir', '-r', required = True, type = str)
    parser.add_argument('--outdir', '-o', required = True, type = str)
    parser.add_argument('--peakcount-path', '-i', required = True, type = str)
    parser.add_argument('--min-peaks', default = 1000, type = int)
    parser.add_argument('--max-counts', default = 1e5, type = float)
    parser.add_argument('--min-frip', default = 0.5, type = float)
    parser.add_argument('--resolution', default = 0.1, type = float)
    parser.add_argument('--min-fragments-in-cluster', default = 4e6, type = float)
    parser.add_argument('--num-lsi-components', default = 15, type = int)
    
def main(args):

    get_cluster_barcodes(
        min_peaks = args.min_peaks,
        max_counts = args.max_counts,
        min_frip = args.min_frip,
        resolution = args.resolution,
        min_fragments = args.min_fragments_in_cluster,
        num_components = args.num_lsi_components,
        data = args.peakcount_path,
        outdir= args.outdir,
        report_dir=args.report_dir
    )