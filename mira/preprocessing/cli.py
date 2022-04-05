import argparse
import os
import sys
import subprocess

from mira.preprocessing import iterative_merge, aggregate_countmatrix, \
        callpeaks, filter_fragment_barcodes, cluster_cells, \
        filter_chromosomes

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help = 'commands')

def add_subcommand(definition_file, cmd_name):

    subparser = subparsers.add_parser(cmd_name)
    definition_file.add_arguments(subparser)
    subparser.set_defaults(func = definition_file.main)

add_subcommand(filter_fragment_barcodes, 'filter-barcodes')
add_subcommand(callpeaks, 'call-peaks')
add_subcommand(iterative_merge, 'merge-peaks')
add_subcommand(aggregate_countmatrix, 'agg-countmatrix')
add_subcommand(cluster_cells, 'cluster-cells')
add_subcommand(filter_chromosomes, 'filter-chroms')

def run_snakemake_pipeline(args):

    try:
        import snakemake
        import leidenalg
        import scanpy
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError('To run the pipeline, user must also install snakemake, leiden-alg, and scanpy packages.')

    snakefile_path = os.path.join(
        os.path.dirname(__file__), 'Snakefile'
    )

    snake_config = dict(
        sample = args.sample_name,
        fragment_file = args.fragments,
        genome_file = args.genome_file,
        genome_size = args.genome_size,
        slop_distance = args.slop_distance,
        ext_size = args.ext_size,
        q_value = args.q_value,
        max_fraglen = args.max_fraglen,
        min_peaks = args.min_peaks,
        max_counts = args.max_counts,
        min_frip = args.min_frip,
        leiden_resolution = args.leiden_resolution,
        min_fragments_in_cluster = args.min_fragments_in_cluster,
        num_lsi_components = args.num_lsi_components,
    )

    snake_args = [
        'snakemake',
        '-s', snakefile_path,
        '--config', *['{}={}'.format(k, v) for k,v in snake_config.items()],
        *args.snake_args,
    ]

    subprocess.run(
        snake_args, check=True
    )

subparser = subparsers.add_parser('run-pipeline')
subparser.add_argument('--sample-name', '-n', type = str, required = True)
subparser.add_argument('--fragments','-f', type = str, required=True,
        help = 'Fragment file, may be gzipped')
subparser.add_argument('--genome-file','-g', type = str, 
        help = 'Genome file (or chromlengths file).', required = True)
subparser.add_argument('--genome-size', required = True)
subparser.add_argument('--slop-distance','-d', type = int, default = 250)
subparser.add_argument('--ext-size','-e', type = int, default = 50)
subparser.add_argument('--q-value', '-q', type = float, default = 0.05)
subparser.add_argument('--max-fraglen', default = 150, type = int)
subparser.add_argument('--min-peaks', default = 1000, type = int)
subparser.add_argument('--max-counts', default = 1e5, type = float)
subparser.add_argument('--min-frip', default = 0.5, type = float)
subparser.add_argument('--leiden-resolution', default = 0.1, type = float)
subparser.add_argument('--min-fragments-in-cluster', default = 10e6, type = float)
subparser.add_argument('--num-lsi-components', default = 15, type = int)
subparser.add_argument('--snake-args','-s',nargs=argparse.REMAINDER)

subparser.set_defaults(func = run_snakemake_pipeline)

def main():
    #____ Execute commands ___

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        print(parser.print_help(), file = sys.stderr)
    else:
        args.func(args)
