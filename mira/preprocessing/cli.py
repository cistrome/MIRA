import argparse
import os
import sys
import subprocess

from mira.preprocessing import iterative_merge, aggregate_countmatrix, \
        callpeaks, filter_fragment_barcodes, cluster_cells, \
        filter_chromosomes, label_fragments, interleave_fragments, \
        run_atac_pipeline, run_rna_pipeline, format_adata

class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

parser = MyArgumentParser(
    fromfile_prefix_chars = '@',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(help = 'commands')

def add_subcommand(definition_file, cmd_name):

    subparser = subparsers.add_parser(cmd_name)
    definition_file.add_arguments(subparser)
    subparser.set_defaults(func = definition_file.main)

add_subcommand(run_atac_pipeline, 'atac-pipeline')
add_subcommand(run_rna_pipeline, 'rna-pipeline')
add_subcommand(filter_fragment_barcodes, 'filter-barcodes')
add_subcommand(callpeaks, 'call-peaks')
add_subcommand(iterative_merge, 'merge-peaks')
add_subcommand(aggregate_countmatrix, 'agg-countmatrix')
add_subcommand(cluster_cells, 'cluster-cells')
add_subcommand(filter_chromosomes, 'filter-chroms')
add_subcommand(label_fragments, 'label-fragments')
add_subcommand(interleave_fragments, 'interleave-fragments')
add_subcommand(format_adata, 'format-adata')

def main():
    #____ Execute commands ___

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        print(parser.print_help(), file = sys.stderr)
    else:
        args.func(args)
