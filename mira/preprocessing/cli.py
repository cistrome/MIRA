import argparse
from mira.preprocessing import iterative_merge, aggregate_countmatrix, callpeaks, filter_fragment_barcodes

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

def main():
    #____ Execute commands ___

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        print(parser.print_help(), file = sys.stderr)
    else:
        args.func(args)
