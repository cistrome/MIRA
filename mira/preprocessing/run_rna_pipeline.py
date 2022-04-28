
import yaml
import argparse
import os
from collections import defaultdict
import sys
import subprocess
from .fastq_parser import parse_fasta_directory

def add_arguments(subparser):

    map_parser = subparser.add_argument_group('Mapping')
    map_parser.add_argument('--star-index', '-x', type = str, required=True)
    map_parser.add_argument('--star-cores', '-sc', type = int, default = 12)
    map_parser.add_argument('--star-features', '-f', type = str, nargs = "+", 
        default = ['Gene'])

    barcode_parser = subparser.add_argument_group('Barcode')
    barcode_parser.add_argument('--whitelist', '-w', type = str, required=True)
    barcode_parser.add_argument('--barcode-start', '-bS', type = int, default = 1)
    barcode_parser.add_argument('--barcode-length', '-bL', type = int, default = 16)
    barcode_parser.add_argument('--umi-start', '-umiS', type = int, default = 17)
    barcode_parser.add_argument('--umi-length', '-umiL', type = int, default = 12)

    data_parser = subparser.add_argument_group('Data')
    data_parser.add_argument('--data', '-d', nargs = 3, type = str, action = 'append',
        help = 'Give multiple data directories as <batch-name> <sample-name> <directory>, \n'
            'where that directory contains the fastqs of the R1, R2, and R3 fastqs of that sample.')

    #data_parser.add_argument('--other-data', '-od', nargs = 4, type = str, action = 'append',
    #    help = 'Give multiple data samples as <batch-name> <sample-name> <read_num> <path>. \n'
    #        'Use this for non-Illumina-formatted data.')

    execution_parser = subparser.add_argument_group('Execution')
    execution_parser.add_argument('--results-dir', '-r', type = str, required = True)
    execution_parser.add_argument('--yes', '-y', action = 'store_true', default = False,
        help = 'Skip data confirmation, useful for submitting job to cluster or non-interactive use.')
    execution_parser.add_argument('--snake-args','-s',nargs=argparse.REMAINDER)


def main(args):

    snakefile_path = os.path.join(
        os.path.dirname(__file__), 'pipelines', 'preprocess-rna.SnakeFile'
    )

    config_path = os.path.join(
        os.path.dirname(__file__), 'config.yaml'
    )

    data_dict = defaultdict(dict)

    for data in args.data:
        data_dict[data[0]][data[1]] = {
            'fastqs' : dict(parse_fasta_directory(data[2], ['R1','R2']))
        }

    data_dict = dict(data_dict)

    config_dict = dict(
        directory = args.results_dir,
        mapping = dict(
            index = args.star_index,
            cores = args.star_cores,
            features = args.star_features
        ),
        barcode = dict(
            barcode_start = args.barcode_start,
            barcode_length = args.barcode_length,
            umi_start = args.umi_start,
            umi_length = args.umi_length,
            whitelist = args.whitelist,
        ),
        data = data_dict,
    )

    if not args.yes:
        print('Please inspect this configuration to make sure that files were parsed correctly:\n')
        print(yaml.dump(data_dict))

        while True:
            answer = input('Proceed [y/n]: ')
            if answer == 'y' or answer == 'Y':
                break
            elif answer == 'n' or answer == 'N':
                sys.exit()

    with open(config_path, 'w') as f:
        print(yaml.dump(config_dict), file = f)

    snake_args = [
        'snakemake',
        '-s', snakefile_path,
        '--configfile', config_path,
        *args.snake_args,
    ]

    subprocess.run(
        snake_args, check=True
    )