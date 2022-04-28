
import yaml
import argparse
import os
from collections import defaultdict
import sys
import subprocess
from .fastq_parser import parse_fasta_directory


def add_arguments(subparser):

    map_parser = subparser.add_argument_group('Mapping')
    map_parser.add_argument('--chromap-index', '-x', type = str, required=True)
    map_parser.add_argument('--chromap-cores', '-cc', type= int, default = 8)

    genome_parser = subparser.add_argument_group('Genome')
    genome_parser.add_argument('--fasta', '-fa', type = str, required=True)
    genome_parser.add_argument('--chrom-sizes','-cs', type = str, 
            help = 'Genome file (or chromlengths file).', required = True)
    genome_parser.add_argument('--genome-size', '-gs', required = True, choices = ['mm','hs'])

    peaks_parser = subparser.add_argument_group('Peaks')
    peaks_parser.add_argument('--slop-distance','-slop', type = int, default = 250)
    peaks_parser.add_argument('--use-all-frags', default = False, action = 'store_true')
    peaks_parser.add_argument('--max-fraglen', default = 150, type = int)

    barcode_parser = subparser.add_argument_group('Barcode')
    barcode_parser.add_argument('--whitelist', '-w', type = str, required=True)
    barcode_parser.add_argument('--whitelist-match','-wm', type = str, required=True)
    barcode_parser.add_argument('--barcode-start', '-bS', type = int, default = 8)
    barcode_parser.add_argument('--barcode-length', '-bL', type = int, default = 16)
    barcode_parser.add_argument('--barcode-reversed', '-bR', default = False, action = 'store_true')

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
        os.path.dirname(__file__), 'pipelines', 'preprocess-atac.Snakefile'
    )

    config_path = os.path.join(
        os.path.dirname(__file__), 'config.yaml'
    )

    data_dict = defaultdict(dict)

    for data in args.data:
        data_dict[data[0]][data[1]] = {
            'fastqs' : dict(parse_fasta_directory(data[2], ['R1','R2','R3']))
        }

    data_dict = dict(data_dict)

    config_dict = dict(
        directory = args.results_dir,
        peaks = dict(
            short_fragments = not args.use_all_frags,
            max_fraglen = args.max_fraglen,
            slop_distance = args.slop_distance,
            genome_size = args.genome_size
        ),
        mapping = dict(
            index = args.chromap_index,
            cores = args.chromap_cores,
        ),
        genome = dict(
            fasta = args.fasta,
            chrom_sizes = args.chrom_sizes
        ),
        barcode = dict(
            barcode_start = args.barcode_start,
            barcode_length = args.barcode_length,
            reversed = args.barcode_reversed,
            whitelist_match = args.whitelist_match,
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
