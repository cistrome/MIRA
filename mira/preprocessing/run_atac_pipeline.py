
import yaml
import argparse
import os
from collections import defaultdict
import sys
import subprocess
from .fastq_parser import parse_fasta_directory


def add_arguments(subparser):

    def check_is_file(filename):

        if not os.path.isfile(filename):
            raise subparser.error("Invalid file name: {}".format(filename))
        
        return filename

    def check_macs_species(species):
        if not species in ['hs','ce','mm','dm'] or type(species) == int:
            raise subparser.error('Invalid MACS2 species name: {}'.format(species))
        return species

    map_parser = subparser.add_argument_group('Mapping')
    map_parser.add_argument('--chromap-cores', '-c', type= int, default = 8,
        help = 'Number of cores allocated to each chromap process.')
    map_parser.add_argument('--min-fragment-length', type = int, default = 30,
        help = 'Minimum fragment length expected to map, used by chromap to build the fastest possible sequence index.')

    genome_parser = subparser.add_argument_group('Genome')
    fasta_or_assembly = genome_parser.add_mutually_exclusive_group(required=True)
    fasta_or_assembly.add_argument('--fasta', '-fa', type = check_is_file)
    fasta_or_assembly.add_argument('--species','-sp', type = str)

    genome_parser.add_argument('--genome-size', '-gs', required = True, type = check_macs_species)

    peaks_parser = subparser.add_argument_group('Peaks')
    peaks_parser.add_argument('--slop-distance','-slop', type = int, default = 250)

    peak_exclusive_group = genome_parser.add_mutually_exclusive_group(required=False)
    peak_exclusive_group.add_argument('--use-all-frags', default = False, action = 'store_true')
    peak_exclusive_group.add_argument('--max-fraglen', default = 150, type = int)

    barcode_parser = subparser.add_argument_group('Barcode')
    barcode_parser.add_argument('--whitelist', '-w', type = check_is_file, required=True)
    barcode_parser.add_argument('--whitelist-match','-wm', type = check_is_file, required=True)

    barcode_parser.add_argument('--barcode-config', '-bcc', type = str, nargs = 4,
        action = 'append', default = [['all', '8','16','+']])

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

    barcode_configs = {}
    for barcode_config in args.barcode_config:
        batch, start, length, strand = [t(x) for t, x in zip([str, int, int, str], barcode_config)]
        assert strand in ['+','-']

        barcode_configs[batch] = {
            'start' : start, 'length' : length, 'strand' : strand
        }

    for batch in data_dict.keys():
        if not batch in barcode_configs:
            if not 'all' in barcode_configs:
                raise ValueError('Barcode not configured for batch {}, and no default given for batch "all". Please specify the barcode format for this batch.'.format(
                    batch
                ))

            print('Barcode not configured for batch {}, defaulting to "all" configuration'.format(batch))
            barcode_configs[batch] = barcode_configs['all']

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
            min_fragment_length = args.min_fragment_length,
            cores = args.chromap_cores,
        ),
        genome = dict(
            fasta = args.fasta,
            species = args.species,
        ),
        whitelist = dict(
            whitelist = args.whitelist,
            whitelist_match = args.whitelist_match,
        ),
        barcode = dict(
            **barcode_configs,
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
