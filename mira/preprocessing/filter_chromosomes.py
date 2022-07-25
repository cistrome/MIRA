import argparse
import sys
import re

def _get_whitelist_chroms(genome, match_string):

    chrom_match = re.compile(match_string)

    return [
        chromosome for chromosome in map(lambda x : x.strip().split('\t')[0], genome)
        if not chrom_match.search(chromosome) is None
    ]

def _apply_filter(fragments, genome, chr_match_string = '^(chr)[(0-9)|(X,Y)]+$'):

    chroms = _get_whitelist_chroms(genome, chr_match_string)

    for fragment in fragments:
        fields = fragment.strip().split('\t')
        if fields[0] in chroms:
            yield fragment.strip()


def add_arguments(parser):

    parser.add_argument('fragments',type=argparse.FileType('r'))
    parser.add_argument('genome', type = argparse.FileType('r'))
    parser.add_argument('--chr-match-string','-match', type = str, default = "^(chr)[(0-9)|(X,Y)]+$")
    

def main(args):

    for fragment in _apply_filter(args.fragments, args.genome, args.chr_match_string):
        print(fragment, file = sys.stdout)