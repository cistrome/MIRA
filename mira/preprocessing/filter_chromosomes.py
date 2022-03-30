import argparse
import sys

def _get_whitelist_chroms(genome):
    return [g.strip().split('\t')[0] for g in genome]

def _apply_filter(fragments, genome):

    chroms = _get_whitelist_chroms(genome)

    for fragment in fragments:
        fields = fragment.strip().split('\t')
        if fields[0] in chroms:
            yield fragment.strip()


def add_arguments(parser):

    parser.add_argument('fragments',type=argparse.FileType('r'))
    parser.add_argument('genome', type = argparse.FileType('r'))
    

def main(args):

    for fragment in _apply_filter(args.fragments, args.genome):
        print(fragment, file = sys.stdout)