import argparse
import sys

def _get_allowed_barcodes(barcode_input):
    return {
        barcode : True for barcode in map(lambda x : x.strip(), barcode_input)
    }

def _apply_filter(fragments, allowed_values, colnum):

    values_dict = _get_allowed_barcodes(allowed_values)

    for fragment in fragments:
        fields = fragment.strip().split('\t')
        if fields[colnum] in values_dict:
            yield fragment.strip()


def add_arguments(parser):

    parser.add_argument('fragments',type=argparse.FileType('r'))
    parser.add_argument('barcodes', type = argparse.FileType('r'))
    

def main(args):

    for fragment in _apply_filter(args.fragments, args.barcodes, 3):
        print(fragment, file = sys.stdout)