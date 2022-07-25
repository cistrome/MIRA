import sys
import argparse

def label_fragments(fragment_stream,*, batch, sample):

    for fragment in fragment_stream:
        line = fragment.strip().split('\t')
        assert(len(line) >= 3)
        
        barcode = line[3]

        barcode = '@{batch}:{sample}:{barcode}'.format(
            batch = str(batch), sample = str(sample), barcode = barcode
        )

        yield '\t'.join(
            [*line[:3], barcode, *line[4:]]
        )

def add_arguments(parser):

    parser.add_argument('--fragment-file', '-f', type = argparse.FileType('r'),
        required=True)
    parser.add_argument('--batch','-b', type = str, required = True)
    parser.add_argument('--sample','-s', type = str, required = True)
    parser.add_argument('--outfile', '-o', type = argparse.FileType('w'),
        default = sys.stdout)

def main(args):

    for line_out in label_fragments(
        args.fragment_file, batch = args.batch, sample = args.sample,
    ):
        print(line_out, file = args.outfile)