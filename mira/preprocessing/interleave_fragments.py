
import gzip
import argparse
import sys

class BedFileRecord:

    def __init__(self, fields):
        self.fields = fields.strip().split('\t')
        self.chr, self.start, self.end = self.fields[:3]

    def __gt__(self, other):
        return self.chr > other.chr or \
            (self.chr == other.chr and self.start > other.start)

    def __ge__(self, other):
        return (self > other) or (self == other)

    def __eq__(self, other):
        return self.chr == other.chr and self.start == other.start

    def __str__(self):
        return '\t'.join(self.fields)


class PeekIterator:

    def __init__(self, iterator, _type):
        self._iterator = iterator
        self._depleted = False
        self._type = _type
        try:
            self._next = self._get_next()
        except StopIteration:
            self._depleted = True

    def _get_next(self):
        return self._type(next(self._iterator))

    def __next__(self):

        if self._depleted:
            raise StopIteration()            

        ret_value = self._next 
        try:
            self._next = self._get_next()
            assert self._next >= ret_value, \
                    'Input stream must be sorted or interleaving will not work!'

        except StopIteration:
            self._depleted = True
        
        return ret_value

    def peek(self):
        if self._depleted:
            raise StopIteration()

        return self._next

    def has_next(self):
        return not self._depleted

    def __eq__(self, other):
        return self.peek() == other.peek()

    def __gt__(self, other):
        return self.peek() > other.peek()


def open_stream(filename, is_gzipped):
    if is_gzipped:
        return gzip.open(filename)
    else:
        return open(filename, 'r')


def interleave_sorted_frags(*stream_handles):

    streams = [
        PeekIterator(stream, BedFileRecord)
        for stream in stream_handles
    ]

    while True:

        streams = [stream for stream in streams if stream.has_next()]
        
        if len(streams) == 0:
            break
        else:
            yield next(min(streams))


def add_arguments(parser):

    parser.add_argument('--fragment-files', '-f', nargs = '+', type = str,
        help = 'List of paths to fragment files', required = True)
    parser.add_argument('--is-gzipped', default = False, action = 'store_true')
    parser.add_argument('--outfile','-o', type = argparse.FileType('w'),
        default = sys.stdout, help = 'Output filename for interleaved fragment file.')

def main(args):

    try:

        stream_handles = [
            open_stream(filename, args.is_gzipped)
            for filename in args.fragment_files
        ]

        for fragment in interleave_sorted_frags(*stream_handles):
            print(fragment, file = args.outfile)
    
    finally:
        for stream in stream_handles:
            stream.close()