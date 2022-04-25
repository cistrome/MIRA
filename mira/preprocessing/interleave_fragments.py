
import os
import gzip

def interleave_sorted_frags(*fragment_streams):

    n_files = len(fragment_streams)

    def get_next(stream):
        try:
            return next(stream)
        except StopIteration:
            return None
    
    next_element = [next(stream) for stream in ]

