
import glob
import re
import os
from collections import defaultdict

def parse_fasta_directory(_dir, expected_reads):

    illumina_filename = r'(?P<sample_name>.+)_S(?P<sample_number>\d+)_L(?P<lane>\d+)_(?P<read_no>R\d+)_001.fastq.gz'
    illum_re = re.compile(illumina_filename)

    fastqs = []
    for filename in os.listdir(_dir):
        matches = illum_re.search(filename)
        if not matches is None:
            fastqs.append({**matches.groupdict(), 'filename' : filename})
    
    assert len(fastqs) > 0, 'No illumina fastq files found in {}'.format(_dir)

    # make sure only one sample present
    assert all([fastq['sample_number'] == fastqs[0]['sample_number']
                for fastq in fastqs
            ]), 'Only one sample may be parsed from a directory. Make sure samples are organized by directory e.g.: <sample>/<fastqs>'

    fastq_hierarchy = defaultdict(lambda : list())
    for fastq in sorted(fastqs, key = lambda x : int(x['lane'])):

        fastq_hierarchy[fastq['read_no']].append(
            os.path.join(_dir, fastq['filename'])
        )

    #make sure all reads are accounted for
    for read_no in expected_reads:
        assert read_no in fastq_hierarchy, \
            '{} reads not found in {}'.format(read_no, _dir)

    # make sure all lanes are accounted for
    assert all([
        len(fastq_hierarchy[read_no]) == len(fastq_hierarchy['R1'])
        for read_no, lanes in fastq_hierarchy.items()
    ]), 'Some fastqs are missing lanes in {}'.format(_dir)

    return fastq_hierarchy