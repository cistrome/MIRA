
import subprocess

def callpeaks(
    q_value = 0.05,
    ext_size = 50,*,
    genome_size,
    outdir,
    outname,
    input_fragments
    ):

   cmd = [
        'macs2','callpeak','-f','BEDPE',
        '-g', st(genome_size),
        '--outdir', outdir,
        '-n', outname, 
        '-B','-q', str(q_value),
        '--no-model','--extsize', str(ext_size),
        '--SPMR','--keep-dup','all',
        '-t', str(input_fragments)
    ]

    subprocess.run(cmd, check_output = True)

def add_arguments(parser):

    parser.add
