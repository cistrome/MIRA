import subprocess

def callpeaks(
    q_value = 0.05,
    ext_size = 50,*,
    genome_size,
    outdir,
    outname,
    fragment_file,
    ):

    cmd = [
        'macs2','callpeak','-f','BEDPE',
        '-g', str(genome_size),
        '--outdir', outdir,
        '-n', outname, 
        '-B','-q', str(q_value),
        '--nomodel','--extsize', str(ext_size),
        '--SPMR','--keep-dup','all',
        '-t', str(fragment_file)
    ]

    subprocess.run(cmd, check = True)

def add_arguments(parser):

    parser.add_argument('--fragment-file','-f', type = str,required=True)
    parser.add_argument('--outdir', '-d', required = True, type = str)
    parser.add_argument('--name','-n',required=True, type = str)
    parser.add_argument('--genome-size','-g', required = True)
    parser.add_argument('--ext-size','-e', type = int, default = 50)
    parser.add_argument('--q-value', '-q', type = float, default = 0.05)
    
def main(args):

    callpeaks(
        q_value= args.q_value,
        ext_size=args.ext_size,
        genome_size=args.genome_size,
        outdir=args.outdir,
        outname = args.name,
        fragment_file=args.fragment_file,
    )
