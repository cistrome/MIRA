
import os
import subprocess

def countmatrix(*,fragment_file, peaks_file, genome_file, outfile,
    chrom_match_string = "^(chr)[(0-9)|(X,Y)]+$"):

    size = os.path.getsize(fragment_file)

    with open(fragment_file, 'r') as input_stream:

        tqdm_command = subprocess.Popen(
            ['tqdm','--total', str(size), '--bytes',
            '--desc', 'Aggregating fragments'],
            stdin= input_stream, stdout= subprocess.PIPE,
        )

        gzip_command = subprocess.Popen(
            ['gzip','-d','-c'],
            stdin= tqdm_command.stdout, stdout=subprocess.PIPE,
        )
        
        filter_command = subprocess.Popen(
            ['mira-preprocess','filter-chroms','-',
            '--genome', genome_file,
            '--chr-match-string', chrom_match_string],
            stdin = gzip_command.stdout, stdout= subprocess.PIPE,
        )

        intersect_command = subprocess.Popen(
            ['bedtools','intersect','-a','-','-b', peaks_file,
            '-loj','-sorted','-wa','-wb'],
            stdin = filter_command.stdout, stdout = subprocess.PIPE,
        )

        count_command = subprocess.Popen(
            ['mira-preprocess','count-intersection',
            '-', '-g',genome_file, '-o', outfile],
            stdin= intersect_command.stdout, stdout= subprocess.PIPE
        )

        count_command.communicate()
        intersect_command.wait()
        filter_command.wait()
        gzip_command.wait()

def main(args):

    countmatrix(
        fragment_file = args.fragment_file,
        peaks_file = args.peaks_file,
        genome_file = args.genome_file,
        outfile = args.outfile,
    )

def add_arguments(parser):
    #fragment_file, peak_file, genome_file
    parser.add_argument('--fragment-file','-f', type = str, required = True)
    parser.add_argument('--genome-file','-g', type = str, 
        help = 'Genome file (or chromlengths file).', required = True)
    parser.add_argument('--peaks-file','-p', required = True, type = str)
    parser.add_argument('--outfile','-o',required=True, type = str,
        help = 'Output filename for adata object.')