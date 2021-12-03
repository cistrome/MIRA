import os
import sys
import configparser
import subprocess
import requests
import zipfile
import io
import tempfile
import pyfaidx
import numpy as np
from scipy import sparse
import logging
from glob import glob
import tqdm
import re
import mira
from mira.adata_interface.core import wraps_functional
from mira.adata_interface.regulators import fetch_peaks, add_factor_hits_data
from functools import partial

logger = logging.getLogger(__name__)

'''config = configparser.ConfigParser()
configpath = __file__[:-3] + '_config.ini'
config.read(configpath)
assert(len(config.sections()) > 0), 'Did not read config file correctly.
config.get('data','motifs')'''

mira_dir = os.path.dirname(mira.__file__)
PWM_DIR = os.path.join(mira_dir, 'package_data/motifs/')
PWM_suffix = 'jaspar'

def validate_peaks(peaks):

    assert(isinstance(peaks, (list, np.ndarray)))
    if isinstance(peaks, np.ndarray):
        assert(peaks.dtype in [np.str, 'S'])
        peaks = peaks.tolist()

    for peak in peaks:
        assert(isinstance(peak, (list,tuple)))
        assert(len(peak) == 3)

        try:
            int(peak[1])
            int(peak[2])
            str(peak[0])
        except ValueError:
            raise Exception('Count not coerce peak {} into (<str chrom>, <int start>, <int end>) format'.format(str(peak)))

    return peaks

'''def convert_jaspar_to_moods_pfm(jaspar_file):

    from Bio.motifs import read

    with open(jaspar_file, 'r') as f:
        motif = read(f, 'jaspar')

    if not motif.name is None and not motif.matrix_id is None:
        factor_name = motif.name.upper()
        motif_id = motif.matrix_id
    else:
        factor_name, motif_id = os.path.basename(jaspar_file).replace('.jaspar','').replace('/','-')\
            .upper().split('_')

    new_motif_filename = os.path.join(config.get('data','motifs'), '{}_{}.{}'.format(
            motif_id, factor_name.replace('(','.').replace(')', ''), config.get('jaspar','pfm_suffix')
        ).replace('/', '-'))

    with open(new_motif_filename, 'w') as f:
        for nuc in ['A','C','G','T']:
            print(*list(map(int, motif.counts[nuc])), sep = '\t', file = f)


def __download_jaspar_motifs__(write_dir):

    if not os.path.isdir(config.get('data','root')):
        os.mkdir(config.get('data','root'))

    r = requests.get(config.get('jaspar','motifs_url'))

    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(write_dir)

        for jaspar_pfm in os.listdir(write_dir):
            convert_jaspar_to_moods_pfm(os.path.join(write_dir, jaspar_pfm))
        
    else:
        raise Exception('Error downloading motifs database from JASPAR')
        
def purge_motif_matrices():
    for matrix in list_motif_matrices():
        os.remove(matrix)'''

def get_motif_glob_str():
    return os.path.join(PWM_DIR, '*.{}'.format(PWM_suffix))

def list_motif_matrices():

    if not os.path.isdir(PWM_DIR):
        return []

    return list(glob(get_motif_glob_str()))


def list_motif_ids():
    return [os.path.basename(x).replace('.jaspar', '').split('_') for x in list_motif_matrices()]


def get_peak_sequences(peaks, genome, output_file):

    logger.info('Getting peak sequences ...')

    fa = pyfaidx.Fasta(genome)

    with open(output_file, 'w') as f:
        for i, (chrom,start,end) in tqdm.tqdm(enumerate(peaks)):
            
            try:
                peak_sequence = fa[chrom][int(start) : int(end)].seq
            except KeyError:
                peak_sequence = 'N'*(int(end) - int(start))

            print('>{idx}\n{sequence}'.format(
                    idx = str(i),
                    sequence = peak_sequence.upper()
                ), file = f, end=  '\n')


def get_motif_hits(peak_sequences_file, num_peaks, pvalue_threshold = 0.00005):

    logger.info('Scanning peaks for motif hits with p >= {} ...'.format(str(pvalue_threshold)))

    motifs_directory = PWM_DIR
    matrix_list = [
        os.path.basename(x) for x in
        glob(os.path.join(motifs_directory, '*.{}'.format(PWM_suffix)))
    ]

    command = ['moods-dna.py', 
        '-m', *matrix_list, 
        '-s', peak_sequences_file, 
        '-p', str(pvalue_threshold), 
        '--batch']

    logger.info('Building motif background models ...')
    process = subprocess.Popen(
        ' '.join(command), 
        stdout=subprocess.PIPE, 
        shell=True, 
        stderr=subprocess.PIPE, 
        cwd = motifs_directory
    )

    motif_matrices = matrix_list 
    motif_idx_map = dict(zip(motif_matrices, np.arange(len(motif_matrices))))

    motif_indices, peak_indices, scores = [],[],[]
    i=0
    while process.stdout.readable():
        line = process.stdout.readline()

        if not line:
            break
        else:
            if i == 0:
                logger.info('Starting scan ...')
            i+=1

            peak_num, motif, hit_pos, strand, score, site, snp = line.decode().strip().split(',')
            
            motif_indices.append(motif_idx_map[motif])
            peak_indices.append(peak_num)
            scores.append(float(score))

            if i%1000000 == 0:
                logger.info('Found {} motif hits ...'.format(str(i)))

    if not process.poll() == 0:
        raise Exception('Error while scanning for motifs: ' + process.stderr.read().decode())

    logger.info('Formatting hits matrix ...')
    return sparse.coo_matrix((scores, (peak_indices, motif_indices)), 
        shape = (num_peaks, len(motif_matrices))).tocsr().T.tocsr()


def _parse_motif_name(motif_name):
    return [x.upper() for x in re.split('[/::()-]', motif_name)][0]


@wraps_functional(fetch_peaks, partial(add_factor_hits_data, factor_type = 'motifs'), ['peaks'])
def get_motif_hits_in_peaks(peaks, pvalue_threshold = 0.0001,*, genome_fasta):

    peaks = validate_peaks(peaks)

    temp_fasta = tempfile.NamedTemporaryFile(delete = False)
    temp_fasta_name = temp_fasta.name
    temp_fasta.close()

    try:

        get_peak_sequences(peaks, genome_fasta, temp_fasta_name)

        hits_matrix = get_motif_hits(temp_fasta_name, len(peaks), pvalue_threshold = pvalue_threshold)

        ids, factors = list(zip(*list_motif_ids()))
        parsed_factor_names = list(map(_parse_motif_name, factors))

        return ids, factors, parsed_factor_names, hits_matrix

    finally:
        os.remove(temp_fasta_name)