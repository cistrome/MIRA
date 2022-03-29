
import argparse
from lisa.core import genome_tools as gt
import tqdm
import numpy as np

def check(genome, region):
    try:
        genome.check_region(region)
        return True
    except gt.NotInGenomeError:
        return False

def slop(self, d, genome):
    return gt.Region(self.chromosome, max(0, self.start - d), min(self.end + d, genome.get_chromlen(self.chromosome)),
                    annotation = self.annotation)
    
        
def add_source(self, source):
    self.source = source
    return self
        
gt.Region.slop = slop
gt.Region.add_source = add_source


def _slop_summits(summit_files, genome, distance = 250):

    summits = [
        r.slop(distance, genome).add_source(summit_file)
        for summit_file in summit_files for r in gt.Region.read_bedfile(summit_file)
        if check(genome, r)
    ]

    return summits

def _iou(r1, r2, threshold = 0.2):
    
    intersection = r1._overlap_distance(r1.start, r1.end, r2.start, r2.end)
    union = len(r1) + len(r2) - intersection
    
    return int(intersection/union > threshold)
    

def _merge(*, overlap_peaks, summit_set, genome):

    significance = np.array([-float(r.annotation[1]) for r in summit_set.regions]).argsort()

    blacklist = {}
    peaklist = []
    for i in tqdm.tqdm(significance, desc = 'Iteratively merging peaks'):
    
        if not i in blacklist:
            
            overlaps_idx = overlap_peaks[i,:].indices
            overlap_regions = []
            for j in overlaps_idx:
                if not j in blacklist:
                    blacklist[j]=True

            peaklist.append(summit_set.regions[i])
            blacklist[i] = True

    peaklist = gt.RegionSet(peaklist, genome)

    return peaklist


def iterative_merge(
    slop_distance = 250,*,
    summit_files,
    genome_file,
    ):

    genome = gt.Genome.from_file(genome_file)

    summits = _slop_summits(summit_files, genome, distance = slop_distance)

    summit_set = gt.RegionSet(summits, genome)
    overlap_peaks = summit_set.map_intersects(summit_set, distance_function=_iou)

    overlap_peaks.eliminate_zeros()
    overlap_peaks = overlap_peaks.tocsr()

    peaks = _merge(
        overlap_peaks = overlap_peaks, 
        summit_set = summit_set, 
        genome = genome)

    return peaks


def add_arguments(parser):

    parser.add_argument('--summit-files', '-s', nargs = '+', type = str,
        help = 'List of MACS summit files to merge', required = True)
    parser.add_argument('--genome-file','-g', type = str, 
        help = 'Genome file (or chromlengths file).', required = True)
    parser.add_argument('--slop-distance','-d', type = int, default = 250)
    parser.add_argument('--outfile','-o', type = argparse.FileType('w'),
        required = True, help = 'Output filename for merged peakset.')

def main(args):

    peaklist = iterative_merge(
        slop_distance=args.slop_distance,
        summit_files=args.summit_files,
        genome_file= args.genome_file
    )

    for peak in peaklist.regions:
        print(peak, file = args.outfile)