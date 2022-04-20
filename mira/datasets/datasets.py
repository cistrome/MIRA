import os
import tarfile
import logging
from urllib.request import urlretrieve
from tqdm.auto import tqdm
from functools import partial

logger = logging.getLogger(__name__)

class Dataset:

    def __init__(self,*, 
        remote_url, 
        download_directory = 'mira-datasets',
        tar = True,
        is_directory = True):

        self.remote_url = remote_url
        self.download_dir = download_directory
        self.tar = tar
        self.is_directory = is_directory

        if self.is_directory:
            assert self.tar, 'If downloading a directory, it must be tarballed.'

    @property 
    def local_filename(self):
        return os.path.join(self.download_dir, self.remote_filename)

    @property
    def remote_filename(self):
        return os.path.basename(self.remote_url)

    @property
    def uncompressed_name(self):
        if self.tar:
            return self.local_filename[:-7]
        else:
            return self.local_filename

    @property
    def is_on_disk(self):

        if self.tar: 
            assert self.remote_filename[-7:] == '.tar.gz', 'If remote file is a tar directory, must end with .tar.gz'

            if self.is_directory:
                return os.path.isdir(self.uncompressed_name)
            elif not self.is_directory:
                return os.path.isfile(self.uncompressed_name)
            else:
                return False
        else:
            return os.path.isfile(self.uncompressed_name)

    @staticmethod
    def _progress(block_num, block_size, total_size,*,
        progress_bar):

        block_size, total_size = block_size//1000, total_size//1000

        if progress_bar.total == None:
            progress_bar.reset(total = max(total_size, 1))

        progress_bar.update(block_size)


    def download(self):

        if not os.path.exists(self.download_dir):
                os.mkdir(self.download_dir)
        
        try:
            with tqdm(total = None, desc = 'Downloading dataset') as bar:
                
                progress_func = partial(self._progress, progress_bar = bar)

                urlretrieve(
                    self.remote_url, self.local_filename, progress_func
                )

                bar.update(max(bar.total - bar.n, 1))

            if self.tar:
                assert tarfile.is_tarfile(self.local_filename)

                with tarfile.open(self.local_filename) as tar:
                    tar.extractall(self.download_dir)

                os.remove(self.local_filename)

        except (Exception, KeyboardInterrupt) as err:
            logger.error('Encountered error, removing downloaded files.')
            if os.path.exists(self.local_filename):
                os.remove(self.local_filename)
            raise err


    def __call__(self):

        if not self.is_on_disk:
            self.download()
        else:
            logger.info('Dataset already on disk.')

        if self.is_directory:
            logger.info(
                'Dataset contents:\n\t* ' + self.uncompressed_name + '\n\t\t* ' + '\n\t\t* '.join(os.listdir(self.uncompressed_name))
            )
        else:
            logger.info('Dataset contents:\n\t* ' + self.local_filename)


def ShareseqSkin_Ma2020(download_directory = 'mira-datasets'):
    '''
    SHARE-seq skin dataset used in paper and tutorials.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/shareseq_Ma2020.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()

#############
# TUTORIALS #
#############


def StreamGraphTutorial(download_directory = 'mira-datasets'):
    '''
    Streamgraph tutorial data
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq.hair_follicle.joint_representation.lineage_inference.h5ad',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()

def PseudotimeTrajectoryInferenceTutorial(download_directory = 'mira-datasets'):
    '''
    Pseudotime trajectory inference tutorial data
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq.hair_follicle.joint_representation.h5ad',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()


def FrankenCell_RNA(download_directory = 'mira-datasets'):
    '''
    Small synthetic test dataset for topic model tuning.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/Frankencell_RNA.h5ad',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()


def ShareseqTopicModels(download_directory = 'mira-datasets'):
    '''
    Topic models trained on SHARE-seq dataset.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq_topic_models.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()

def ShareseqBaseData(download_directory = 'mira-datasets'):
    '''
    Raw count matrices for SHARE-seq skin dataset.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq_base_data.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()


def ShareseqAnnotatedData(download_directory = 'mira-datasets'):
    '''
    Annotated and modeled count matrices for SHARE-seq skin dataset.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq_annotated_data.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()

def ShareseqRPModels(download_directory = 'mira-datasets'):
    '''
    Example RP models for tutorial
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/shareseq/shareseq_example_rp_models.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()

def MouseBrainDataset(download_directory = 'mira-datasets'):
    '''
    Count matrix and topic models for mouse brain dataset
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/tutorials/e18_10X_brain_dataset.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()


###############
# ANNOTATIONS #
###############

def mm10_chrom_sizes(download_directory = 'mira-datasets'):
    '''
    Chromosome sizes for mm10 genome.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/mm10/mm10.chrom.sizes',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()

def mm10_tss_data(download_directory = 'mira-datasets'):
    '''
    Non-redundant canonical TSS locations for mm10 genome.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/mm10/mm10_tss_data.bed12',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()

def hg38_chrom_sizes(download_directory = 'mira-datasets'):
    '''
    Chromosome sizes for hg38 genome.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/hg38/hg38.chrom.sizes',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()

def hg38_tss_data(download_directory = 'mira-datasets'):
    '''
    Chromosome sizes for hg38 genome.
    '''

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/hg38/hg38_tss_data.bed12',
        tar=False, is_directory=False,
        download_directory=download_directory,
    )()

def test_download(download_directory = 'mira-datasets'):

    Dataset(
        remote_url='http://cistrome.org/~alynch/data/mira-data/test_download.tar.gz',
        tar=True, is_directory=True,
        download_directory=download_directory,
    )()