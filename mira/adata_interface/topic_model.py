
import numpy as np
import logging
logger = logging.getLogger(__name__)

from scipy import sparse
from mira.adata_interface.core import fetch_layer, add_obs_col, \
        add_obsm, project_matrix, add_varm
from torch.utils.data import Dataset, IterableDataset
import os
import glob
import torch
from torch.utils.data import DataLoader
from functools import partial
import anndata
from tqdm.auto import tqdm
from collections import defaultdict
import pickle


def _transpose_list_of_dict(list_of_dicts):
    
    keys = list_of_dicts[0].keys()
    dict_of_lists = defaultdict(list)
    
    for l in list_of_dicts:
        for k in keys:
            dict_of_lists[k].append(l[k])
            
    return dict(dict_of_lists)


class TopicModelDataset:

    @staticmethod
    def collate_batch(batch,*,model):

        def collate(*, endog_features, exog_features, 
                covariates, categorical_covariates, continuous_covariates, 
                extra_features):

            endog, exog = sparse.vstack(endog_features), sparse.vstack(exog_features)

            # covars, categorical, continuous
            covariates = np.hstack([
                np.array(covariates), 
                model.preprocess_categorical_covariates(
                    np.vstack(categorical_covariates)
                ),
                model.preprocess_continuous_covariates(
                    np.vstack(continuous_covariates)
                ),
            ]).astype(np.float32)

            features = {
                'endog_features' : model.preprocess_endog(endog),
                'exog_features' : model.preprocess_exog(exog),
                'read_depth' : model.preprocess_read_depth(exog),
                'covariates' : covariates,
                'extra_features' : np.array(extra_features).astype(np.float32),
            }

            return {
                k : torch.tensor(v, requires_grad = False)
                for k, v in features.items()
            }

        return collate(**_transpose_list_of_dict(batch))

    def check_meta(self, model):
        
        for check, value in self.get_fit_meta().items():
            assert getattr(model, check) == value, \
                    'This on-disk dataset may not have been compiled with ' \
                    'this model\'s parameters. Mismatch in {}: model has {}, dataset has {}'.format(
                        check, getattr(model, check), value
                    )


    def get_fit_meta(self):
        return self._fit_meta

    def get_dataloader(self,
        model,
        training = False,
        batch_size = None):

        if batch_size is None:
            batch_size = model.batch_size

        if training:
            extra_kwargs = dict(
                drop_last = True,
            )
            if model.dataset_loader_workers > 0:
                extra_kwargs.update(dict(
                    num_workers = model.dataset_loader_workers,
                    prefetch_factor = 5
                ))

            if not isinstance(self, IterableDataset):
                extra_kwargs['shuffle'] = True
                
        else:
            extra_kwargs = {}

        return DataLoader(
            self, 
            batch_size = batch_size, 
            **extra_kwargs,
            collate_fn= partial(self.collate_batch, model = model)
        )


class OnDiskDataset(TopicModelDataset, IterableDataset):

    @classmethod
    def write_to_disk(cls, batch_size = 128,*,
            dirname, features, 
            highly_variable, dataset):

        chunk_size = batch_size * 8

        chunked_data = DataLoader(
            dataset, 
            shuffle = True,
            batch_size = chunk_size,
            collate_fn= lambda x : x,
        )

        os.mkdir(dirname)
        os.mkdir(os.path.join(dirname, 'chunks'))

        meta = {
            'features' : features,
            'highly_variable' : highly_variable,
            'length' : len(dataset),
            'fit_meta' : dataset.get_fit_meta(),
        }
        
        with open(os.path.join(dirname, 'dataset_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        
        for i, chunk in enumerate(tqdm(chunked_data, desc = 'Writing dataset to disk')):
            
            with open(os.path.join( dirname, 'chunks', '{}.pkl'.format(str(i)) ), 'wb') as f:
                pickle.dump(chunk, f)


    def __init__(self, 
        dirname = './dataset',
        seed = 0):

        assert os.path.isdir(dirname)
        assert os.path.isdir(os.path.join(dirname, 'chunks'))
        assert os.path.exists(os.path.join(dirname, 'dataset_meta.pkl'))

        self.dirname = dirname

        with open(os.path.join(dirname, 'dataset_meta.pkl'), 'rb') as f:
            self.dataset_meta = pickle.load(f)

        self.features = self.dataset_meta['features']
        self.highly_variable = self.dataset_meta['highly_variable']
        self._fit_meta = self.dataset_meta['fit_meta']

        self.chunk_names = glob.glob(
            os.path.join(dirname, 'chunks', '*.pkl')
        )

        self.random_state = np.random.RandomState(seed)
        self.num_chunks = len(self.chunk_names)


    def __len__(self):
        return self.dataset_meta['length']


    def __iter__(self):
        
        chunk_load_order = self.random_state.permutation(
            self.num_chunks
        )

        for chunk_num in chunk_load_order:

            with open( os.path.join(self.dirname, 'chunks', str(chunk_num) + '.pkl'), 'rb' ) as f:
                chunk = pickle.load(f)

                chunk_yield_order = self.random_state.permutation(len(chunk))

                for item_num in chunk_yield_order:
                    yield chunk[item_num]


class InMemoryDataset(TopicModelDataset, Dataset):

    @classmethod
    def get_features(cls, adata, endogenous_key, exogenous_key):

        if exogenous_key is None:
            predict_mask = np.ones(adata.shape[-1]).astype(bool)
        else:
            predict_mask = adata.var_vector(exogenous_key)
            logger.info('Predicting expression from genes from col: ' + exogenous_key)
                
        adata = adata[:, predict_mask]

        if endogenous_key is None:
            highly_variable = np.ones(adata.shape[-1]).astype(bool)
        else:
            highly_variable = adata.var_vector(endogenous_key)
            logger.info('Using highly-variable genes from col: ' + endogenous_key)

        features = adata.var_names.values
        highly_variable = highly_variable

        return features, highly_variable


    def __init__(self, adata,
        exogenous_key = None, endogenous_key = None,
        features = None, highly_variable = None,*,
        categorical_covariates, continuous_covariates, covariates_keys, 
        extra_features_keys, counts_layer,
        ):

        self._fit_meta = dict(
            categorical_covariates = categorical_covariates,
            continuous_covariates = continuous_covariates,
            extra_features_keys = extra_features_keys,
            counts_layer = counts_layer,
            exogenous_key = exogenous_key,
            endogenous_key = endogenous_key,
        )

        if features is None or highly_variable is None:
            features, highly_variable = self.get_features(adata, endogenous_key, exogenous_key)

        assert len(features) == len(highly_variable)

        self.features = features
        self.highly_variable = highly_variable

        self.exog_features = fetch_layer(self, adata[:, self.features], counts_layer, copy = False)

        self.covariates = fetch_columns(self, adata, covariates_keys)
        self.continuous_covariates = fetch_columns(self, adata, continuous_covariates)
        self.categorical_covariates = fetch_columns(self, adata, categorical_covariates, dtype = str)

        self.extra_features = fetch_columns(self, adata, extra_features_keys)

        assert isinstance(self.exog_features, sparse.spmatrix)
        assert isinstance(self.exog_features, sparse.spmatrix)


    def __len__(self):
        return self.exog_features.shape[0]

    def __getitem__(self, idx):
        return {
            'endog_features' : self.exog_features[idx, self.highly_variable],
            'exog_features' : self.exog_features[idx],
            'covariates' : self.covariates[idx] if not self.covariates is None else [],
            'categorical_covariates' : self.categorical_covariates[idx] if not self.categorical_covariates is None else [],
            'continuous_covariates' : self.continuous_covariates[idx] if not self.continuous_covariates is None else [],
            'extra_features' : self.extra_features[idx] if not self.extra_features is None else []
        }


def fit_adata(self, adata):

    dataset = InMemoryDataset(
        adata, 
        endogenous_key=self.endogenous_key,
        exogenous_key=self.exogenous_key,
        covariates_keys=self.covariates_keys,
        continuous_covariates=self.continuous_covariates,
        categorical_covariates=self.categorical_covariates,
        extra_features_keys=self.extra_features_keys,
        counts_layer=self.counts_layer
    )

    return dict(
        features = dataset.features,
        highly_variable = dataset.highly_variable,
        dataset = dataset,
    )


def fit_on_disk_dataset(self, dirname):
    
    dataset = OnDiskDataset(
        dirname = dirname, 
        seed = self.seed
    )

    dataset.check_meta(self)
    
    return dict(
        features = dataset.features,
        highly_variable = dataset.highly_variable,
        dataset = dataset,
    )


def fit(self, adata_or_dirname):

    if isinstance(adata_or_dirname, str):
        return fit_on_disk_dataset(self, adata_or_dirname)
    elif isinstance(adata_or_dirname, anndata.AnnData):
        return fit_adata(self, adata_or_dirname)
    else:
        raise ValueError(
            'Passed data of type {}, only str (dirname for on disk dataset) or AnnData are supported'\
                .format(type(adata_or_dirname))
        )


def fetch_features(self, adata_or_dirname):

    if isinstance(adata_or_dirname, str):

        dataset = OnDiskDataset(dirname=adata_or_dirname)

        assert all(self.features == dataset.features)
        assert all(self.highly_variable == dataset.highly_variable)
        dataset.check_meta(self)

        return {'dataset' : dataset}

    elif isinstance(adata_or_dirname, anndata.AnnData):

        return {'dataset' : InMemoryDataset(
                adata_or_dirname,
                features = self.features,
                highly_variable = self.highly_variable,
                counts_layer = self.counts_layer,
                covariates_keys = self.covariates_keys,
                continuous_covariates=self.continuous_covariates,
                categorical_covariates=self.categorical_covariates,
                extra_features_keys = self.extra_features_keys
                )
            }

    else:
        raise ValueError(
            'Passed data of type {}, only str (dirname for on disk dataset) or AnnData are supported'\
                .format(type(adata_or_dirname))
        )

    
def add_test_column(adata, output):
    test_column, test_cell_mask = output

    logger.info('Added col: ' + str(test_column))
    add_obs_col(adata, test_cell_mask, colname = test_column)



def fetch_split_train_test(self, adata):
    assert(adata.obs_vector(self.test_column).dtype == bool), 'Test set column must be boolean vector'
    assert(adata.obs_vector(self.test_column).any()), 'No cells are in the test set.'

    return dict(
        all_data = adata,
        train_data = adata[~adata.obs_vector(self.test_column)],
        test_data = adata[adata.obs_vector(self.test_column)]
    )


def fetch_columns(self, adata, cols, dtype = np.float32):

    assert(isinstance(cols, list) or cols is None)
    if cols is None:
        return np.array([]).reshape(0, len(adata)).T
    else:
        return np.hstack([
            adata.obs_vector(col).astype(dtype)[:, np.newaxis] 
            for col in cols
        ])


def fetch_topic_comps(self, adata, key = 'X_topic_compositions'):
    logger.info('Fetching key {} from obsm'.format(key))

    covariates = fetch_columns(self, adata, self.covariates_keys)
    
    continuous_covariates = self.preprocess_continuous_covariates(
        fetch_columns(self, adata, self.continuous_covariates)
    )

    categorical_covariates = self.preprocess_categorical_covariates(
        fetch_columns(self, adata, self.categorical_covariates, dtype = str)
    )

    covariates = np.hstack([covariates, categorical_covariates, continuous_covariates])
    extra_features = fetch_columns(self, adata, self.extra_features_keys)

    return dict(topic_compositions = adata.obsm[key].astype(np.float32),
                covariates = covariates, extra_features = extra_features)


def fetch_topic_comps_and_linkage_matrix(self, adata, key = 'X_topic_compositions', 
    dendogram_key = 'topic_dendogram'):
    
    logger.info('Fetching key {} from obsm'.format(key))
    logger.info('Fetching key {} from uns'.format(dendogram_key))

    return dict(
        topic_compositions = adata.obsm[key],
        linkage_matrix = adata.uns[dendogram_key],
    )


def add_topic_comps(adata, output, add_key = 'X_topic_compositions', 
        add_cols = True, col_prefix = 'topic_'):

    cell_topic_dists = output['cell_topic_dists']
    
    add_obsm(adata, cell_topic_dists, add_key = add_key)
    add_obsm(adata, output['umap_features'], add_key = 'X_umap_features')

    if add_cols:
        K = cell_topic_dists.shape[-1]
        cols = [col_prefix + str(i) for i in range(K)]
        logger.info('Added cols: ' + ', '.join(cols))
        adata.obs[cols] = cell_topic_dists

    add_varm(adata, project_matrix(adata.var_names, 
            output['feature_names'], output['topic_feature_dists']).T,
            add_key = 'topic_feature_compositions')
    
    add_varm(adata, project_matrix(adata.var_names, 
            output['feature_names'], output['topic_feature_activations']).T,
            add_key='topic_feature_activations')

    logger.info('Added key to uns: topic_dendogram')
    adata.uns['topic_dendogram'] = output['topic_dendogram']


def add_umap_features(adata, output, add_key = 'X_umap_features'):
    logger.info('Added key to obsm: ' + add_key)
    adata.obsm[add_key] = output

def add_phylo(adata, output, add_key = 'X_umap_features'):
    features, linkage = output
    add_obsm(adata, features, add_key = add_key)
    logger.info('Added key to uns: topic_dendogram')
    adata.uns['topic_dendogram'] = linkage
