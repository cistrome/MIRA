
import numpy as np
import logging
logger = logging.getLogger(__name__)

from scipy import sparse
from mira.adata_interface.core import fetch_layer, add_obs_col, \
        add_obsm, project_matrix, add_varm
from torch.utils.data import Dataset
import os
import glob
import torch
from torch.utils.data import DataLoader
from functools import partial
import anndata
from tqdm.auto import tqdm
from collections import defaultdict


def _transpose_list_of_dict(list_of_dicts):
    
    keys = list_of_dicts[0].keys()
    dict_of_lists = defaultdict(list)
    
    for l in list_of_dicts:
        for k in keys:
            dict_of_lists[k].append(l[k])
            
    return dict(dict_of_lists)


class OnDiskDataset(Dataset):

    @classmethod
    def write_to_disk(cls, batch_size = 128,*,
            dirname, features, 
            highly_variable, data_loader):

        os.mkdir(dirname)
        os.mkdir(os.path.join(dirname, 'batches'))

        meta = {
            'features' : features,
            'highly_variable' : highly_variable,
            'batch_size' : batch_size,
        }
        
        torch.save(meta, os.path.join(dirname, 'dataset_meta.pth'))
        
        for i, batch in tqdm(enumerate(data_loader), desc = 'Writing dataset to disk'):

            torch.save(
                batch, 
                os.path.join( dirname, 'batches', '{i}.pth'.format(str(i)) )
            )


    def __init__(self, 
        dirname = './dataset',
        seed = 0):

        assert os.path.isdir(dirname)
        assert os.path.isdir(os.path.join(dirname, 'batches'))
        assert os.path.exists(os.path.join(dirname, 'dataset_meta.pth'))

        self.dirname = dirname

        self.dataset_meta = torch.load(
            os.path.join(dirname, 'dataset_meta.pth')
        )

        self.batch_names = glob.glob(
            os.path.join(dirname, 'batches', '*.pth')
        )

        self.random_state = np.random.RandomState(seed)

        self.num_batches = len(self.batch_names)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return torch.load(
            self.batch_names[idx]
        )

    def get_dataloader(self,
        model,
        training = False,
        shuffle_only = False,
        batch_size = None):

        extra_kwargs = {}
        if model.dataset_loader_workers > 0:
            extra_kwargs.update(dict(
                num_workers = model.dataset_loader_workers,
                prefetch_factor = 5
            ))

        return DataLoader(
            self, 
            **extra_kwargs,
            shuffle = training,
        )


class InMemoryDataset(Dataset):

    @classmethod
    def get_features(cls, model, adata):

        if model.exogenous_key is None:
            predict_mask = np.ones(adata.shape[-1]).astype(bool)
        else:
            predict_mask = adata.var_vector(model.exogenous_key)
            logger.info('Predicting expression from genes from col: ' + model.exogenous_key)
                
        adata = adata[:, predict_mask]

        if model.endogenous_key is None:
            highly_variable = np.ones(adata.shape[-1]).astype(bool)
        else:
            highly_variable = adata.var_vector(model.endogenous_key)
            logger.info('Using highly-variable genes from col: ' + model.endogenous_key)

        features = adata.var_names.values
        highly_variable = highly_variable

        return features, highly_variable


    def __init__(self, adata,*,
        features, highly_variable, covariates_keys, extra_features_keys,
        counts_layer):

        assert len(features) == len(highly_variable)

        self.features = features
        self.highly_variable = highly_variable

        self.exog_features = fetch_layer(self, adata[:, self.features], counts_layer, copy = False)
        self.covariates = fetch_columns(self, adata, covariates_keys)
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
            'extra_features' : self.extra_features[idx] if not self.extra_features is None else []
        }


    @staticmethod
    def collate_batch(batch,*,model):

        def collate(*, endog_features, exog_features, 
                covariates, extra_features):

            endog, exog = sparse.vstack(endog_features), sparse.vstack(exog_features)

            features = {
                'endog_features' : model.preprocess_endog(endog),
                'exog_features' : model.preprocess_exog(exog),
                'read_depth' : model.preprocess_read_depth(exog),
                'covariates' : np.array(covariates).astype(np.float32),
                'extra_features' : np.array(extra_features).astype(np.float32),
            }

            return {
                k : torch.tensor(v, requires_grad = False).to(model.device)
                for k, v in features.items()
            }

        return collate(**_transpose_list_of_dict(batch))


    def get_dataloader(self,
        model,
        training = False,
        shuffle_only = False,
        batch_size = None):

        if batch_size is None:
            batch_size = model.batch_size

        if training:
            extra_kwargs = dict(
                shuffle = True, drop_last = True,
            )
            if model.dataset_loader_workers > 0:
                extra_kwargs.update(dict(
                    num_workers = model.dataset_loader_workers,
                    prefetch_factor = 5
                ))
        elif shuffle_only:
            extra_kwargs = {'shuffle' : True}
        else:
            extra_kwargs = {}

        return DataLoader(
            self, 
            batch_size = batch_size, 
            **extra_kwargs,
            collate_fn= partial(self.collate_batch, model = model)
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


def fetch_columns(self, adata, cols):

    assert(isinstance(cols, list) or cols is None)
    if cols is None:
        return None
    else:
       return np.hstack([
            adata.obs_vector(col).astype(np.float32)[:, np.newaxis] for col in cols
        ])


def fit_adata(self, adata):

    features, highly_variable = InMemoryDataset.get_features(self, adata)

    dataset = InMemoryDataset(
        adata, 
        features=features,
        highly_variable=highly_variable,
        covariates_keys=self.covariates_keys,
        extra_features_keys=self.extra_features_keys,
        counts_layer=self.counts_layer
    )

    return dict(
        features = dataset.features,
        highly_variable = dataset.highly_variable,
        data_loader = dataset.get_dataloader(
            self, training=True
        )
    )


def fit_on_disk_dataset(self, dirname):
    pass


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


def fetch_features(self, adata):

    return {'data_loader' : InMemoryDataset(
                adata,
                features = self.features,
                highly_variable = self.highly_variable,
                counts_layer = self.counts_layer,
                covariates_keys = self.covariates_keys,
                extra_features_keys = self.extra_features_keys
                ).get_dataloader(
                    self, training=False
                )
            }


def fetch_topic_comps(self, adata, key = 'X_topic_compositions'):
    logger.info('Fetching key {} from obsm'.format(key))

    covariates = fetch_columns(self, adata, self.covariates_keys)
    extra_features = fetch_columns(self, adata, self.extra_features_keys)

    return dict(topic_compositions = adata.obsm[key],
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

    #logger.info('Added key to uns: topic_dendogram')
    #adata.uns['topic_dendogram'] = output['topic_dendogram']


def add_umap_features(adata, output, add_key = 'X_umap_features'):
    logger.info('Added key to obsm: ' + add_key)
    adata.obsm[add_key] = output

def add_phylo(adata, output, add_key = 'X_umap_features'):
    features, linkage = output
    add_obsm(adata, features, add_key = add_key)
    logger.info('Added key to uns: topic_dendogram')
    adata.uns['topic_dendogram'] = linkage
