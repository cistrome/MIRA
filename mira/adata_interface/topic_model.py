
import numpy as np
import logging
from scipy import sparse
from mira.adata_interface.core import fetch_layer, add_obs_col, \
        add_obsm, project_matrix, add_varm
from torch.utils.data import Dataset

def collate_batch(batch,*,
    preprocess_endog, 
    preprocess_exog, 
    preprocess_read_depth):

    endog, exog = list(zip(*batch))
    endog, exog = sparse.vstack(endog), sparse.vstack(exog)

    return {
        'endog_features' : preprocess_endog(endog),
        'exog_features' : preprocess_exog(exog),
        'read_depth' : preprocess_read_depth(exog)
    }


class InMemoryDataset(Dataset):

    def __init__(self,*, features, highly_variable, 
        counts_layer, adata):

        self.features = features
        self.highly_variable = highly_variable

        adata = adata[:, self.features]

        self.exog_features = fetch_layer(self, adata, counts_layer)
        self.endog_features = fetch_layer(self, adata[:, highly_variable], counts_layer)

        assert isinstance(self.exog_features, sparse.spmatrix)
        assert isinstance(self.exog_features, sparse.spmatrix)

    def __len__(self):
        return self.exog_features.shape[0]

    def __getitem__(self, idx):
        return self.endog_features[idx], self.exog_features[idx]


logger = logging.getLogger(__name__)

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


def fit_adata(self, adata):

    if self.exogenous_key is None:
        predict_mask = np.ones(adata.shape[-1]).astype(bool)
    else:
        predict_mask = adata.var_vector(self.exogenous_key)
        logger.info('Predicting expression from genes from col: ' + self.exogenous_key)
            
    adata = adata[:, predict_mask]

    if self.endogenous_key is None:
        highly_variable = np.ones(adata.shape[-1]).astype(bool)
    else:
        highly_variable = adata.var_vector(self.endogenous_key)
        logger.info('Using highly-variable genes from col: ' + self.endogenous_key)

    features = adata.var_names.values

    return dict(
        features = features,
        highly_variable = highly_variable,
        dataset = InMemoryDataset(
            features = features,
            highly_variable = highly_variable,
            counts_layer = self.counts_layer,
            adata = adata
        )
    )

def fetch_features(self, adata):

    return {'dataset' : InMemoryDataset(
                features = self.features,
                highly_variable = self.highly_variable,
                counts_layer = self.counts_layer,
                adata = adata)
            }


def fetch_topic_comps(self, adata, key = 'X_topic_compositions'):
    logger.info('Fetching key {} from obsm'.format(key))
    return dict(topic_compositions = adata.obsm[key])


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