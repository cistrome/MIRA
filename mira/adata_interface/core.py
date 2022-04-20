
import inspect
from functools import wraps
import numpy as np
import logging
from scipy.sparse import isspmatrix
from scipy import sparse
from anndata import AnnData
logger = logging.getLogger(__name__)

def return_output(adata, output):
    return output

def wraps_functional(
    fetch,
    add = return_output,
    fill_kwargs = []
):

    def run(func):

        getter_signature = inspect.signature(fetch).parameters.copy()
        adder_signature = inspect.signature(add).parameters.copy()
        func_signature = inspect.signature(func).parameters.copy()

        for del_kwarg in fill_kwargs:
            func_signature.pop(del_kwarg)

        getter_signature.pop('self')
        #adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(adata, *args, **kwargs):
            
            if not isinstance(adata, AnnData):
                raise TypeError('First argument of this function must be an AnnData object')

            if not len(args) == 0:
                raise TypeError('Positional arguments are not allowed for this function')

            getter_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(fetch).parameters.copy().keys() if arg in kwargs
            }

            adder_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(add).parameters.copy().keys() if arg in kwargs
            }

            function_kwargs = {
                arg: kwargs[arg]
                for arg in func_signature.keys() if arg in kwargs
           }

            for kwarg in kwargs.keys():
                if not any(
                    [kwarg in subfunction_kwargs.keys() for subfunction_kwargs in [getter_kwargs, adder_kwargs, function_kwargs]]
                ):
                    raise TypeError('{} is not a valid keyword arg for this function.'.format(kwarg))

            #print(function_kwargs)
            #print(fetch(None, adata, **getter_kwargs))

            output = func(**fetch(None, adata, **getter_kwargs), **function_kwargs)
            #print(output, adata, adder_kwargs)
            return add(adata, output, **adder_kwargs)

        _run.__name__ = func.__name__

        return _run
    
    return run

def wraps_modelfunc(
    fetch = lambda self, adata : {},
    add = return_output,
    fill_kwargs = []
):

    def run(func):

        getter_signature = inspect.signature(fetch).parameters.copy()
        adder_signature = inspect.signature(add).parameters.copy()
        func_signature = inspect.signature(func).parameters.copy()

        for del_kwarg in fill_kwargs:
            func_signature.pop(del_kwarg)
    
        func_signature.pop('self')
        #adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(self, adata, *args, **kwargs):

            if not isinstance(adata, AnnData):
                raise TypeError('First argument of this function must be an AnnData object')

            if not len(args) == 0:
                raise TypeError('Positional arguments are not allowed for this function')

            getter_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(fetch).parameters.copy().keys() if arg in kwargs
            }

            adder_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(add).parameters.copy().keys() if arg in kwargs
            }

            function_kwargs = {
                arg: kwargs[arg]
                for arg in func_signature.keys() if arg in kwargs
            }

            for kwarg in kwargs.keys():
                if not any(
                    [kwarg in subfunction_kwargs.keys() for subfunction_kwargs in [getter_kwargs, adder_kwargs, function_kwargs]]
                ):
                    raise TypeError('{} is not a valid keyword arg for this function.'.format(kwarg))

            output = func(self, **fetch(self, adata, **getter_kwargs), **function_kwargs)

            return add(adata, output, **adder_kwargs)

        _run.__name__ = func.__name__

        return _run
    
    return run

## GENERAL ACCESSORS ##
def fetch_layer(self, adata, layer):
    if layer is None:
        return adata.X.copy()
    else:
        return adata.layers[layer].copy()


def fetch_adata_shape(self, adata):
    return dict(
        shape = adata.shape
    )


def return_adata(adata, output):
    return adata


def add_obs_col(adata, output,*,colname):
    logger.info('Added cols to obs: ' + str(colname))
    adata.obs[colname] = output


def project_matrix(adata_index, project_features, vals):

    assert(isinstance(vals, np.ndarray))

    orig_feature_idx = dict(zip(adata_index, np.arange(len(adata_index))))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    matrix = np.full((vals.shape[0], len(adata_index)), np.nan)
    matrix[:, original_to_imputed_map] = vals

    return matrix


def project_sparse_matrix(adata_index, project_features, vals):

    assert(isinstance(vals, np.ndarray))

    orig_feature_idx = dict(zip(adata_index, np.arange(len(adata_index))))

    bin_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    index_converted = sparse.coo_matrix(vals)

    vals = sparse.coo_matrix(
        (index_converted.data, (index_converted.row, bin_map[index_converted.col])), 
        shape = (vals.shape[0], len(adata_index))
    ).tocsr()

    return vals


'''def project_row(adata_index, project_features, vals, width):

    orig_feature_idx = dict(zip(adata_index, np.arange(width)))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    new_row = np.full(width, np.nan)
    new_row[original_to_imputed_map] = vals
    return new_row
'''

def get_dense_columns(self, adata, layer):

    layer = fetch_layer(self, adata, layer)

    assert(isspmatrix(layer))
    layer = layer.tocsr()

    column_mask = np.isin(
        np.arange(layer.shape[-1]), 
        np.unique(layer.indices),
        assume_unique = True,
    )

    return column_mask


def add_layer(adata, output, add_layer = 'imputed', sparse = False):
    features, vals = output

    logger.info('Added layer: ' + add_layer)

    if not sparse:
        new_layer = project_matrix(adata.var_names, features, vals)
    else:
        new_layer = project_sparse_matrix(adata.var_names, features, vals)

    adata.layers[add_layer] = new_layer


def add_obsm(adata, output,*,add_key):

    logger.info('Added key to obsm: ' + str(add_key))
    adata.obsm[add_key] = output

def add_varm(adata, output,*,add_key):

    logger.info('Added key to varm: ' + str(add_key))
    adata.varm[add_key] = output