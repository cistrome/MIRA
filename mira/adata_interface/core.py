import anndata
import inspect
from functools import wraps
import numpy as np
import logging
from tqdm import tqdm
from scipy.sparse import isspmatrix
from scipy import sparse
logger = logging.getLogger(__name__)

def return_output(self, adata, output):
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
        adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(adata, **kwargs):

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

            output = func(**fetch(None, adata, **getter_kwargs), **function_kwargs)

            #print(output, adata, adder_kwargs)
            return add(None, adata, output, **adder_kwargs)

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
        adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(self, adata, **kwargs):

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

            return add(self, adata, output, **adder_kwargs)

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

def return_adata(self, adata, output):
    return adata

def add_obs_col(self, adata, output,*,colname):
    adata.obs[colname] = output

def project_matrix(adata_index, project_features, vals):

    orig_feature_idx = dict(zip(adata_index, np.arange(len(adata_index))))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    matrix = np.full((vals.shape[0], len(adata_index)), np.nan)
    matrix[:, original_to_imputed_map] = vals

    return matrix

def add_layer(self, adata, output, add_layer = 'imputed'):
    
    logger.info('Added layer: ' + add_layer)
    adata_features = adata.var_names.values

    orig_feature_idx = dict(zip(adata_features, np.arange(adata.shape[-1])))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in self.features]
    )

    new_layer = np.full(adata.shape, np.nan)
    new_layer[:, original_to_imputed_map] = output

    adata.layers[add_layer] = new_layer