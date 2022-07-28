from mira.topic_model.accessibility_model import AccessibilityModel
from mira.topic_model.expression_model import ExpressionModel
from mira.topic_model.covariate_model import CovariateModel
from mira.topic_model.dirichlet_process import \
        ExpressionDirichletProcessModel, AccessibilityDirichletProcessModel
from mira.topic_model.dirichlet_model import \
        ExpressionDirichletModel, AccessibilityDirichletModel
from mira.topic_model.vampmodel import ExpressionVampModel
from mira.topic_model.base import BaseModel, logger
import numpy as np

def TopicModel(
    n_samples, n_features,*,
    feature_type,
    endogenous_key = None,
    exogenous_key = None,
    counts_layer = None,
    categorical_covariates = None,
    continuous_covariates = None,
    covariates_keys = None,
    extra_features_keys = None,
    latent_space = 'dirichlet',
    automatically_set_params = True,
    **kwargs,
):

    assert(latent_space in ['dp', 'dirichlet', 'vamp'])
    assert(feature_type in ['expression','accessibility'])

    if latent_space == 'dp':
        if n_samples < 40000:
            logger.warn(
                'The dirichlet process model is intended for atlas-level experiments.\n'
                'For smaller datasets, please use the "dirichlet" latent space, and use the tuner'
                'to find the optimal number of topics.'
            )
            
        latent_space = 'dirichlet-process'

    basename = 'model'
    if not all([c is None for c in [categorical_covariates, continuous_covariates, covariates_keys]]):
        basename = 'covariate-model'
        baseclass = CovariateModel
    else:
        baseclass = BaseModel

    if feature_type == 'expression':
        feature_model = ExpressionModel
    elif feature_type == 'accessibility':
        feature_model = AccessibilityModel

    generative_map = {
        ('expression','dirichlet') : ExpressionDirichletModel,
        ('expression','dirichlet-process') : ExpressionDirichletProcessModel,
        ('accessibility', 'dirichlet') : AccessibilityDirichletModel,
        ('accessibility', 'dirichlet-process') : AccessibilityDirichletProcessModel,
        ('expression', 'vamp')  : ExpressionVampModel
    }

    generative_model = generative_map[(feature_type, latent_space)]

    _class = type(
        '_'.join([latent_space, feature_type, basename]),
        (generative_model, feature_model, baseclass),
        {}
    )

    def none_or_1d(x):
        if x is None:
            return None
        else:
            return list(np.atleast_1d(x))


    instance = _class(
        endogenous_key = endogenous_key,
        exogenous_key = exogenous_key,
        counts_layer = counts_layer,
        categorical_covariates = none_or_1d(categorical_covariates),
        continuous_covariates = none_or_1d(continuous_covariates),
        covariates_keys = none_or_1d(covariates_keys),
        extra_features_keys = none_or_1d(extra_features_keys),
    )

    if automatically_set_params:
        instance.set_params(
            **instance.recommend_parameters(n_samples, n_features)
        )

    instance.set_params(**kwargs)

    return instance