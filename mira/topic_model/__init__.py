from mira.topic_model.accessibility_model import AccessibilityTopicModel
from mira.topic_model.expression_model import ExpressionTopicModel
from mira.topic_model.trainer import TopicModelTuner, print_study, SpeedyTuner
from mira.topic_model.covariate_model import CovariateModelMixin
from mira.topic_model.dirichlet_process import \
        ExpressionDirichletProcessModel, AccessibilityDirichletProcessModel
from mira.topic_model.base import Tracker
from torch.utils.tensorboard import SummaryWriter as TensorboardTracker
import numpy as np

def MakeModel(
    endogenous_key = None,
    exogenous_key = None,
    counts_layer = None,
    covariates_keys = None,
    extra_features_keys = None,
    latent_space = 'dirichlet',
    feature_type = 'expression',
):

    assert(latent_space in ['dp', 'dirichlet'])
    assert(feature_type in ['expression','accessibility'])

    if isinstance(covariates_keys, (list, np.ndarray)) \
            and len(covariates_keys) > 0:
    
        pass
