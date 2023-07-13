
from mira.adata_interface.topic_model import fit as fit_adata
from mira.topic_model.base import ModelParamError
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from mira.topic_model.hyperparameter_optim.trainer import DisableLogger, baselogger, \
        corelogger, interfacelogger
from numpy.random import randint

def gradient_tune(model, data, max_attempts = 5, max_topics = None):
    '''
    Tune number of topcis using a gradient-based estimator based on the Dirichlet Process model. 
    This tuner is very fast, though less comprehensive than the BayesianTuner. We recommend using this 
    tuner for large datasets (>40K cells).

    Parameters
    ----------
    model : mira.topics.TopicModel
        Topic model to tune. The provided model should have columns specified
        to retrieve endogenous and exogenous features, and should have the learning
        rate configued by ``get_learning_rate_bounds``.
    data : anndata.AnnData
        Anndata of expression or accessibility data.
    max_topics : int > 0 or None
        If none, MIRA automatically chooses an upper limit on the number of topics to model
        based on a generous hueristic calculated from the number of cells in the provided dataset.
        If a value is provided, that upper limit is used instead.

    Returns
    -------
    num_topics : int
        Estimated number of topics in dataset
    max_topic_contributions : np.ndarray[float] of shape (n_topics,)
        For each topic attempted to learn from the data, its maximum contribution to any cell.
    
    '''

    assert max_topics is None or (isinstance(max_topics, (int, float)) and max_topics > 1), \
        'If provided, `max_topics` must be a positive integer or float.'
    

    with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):

        _dp_model = model._get_dp_model()

        train_meta = fit_adata(_dp_model, data)

        _dp_model.set_params(
            num_topics = _dp_model._recommend_num_topics(len(train_meta['dataset'])) \
                    if max_topics is None else int(max_topics),
            max_learning_rate = 0.1,
        )

        del train_meta

        for _ in range(max_attempts):
            try:
                _dp_model.fit(data)
                break
            except ModelParamError:

                logger.warn(
                    'Model experienced gradient overflow during training, which happens sometimes with the gradient-based tuner. Changing the stochastic seed, then trying again...'
                )

                _dp_model.seed = int(randint(10000))

                # increase regularization from defaults
                _dp_model.encoder_dropout = 0.05
                _dp_model.decoder_dropout = 0.10
                _dp_model.embedding_dropout = 0.1

        else:
            raise ValueError(
                'Could not train the Gradient-based tuner.\n'
                '\u2022 This can happen if the maximum learning rate was initially set wayyyyy too high. Please ensure you have run the learning rate range test and set reasonable learning rate boundaries.\n'
                '\u2022 This could also happen because there are outliers in the dataset in terms of the number of reads in a cell. Make sure to remove outlier cells from the dataset, especially those with too many counts.\n'
                '\u2022 For accessibility (ATAC-seq) models, this can occur when modeling too many features (>150K). Removing extremenly rarely-accessible peaks to reduce the feature space will help.\n'
                'If none of the above work, the standard Bayesian Tuning approach is not affected by numberical stability issues like the gradient-based estimator, so try that next.'
            )

        topic_max_contributions = _dp_model._predict_topic_comps_direct_return(data).max(0)

        return sorted(topic_max_contributions)[::-1]

