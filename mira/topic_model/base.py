from functools import partial
from scipy import interpolate
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW as Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm.auto import tqdm, trange
import numpy as np
import logging
import time
from sklearn.base import BaseEstimator
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
import gc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import mira.topic_model.ilr_tools as ilr
import time
from torch.distributions import kl_divergence
from collections import defaultdict
from mira.topic_model.CODAL.mine import ConcatLayer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
from shutil import rmtree


logger = logging.getLogger(__name__)

class TopicModel:
    pass

class Writer(defaultdict):
    
    def __init__(self):
        super().__init__(list)
        
    def add_scalar(self, key, value, 
                   step_count = 0):
        self[key].append(value)


class EarlyStopping:

    def __init__(self, 
                 tolerance = 1e-4,
                 patience=3,
                 convergence_check = True):

        self.tolerance = tolerance
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.convergence_check = convergence_check

    def __call__(self, current_loss):
        
        if current_loss is None:
            pass
        else:
            if ((current_loss - self.best_loss) < -self.tolerance) or \
                (self.convergence_check and ((current_loss - self.best_loss) > 10*self.tolerance)):
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    return True
                self.wait += 1

            if current_loss < self.best_loss:
                self.best_loss = current_loss

        return False


class TraceMeanFieldLatentKL(TraceMeanField_ELBO):

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        
        try:
            guide_site = guide_trace.nodes['rna/theta']
            model_site = model_trace.nodes['rna/theta']
        except KeyError:
            guide_site = guide_trace.nodes['atac/theta']
            model_site = model_trace.nodes['atac/theta']

        return kl_divergence(guide_site["fn"], model_site["fn"]).sum().item(), None


def encoder_layer(input_dim, output_dim, nonlin = True, dropout = 0.2):
    layers = [nn.Linear(input_dim, output_dim, bias = False), nn.BatchNorm1d(output_dim)]
    if nonlin:
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_fc_stack(layer_dims = [256, 128, 128, 128], dropout = 0.2, skip_nonlin = True):
    return nn.Sequential(*[
        encoder_layer(input_dim, output_dim, nonlin= not ((i >= (len(layer_dims) - 2)) and skip_nonlin), dropout = dropout)
        for i, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:]))
    ])


def _mask_drop(x, dropout_rate = 1/20, training = False):

    if training:
        return torch.multiply(
                    torch.empty_like(x, requires_grad = False).bernoulli_(1-dropout_rate),
                    x
            )
    else:
        return x


class Decoder(nn.Module):
    
    def __init__(self, covariates_hidden = 32,
        covariates_dropout = 0.05, mask_dropout = 0.05,*,
        num_exog_features, num_topics, num_covariates, topics_dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_exog_features, bias = False)
        self.bn = nn.BatchNorm1d(num_exog_features)

        self.drop1 = nn.Dropout(covariates_dropout)
        self.drop2 = nn.Dropout(topics_dropout)
        self.mask_drop = partial(_mask_drop, dropout_rate = mask_dropout)

        self.num_topics = num_topics
        self.num_covariates = num_covariates

        if num_covariates > 0:

            self.batch_effect_model = nn.Sequential(
                ConcatLayer(1),
                encoder_layer(num_topics + num_covariates, 
                    covariates_hidden, dropout=covariates_dropout, nonlin=True),
                nn.Linear(covariates_hidden, num_exog_features),
                nn.BatchNorm1d(num_exog_features, affine = False),
            )

            if num_covariates > 0:
                self.batch_effect_gamma = nn.Parameter(
                    torch.zeros(num_exog_features)
                )

    @property
    def is_correcting(self):
        return self.num_covariates > 0

    def forward(self, theta, covariates, nullify_covariates = False):
        
        X1 = self.drop1(theta)
        X2 = self.drop2(theta)
        
        if self.is_correcting:
            
            self.covariate_signal = self.get_batch_effect(X1, covariates, 
                nullify_covariates = nullify_covariates)

            self.biological_signal = self.get_biological_effect(X1)

        return F.softmax(
                self.get_biological_effect(X2) + \
                self.mask_drop(
                    self.get_batch_effect(X2, covariates, nullify_covariates = nullify_covariates), 
                    training = self.training
                ), 
                dim=1
            )


    def get_biological_effect(self, theta):
        return self.bn(self.beta(theta))


    def get_batch_effect(self, theta, covariates, nullify_covariates = False):
        
        if not self.is_correcting or nullify_covariates: 
            batch_effect = theta.new_zeros(1)
            batch_effect.requires_grad = False
        else:
            batch_effect = self.batch_effect_gamma * self.batch_effect_model(
                    (theta, covariates)
                )

        return batch_effect


    def get_softmax_denom(self, theta, covariates, include_batcheffects = True):
        return (
            self.get_biological_effect(theta) + \
            self.get_batch_effect(theta, covariates, nullify_covariates = not include_batcheffects)
        ).exp().sum(-1)


class ModelParamError(ValueError):
    pass


class OneCycleLR_Wrapper(torch.optim.lr_scheduler.OneCycleLR):

    def __init__(self, optimizer, **kwargs):
        max_lr = kwargs.pop('max_lr')
        super().__init__(optimizer, max_lr, **kwargs)


class DataCache:

    def __init__(self, model, adata, prefix, 
        cache_dir = './', seed = 0, train_size = 0.8):

        self.cache_dir = cache_dir
        self.model = model
        self.prefix = prefix
        self.adata = adata
        self.seed = seed
        self.train_size = train_size

    def get_cache_path(self):
        return os.path.join(
            self.cache_dir, '.' + str(self.prefix) + '-' + self.model.get_datacache_hash()
        )

    @property
    def train_cache(self):
        return os.path.join(self.get_cache_path(), 'train/')

    @property
    def test_cache(self):
        return os.path.join(self.get_cache_path(), 'test/')

    def rm_cache(self):
        rmtree(self.get_cache_path())

    def cache_data(self, overwrite = False):

        if os.path.isdir(self.get_cache_path()):
            if overwrite:
                self.rm_cache()
                os.mkdir(self.get_cache_path())
            else:
                logger.warn('Cache already exists for this dataset, skipping writing step. If you wish to overwrite, \n'
                            'pass "overwrite = True" to this method.')
                return self.train_cache, self.test_cache
        else:
            os.mkdir(self.get_cache_path())

        train, test = self.model.train_test_split(
            self.adata, 
            seed = self.seed, 
            train_size = self.train_size
        )

        self.model.write_ondisk_dataset(train, dirname = self.train_cache)
        self.model.write_ondisk_dataset(test, dirname = self.test_cache)

        return self.train_cache, self.test_cache


    def __enter__(self):
        return self.cache_data()
        
    def __exit__(self ,type, value, traceback):
        self.rm_cache()
 

class BaseModel(torch.nn.Module, BaseEstimator):

    _decoder_model = Decoder
    _min_dropout = 0.05

    @classmethod
    def load(cls, filename):
        '''
        Load a pre-trained topic model from disk.
        
        Parameters
        ----------
        filename : str
            File name of saved topic model

        Examples
        --------

        .. code-block:: python

            >>> rna_model = mira.topics.ExpressionTopicModel.load('rna_model.pth')
            >>> atac_model = mira.topics.AccessibilityTopicModel.load('atac_model.pth')

        '''

        data = torch.load(filename,map_location=torch.device('cpu'))

        model = cls(**data['params'])
        model._set_weights(data['fit_params'], data['weights'])

        return model


    def __init__(self,
            endogenous_key = None,
            exogenous_key = None,
            counts_layer = None,
            covariates_keys = None,
            categorical_covariates = None,
            continuous_covariates = None,
            extra_features_keys = None,
            num_topics = 16,
            hidden = 128,
            num_layers = 3,
            num_epochs = 24,
            decoder_dropout = 0.055,
            cost_beta = 1.,
            encoder_dropout = 0.01,
            use_cuda = True,
            seed = 0,
            min_learning_rate = 1e-3,
            max_learning_rate = 1e-1,
            beta = 0.92,
            batch_size = 64,
            initial_pseudocounts = 50,
            nb_parameterize_logspace = True,
            embedding_size = None,
            kl_strategy = 'cyclic',
            dataset_loader_workers = 0,
            weight_decay = 0.001,
            min_momentum = 0.85,
            max_momentum = 0.95,
            embedding_dropout = 0.05,
            reconstruction_weight = 1.,
            atac_encoder = 'skipDAN',
            ):
        '''
        
        Attributes
        ----------
        features : np.ndarray[str]
            Array of exogenous feature names, all features used in learning topics
        highly_variable : np.ndarray[boolean]
            Boolean array marking which features were 
            "highly_variable"/endogenous, used to train encoder
        encoder : torch.nn.Sequential
            Encoder neural network
        decoder : torch.nn.Sequential
            Decoder neural network
        num_exog_features : int
            Number of exogenous features to predict using decoder network
        num_endog_features : int
            Number of endogenous feature used for encoder network
        device : torch.device
            Device on which model is allocated
        enrichments : dict
            Results from enrichment analysis of topics. For expression topic model,
            this gives geneset enrichments from Enrichr. For accessibility topic
            model, this gives motif enrichments.
        topic_cols : list
            The names of the columns for the topics added by the
            `predict` method to an anndata object. Useful for quickly accessing
            topic columns for plotting.
                
        '''
        super().__init__()

        self.endogenous_key = endogenous_key
        self.exogenous_key = exogenous_key
        self.counts_layer = counts_layer
        self.categorical_covariates = categorical_covariates
        self.continuous_covariates = continuous_covariates
        self.covariates_keys = covariates_keys
        self.extra_features_keys = extra_features_keys
        self.num_topics = num_topics
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.use_cuda = use_cuda
        self.seed = seed
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.initial_pseudocounts = initial_pseudocounts
        self.nb_parameterize_logspace = nb_parameterize_logspace
        self.embedding_size = embedding_size
        self.kl_strategy = kl_strategy
        self.reconstruction_weight = reconstruction_weight
        self.dataset_loader_workers = dataset_loader_workers
        self.weight_decay = weight_decay
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.embedding_dropout = embedding_dropout
        self.cost_beta = cost_beta
        self.atac_encoder = atac_encoder
        

    def _spawn_submodel(self, generative_model):

        _, feature_model, baseclass, docclass \
                = self.__class__.__bases__

        names = self.__class__.__name__.split('_')

        _class = type(
            '_'.join(['dirichletprocess', *names[1:]]),
            (generative_model, feature_model, baseclass, docclass),
            {}
        )

        instance = _class(
            **self.get_params()
        )

        return instance
    

    def get_datacache_hash(self):

        def none_to_list(x):
            if x is None:
                return []
            else:
                return x

        return str(hash(
            ':'.join(
                map(str, [
                    self.endogenous_key, 
                    self.exogenous_key, 
                    self.counts_layer,
                    str(self.__class__),
                    *none_to_list(self.continuous_covariates),
                    *none_to_list(self.categorical_covariates),
                    *none_to_list(self.covariates_keys),
                    *none_to_list(self.extra_features_keys),
                ])
            )
        ))

    @staticmethod
    def digitize_categorical_covariate(covar):

        covar = np.array(covar).astype(str).reshape(-1)
        categories = np.unique(covar)
        category_map = dict(zip(categories, range(len(categories))))

        return np.array(
            list(map(lambda x : category_map[x], covar))
        ).astype(int)


    @staticmethod
    def digitize_continuous_covariate(covar):

        covar = np.array(covar).reshape(-1).astype(float)
        mean, std = covar.mean(), covar.std()

        standardized = np.floor(
            (covar - mean)/std
        ).astype(int)

        return np.clip(standardized, -3, 3)


    def train_test_split(self, adata,
        train_size = 0.8, seed = 0,
        stratify = None):

        if stratify is None \
            and (not self.categorical_covariates is None \
            or not self.continuous_covariates is None) :

            covariates_bins = []
            if not self.categorical_covariates is None:
                for covar in self.categorical_covariates:
                    covariates_bins.append(
                        self.digitize_categorical_covariate(
                            adata.obs_vector(covar)
                        )
                    )

            if not self.continuous_covariates is None:
                for covar in self.continuous_covariates:
                    covariates_bins.append(
                        self.digitize_continuous_covariate(
                            adata.obs_vector(covar)
                        )
                    )

            stratify = list(map(tuple,  list(zip(*covariates_bins)) ))

        return train_test_split(adata, 
            train_size = train_size, 
            random_state = seed, 
            shuffle = True, 
            stratify = stratify
        )

    def _recommend_batchsize(self, n_samples):
        if n_samples >= 5000 and n_samples <= 20000:
            return 64
        elif n_samples > 20000:
            return 128
        else:
            return 128

    def _recommend_embedding_size(self, n_samples):
        return None

    def _recommend_hidden(self, n_samples):
        if n_samples <= 1000:
            return 64
        elif n_samples <= 4000:
            return 128
        elif n_samples <= 10000:
            return 256
        else:
            return 512

    def _recommend_num_layers(self, n_samples):
        if n_samples < 2400:
            return 2
        else:
            return 3

    def _recommend_num_topics(self, n_samples):
        
        #boxcox transform of the number of samples
        return int(
            (n_samples**0.2 - 1)/0.2
        )

    def _recommend_cost_beta(self, n_samples):

        if n_samples <= 2000:
            return 1.5
        elif n_samples <= 4000:
            return 1.25
        else:
            return 1.

    def _recommend_num_epochs(self, n_samples):

        return int( np.min([round((200000 / n_samples) * 24), 24]) )
    
    def suggest_parameters(self, tuner, trial): 

        return dict(        
            num_topics = trial.suggest_int('num_topics', tuner.min_topics, 
                tuner.max_topics, log=False),
            decoder_dropout = \
                    trial.suggest_float('decoder_dropout', self._min_dropout, 0.065, log = True)
        )


    def recommend_parameters(self, n_samples, n_features, finetune = False):

        assert isinstance(n_samples, int) and n_samples > 0

        params = {
            'batch_size' : self._recommend_batchsize(n_samples),
            'hidden' : self._recommend_hidden(n_samples),
            'num_layers' : self._recommend_num_layers(n_samples),
            'num_epochs' : self._recommend_num_epochs(n_samples),
            'num_topics' : self._recommend_num_topics(n_samples),
            'embedding_size' : self._recommend_embedding_size(n_samples),
            'cost_beta' : self._recommend_cost_beta(n_samples)
        }

        return params


    @adi.wraps_modelfunc(fetch = tmi.fit_adata, 
        fill_kwargs=['features','highly_variable','dataset'])
    def write_ondisk_dataset(self,dirname = None,*,features, highly_variable, dataset):
        
        if dirname is None:
            raise ValueError('Must provide a "dirname" for to write the dataset')

        tmi.OnDiskDataset.write_to_disk(
            batch_size = self.batch_size,
            dirname = dirname, 
            features = features,
            highly_variable = highly_variable, 
            dataset = dataset
        )


    @staticmethod
    def _iterate_batch_idx(N, batch_size, bar = False, desc = None):
        
        num_batches = N//batch_size + int(N % batch_size > 0)

        for i in range(num_batches) if not bar else tqdm(range(num_batches), desc = desc):

            start, end = i * batch_size, (i + 1) * batch_size

            if N - end == 1:
                yield start, end + 1
                raise StopIteration()
            else:
                yield start, end


    def _set_seeds(self):
        if self.seed is None:
            self.seed = int(time.time() * 1e7)%(2**32-1)

        torch.manual_seed(self.seed)
        pyro.set_rng_seed(self.seed)
        np.random.seed(self.seed)


    def preprocess_read_depth(self, X):
        return np.array(X.sum(-1)).reshape((-1,1)).astype(np.float32)


    def preprocess_categorical_covariates(self, X):

        if X.size > 0:
            return self.categorical_transformer.transform(np.atleast_2d(X))
        else:
            return X

    def preprocess_continuous_covariates(self, X, clip = 10.):

        if X.size > 0:
            return np.clip(
                self.continuous_transformer.transform(np.atleast_2d(X)),
                -clip, clip
            ).astype(np.float32)
        else:
            return X
        

    def get_training_sampler(self):
        
        return dict(batch_size = self.batch_size, 
            shuffle = True, drop_last=True)


    def _get_dataset_statistics(self, dataset, training_bar = True):

        self.categorical_transformer = OneHotEncoder(
            sparse = False, 
            dtype = np.float32
        )
        self.continuous_transformer = StandardScaler()


    def _get_loss_adjustment(self, batch):
        return 64/len(batch['read_depth'])

    def _get_weights(self, on_gpu = True, inference_mode = False,*,
            num_exog_features, num_endog_features, 
            num_covariates, num_extra_features):
        
        try:
            del self.svi, self.encoder, self.decoder
        except AttributeError:
            pass

        gc.collect()
        pyro.clear_param_store()
        torch.cuda.empty_cache()
        self._set_seeds()

        assert(isinstance(self.use_cuda, bool))
        assert(isinstance(self.num_topics, int) and self.num_topics > 0)
        assert(isinstance(self.features, (list, np.ndarray)))
        assert(len(self.features) == self.num_exog_features)
        assert isinstance(self.cost_beta, (int, float)) and self.cost_beta > 0

        use_cuda = torch.cuda.is_available() and self.use_cuda and on_gpu
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        if not use_cuda:
            if not inference_mode:
                logger.warn('Cuda unavailable. Will not use GPU speedup while training.')
            else:
                logger.info('Moving model to CPU for inference.')

        try:
            self.covariates_hidden
        except AttributeError:
            if num_covariates > 0:
                raise ValueError('Cannot use base topic model with covariate correction. Make sure to set the covariates keys.')

        decoder_kwargs = {}
        if num_covariates > 0:
            decoder_kwargs.update({
                'covariates_hidden' : self.covariates_hidden,
                'covariates_dropout' : self.covariates_dropout,
                'mask_dropout' : self.mask_dropout,
            })

        self.decoder = self._decoder_model(
            num_exog_features = num_exog_features,
            num_topics = self.num_topics, 
            num_covariates = num_covariates, 
            topics_dropout = self.decoder_dropout, 
            **decoder_kwargs,
        )

        self.encoder = self.encoder_model(
            embedding_size = self.embedding_size,
            num_endog_features = num_endog_features, 
            num_exog_features = num_exog_features,
            num_topics = self.num_topics, 
            num_covariates = num_covariates,
            num_extra_features = num_extra_features,
            embedding_dropout = self.embedding_dropout,
            hidden = self.hidden, 
            dropout = self.encoder_dropout, 
            num_layers = self.num_layers
        )

        self.K = torch.tensor(self.num_topics, requires_grad = False)
        self.to(self.device)


    def transform_batch(self, data_loader, bar = True, desc = ''):

        for batch in tqdm(data_loader, desc = desc) if bar else data_loader:
            yield {k : v.to(self.device) for k,v in batch.items()}

    def _get_1cycle_scheduler(self, n_batches_per_epoch):
        
        return pyro.optim.lr_scheduler.PyroLRScheduler(OneCycleLR_Wrapper, 
            {'optimizer' : Adam, 
            'optim_args' : {'lr' : self.min_learning_rate, 
                'betas' : (self.beta, 0.999), 
                'weight_decay' : self.weight_decay}, 
            'max_lr' : self.max_learning_rate, 
            'steps_per_epoch' : n_batches_per_epoch, 'epochs' : self.num_epochs, 
            'div_factor' : self.max_learning_rate/self.min_learning_rate,
            'cycle_momentum' : True, 
            'three_phase' : False, 
            'verbose' : False,
            'base_momentum' : 0.85,
            'max_momentum' : 0.95,
            })

    @staticmethod
    def _get_monotonic_kl_factor(step_num, *, n_epochs, n_batches_per_epoch):
        
        total_steps = n_epochs * n_batches_per_epoch
        return min(1., (step_num + 1)/(total_steps * 1/2 + 1)) 

    @staticmethod
    def _get_cyclic_KL_factor(step_num, *, n_epochs, n_batches_per_epoch):
        
        total_steps = n_epochs * n_batches_per_epoch
        n_cycles = 3
        tau = ((step_num+1) % (total_steps/n_cycles))/(total_steps/n_cycles)

        if tau > 0.5 or step_num >= (0.95 * total_steps):
            return 1.
        else:
            return max(tau/0.5, n_cycles/total_steps)


    @staticmethod
    def _get_stepup_cyclic_KL(step_num, *, n_epochs, n_batches_per_epoch):
        
        total_steps = n_epochs * n_batches_per_epoch
        n_cycles = 3
        steps_per_cycle = total_steps/n_cycles + 1

        tau = (step_num % steps_per_cycle)/steps_per_cycle

        if tau > 0.5 or step_num >= (0.95 * total_steps):
            x = 1.
        else:
            x = max(tau/0.5, n_cycles/total_steps)

        return x * min((step_num+1)//steps_per_cycle + 1, n_cycles)/n_cycles


    @property
    def highly_variable(self):
        return self._highly_variable

    @highly_variable.setter
    def highly_variable(self, h):
        assert(isinstance(h, (list, np.ndarray)))
        h = np.ravel(np.array(h))
        assert(h.dtype == bool)
        #assert(len(h) == self.num_exog_features)
        self._highly_variable = h

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, f):
        assert(isinstance(f, (list, np.ndarray)))
        f = np.ravel(np.array(f))
        #assert(len(f) == self.num_exog_features)
        self._features = f

    def _instantiate_model(self, training_bar = True,*, features, highly_variable, dataset):

        assert(isinstance(self.num_epochs, int) and self.num_epochs > 0)
        assert(isinstance(self.batch_size, int) and self.batch_size > 0)
        assert(isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0)

        if self.max_learning_rate is None:
            self.max_learning_rate = self.min_learning_rate
        else:
            assert(isinstance(self.max_learning_rate, float) and self.max_learning_rate > 0)

        self.enrichments = {}
        self.features = features
        self.highly_variable = highly_variable

        self._get_dataset_statistics(dataset, training_bar = training_bar)
        batch = next(iter(dataset.get_dataloader(self, training = True, batch_size = 2)))

        self.num_endog_features = self.highly_variable.sum() #batch['endog_features'].shape[-1]
        self.num_exog_features = len(self.features) #batch['exog_features'].shape[-1]
        self.num_covariates = batch['covariates'].shape[-1]
        self.num_extra_features = batch['extra_features'].shape[-1]
        self.covariate_compensation = self.num_covariates > 0

        self._get_weights(
            on_gpu=True, inference_mode=False,
            num_covariates = self.num_covariates,
            num_exog_features = self.num_exog_features,
            num_endog_features = self.num_endog_features,
            num_extra_features = self.num_extra_features,
        )


    def _step(self, batch, anneal_factor, batch_size_adjustment):

        return {
            'loss' : float(
                    self.svi.step(**batch, 
                        anneal_factor = anneal_factor,
                        batch_size_adjustment = batch_size_adjustment
                    )
                ),
            'anneal_factor' : anneal_factor
            }

    @adi.wraps_modelfunc(fetch = tmi.fit, 
        fill_kwargs=['features','highly_variable','dataset'], requires_adata = False)
    def get_learning_rate_bounds(self, num_steps = 100, eval_every = 3, num_epochs = 3,
        lower_bound_lr = 1e-6, upper_bound_lr = 5,*,
        features, highly_variable, dataset):
        '''
        Use the learning rate range test (LRRT) to determine minimum and maximum learning
        rates that enable the model to traverse the gradient of the loss. 

        Steps through linear increase in log-learning rate from `lower_bound_lr`
        to `upper_bound_lr` while recording loss of model. Learning rates which
        produce greater decreases in model loss mark range of possible
        learning rates.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        num_steps : int, default=100
            Number of steps to run the LRRT.
        eval_every : int, default=3,
            Aggregate this number of batches per evaluation of the objective loss.
            Larger numbers give lower variance estimates of model performance.
        lower_bound_lr : float, default=1e-6
            Start learning rate of LRRT
        upper_bound_lr : float, default=5 
            End learning rate of LRRT

        Returns
        -------
        min_learning_rate : float
            Lower bound of estimated range of optimal learning rates
        max_learning_rate : float
            Upper bound of estimated range of optimal learning rates

        Examples
        --------

        .. code-block:: python

            >>> model.get_learning_rate_bounds(rna_data, num_epochs = 3)
            Learning rate range test: 100%|██████████| 85/85 [00:17<00:00,  4.73it/s]
            (4.619921114045972e-06, 0.1800121741235493)

        '''
        self._instantiate_model(
            features = features, 
            highly_variable = highly_variable, 
            dataset = dataset
        )

        data_loader = dataset.get_dataloader(self, 
            training=True, batch_size=self.batch_size)

        #n_batches = len(data_loader)
        eval_steps = int(num_steps)# ceil((n_batches * num_epochs)/eval_every)

        learning_rates = np.exp(
                np.linspace(np.log(lower_bound_lr), 
                np.log(upper_bound_lr), 
                eval_steps+1)
            )

        def lr_function(e):
            return learning_rates[e]/learning_rates[0]

        scheduler = pyro.optim.LambdaLR(
            {'optimizer': Adam, 'optim_args': {'lr': learning_rates[0], 'betas' : (0.90, 0.999), 'weight_decay' : self.weight_decay},
            'lr_lambda' : lr_function})

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        batches_complete, step_loss = 0,0
        learning_rate_losses = []
        
        try:

            t = trange(eval_steps-2, desc = 'Learning rate range test', leave = True)
            _t = iter(t)

            while True:

                self.train()
                for batch in self.transform_batch(data_loader, bar = False):
                    
                    try:
                        step_loss += self._step(batch, self.cost_beta, self._get_loss_adjustment(batch))['loss']
                    except ValueError:
                        raise ModelParamError()

                    batches_complete+=1
                    
                    if batches_complete % eval_every == 0 and batches_complete > 0:
                        scheduler.step()
                        learning_rate_losses.append(step_loss/(eval_every * self.batch_size * self.num_exog_features))
                        step_loss = 0.0
                        
                        next(_t)

        except (ModelParamError, StopIteration) as err:
            pass

        self.gradient_lr = np.array(learning_rates[:len(learning_rate_losses)])
        self.gradient_loss = np.array(learning_rate_losses)

        return self.trim_learning_rate_bounds()


    @staticmethod
    def _define_boundaries(learning_rate, loss, lower_bound_trim, upper_bound_trim):

        assert(isinstance(learning_rate, np.ndarray))
        assert(isinstance(loss, np.ndarray))
        assert(learning_rate.shape == loss.shape)
        
        learning_rate = np.log(learning_rate)
        bounds = learning_rate.min()-1, learning_rate.max()+1
        
        x = np.concatenate([[bounds[0]], learning_rate, [bounds[1]]])
        y = np.concatenate([[loss.min()], loss, [loss.max()]])
        spline_fit = interpolate.splrep(x, y, k = 5, s= 5)
        
        x_fit = np.linspace(*bounds, 100)
        
        first_div = interpolate.splev(x_fit, spline_fit, der = 1)
        
        cross_points = np.concatenate([[0], np.argwhere(np.abs(np.diff(np.sign(first_div))) > 0)[:,0], [len(first_div) - 1]])
        longest_segment = np.argmax(np.diff(cross_points))
        
        left, right = cross_points[[longest_segment, longest_segment+1]]
        
        start, end = x_fit[[left, right]] + np.array([lower_bound_trim, -upper_bound_trim])
        #optimal_lr = x_fit[left + first_div[left:right].argmin()]
        
        return np.exp(start), np.exp(end), spline_fit

    def set_learning_rates(self, min_lr, max_lr):
        '''
        Set the lower and upper learning rate bounds for the One-cycle
        learning rate policy.

        Parameters
        ----------
        min_lr : float
            Lower learning rate boundary
        max_lr : float
            Upper learning rate boundary

        Returns
        -------
        None        
        '''
        self.set_params(min_learning_rate = min_lr, max_learning_rate= max_lr)


    def trim_learning_rate_bounds(self, 
        lower_bound_trim = 0., 
        upper_bound_trim = 0.5):
        '''
        Adjust the learning rate boundaries for the One-cycle learning rate policy.
        The lower and upper bounds should span the learning rates with the 
        greatest downwards slope in loss.

        Parameters
        ----------
        lower_bound_trim : float>=0, default=0.
            Log increase in learning rate of lower bound relative to estimated
            boundary given from LRRT. For example, if the estimated boundary by
            LRRT is 1e-4 and user gives `lower_bound_trim`=1, the new lower
            learning rate bound is set at 1e-3.
        upper_bound_trim : float>=0, default=0.5,
            Log decrease in learning rate of upper bound relative to estimated
            boundary give from LRRT. 

        Returns
        -------
        min_learning_rate : float
            Lower bound of estimated range of optimal learning rates
        max_learning_rate : float
            Upper bound of estimated range of optimal learning rates

        Examples
        --------

        .. code-block:: python

            >>> model.trim_learning_rate_bounds(2, 1)
            (4.619921114045972e-04, 0.1800121741235493e-1)
        '''

        try:
            self.gradient_lr
        except AttributeError:
            raise Exception('User must run "get_learning_rate_bounds" before running this function')

        assert(isinstance(lower_bound_trim, (int,float)) and lower_bound_trim >= 0)
        assert(isinstance(upper_bound_trim, (float, int)) and upper_bound_trim >= 0)

        min_lr, max_lr, self.spline = \
            self._define_boundaries(self.gradient_lr, 
                                    self.gradient_loss, 
                                    lower_bound_trim = lower_bound_trim,
                                    upper_bound_trim = upper_bound_trim,
            )

        self.set_learning_rates(min_lr, max_lr)
        logger.info('Set learning rates to: ' + str((min_lr, max_lr)))
        return min_lr, max_lr


    def plot_learning_rate_bounds(self, figsize = (10,7), ax = None,
            show_spline = True):
        '''
        Plot the loss vs. learning rate curve generated by the LRRT test with
        the current boundaries.

        Parameters
        ----------
        figsize : tuple(int, int), default=(10,7)
            Size of the figure
        ax : matplotlib.pyplot.axes or None, default = None
            Pre-supplied axes for plot. If None, will generate new axes

        Returns
        -------
        ax : matplotlib.pyplot.axes

        Examples
        --------

        When setting the learning rate bounds for the topic model, first
        run the LLRT test with `get_learning_rate_bounds`. Then,
        set the bounds on their respective ends of the part of the LRRT
        plot with the greatest slope, like below.

        .. code-block:: python
            
            >>> model.trim_learning_rate_bounds(5,1) # adjust the bounds
            >>> model.plot_learning_rate_bounds()

        .. image:: /_static/mira.topics.plot_learning_rate_bounds.svg
            :width: 400

        *If the LRRT line appears to vary cyclically*, that means your 
        batches may be not be independent of each other (perhaps batches
        later in the epoch are more similar to eachother and more difficult than
        earlier batches. If this happens, randomize the order of your
        input data using.
        '''

        try:
            self.gradient_lr
        except AttributeError:
            raise Exception('User must run "get_learning_rate_bounds" before running this function')

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = figsize)

        x = np.log(self.gradient_lr)
        bounds = x.min(), x.max()

        x_fit = np.linspace(*bounds, 100)
        y_spline = interpolate.splev(x_fit, self.spline)

        ax.scatter(self.gradient_lr, self.gradient_loss, color = 'lightgrey', label = 'Batch Loss')

        if show_spline:
            ax.plot(np.exp(x_fit), y_spline, color = 'grey', label = '')

        ax.axvline(self.min_learning_rate, color = 'black', label = 'Min Learning Rate')
        ax.axvline(self.max_learning_rate, color = 'red', label = 'Max Learning Rate')

        legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, fontsize='large', bbox_to_anchor=(1.0, 1.05))
        ax.legend(**legend_kwargs)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(xlabel = 'Learning Rate', ylabel = 'Loss', xscale = 'log')
        return ax


    @adi.wraps_modelfunc(tmi.fit, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'], requires_adata = False)
    def instantiate_model(self,*, features, highly_variable, dataset):
        '''
        Given the exogenous and enxogenous features provided, instantiate weights
        of neural network. Called internally by `fit`.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model

        Returns
        -------
        self : object
        '''
        self._instantiate_model(
                features = features, highly_variable = highly_variable, 
                dataset = dataset,
            )

        return self


    def _fit(self, writer = None, training_bar = True, reinit = True, log_every = 10,*,
            dataset, features, highly_variable):
        
        if reinit:
            self._instantiate_model(
                features = features, 
                highly_variable = highly_variable, 
                dataset = dataset,
                training_bar = training_bar
            )

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4, convergence_check=False)

        n_observations = len(dataset)
        data_loader = dataset.get_dataloader(self, 
            training=True,
            batch_size=self.batch_size
        )
        n_batches = len(data_loader)

        scheduler = self._get_1cycle_scheduler(n_batches)
        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())
        self.training_loss = []
        
        anneal_fn = partial(self._get_stepup_cyclic_KL if self.kl_strategy == 'cyclic' else self._get_monotonic_kl_factor, 
            n_epochs = self.num_epochs, n_batches_per_epoch = n_batches)

        step_count = 0
        t = trange(self.num_epochs, desc = 'Training model', leave = True) if training_bar else range(self.num_epochs)
        _t = iter(t)
        next(_t)
        epoch = 0

        while True:
            
            self.train()
            running_loss = 0.0
            for batch in self.transform_batch(data_loader, bar = False):
                
                anneal_factor = anneal_fn(step_count) * self.cost_beta

                try:
                    
                    metrics = self._step(batch, anneal_factor, self._get_loss_adjustment(batch))
                    
                except ValueError:
                    raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate.\nTry setting a lower maximum learning rate or changing the model seed.')

                if not writer is None and step_count % log_every == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, step_count)

                running_loss+=metrics['loss']
                step_count+=1

                if epoch < self.num_epochs:
                    scheduler.step()
            
            epoch_loss = running_loss/(n_observations * self.num_exog_features)
            self.training_loss.append(epoch_loss)
            recent_losses = self.training_loss[-5:]

            try:
                next(_t)
            except StopIteration:
                pass

            if early_stopper(recent_losses[-1]) and epoch > self.num_epochs:
                break

            epoch+=1

            yield epoch, epoch_loss, anneal_factor

        self.set_device('cpu')
        self.eval()
        return self


    @adi.wraps_modelfunc(tmi.fit, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'], requires_adata = False)
    def fit(self, writer = None, reinit = True, log_every = 10,*,
        features, highly_variable, dataset):
        '''
        Initializes new weights, then fits model to data.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        writer : torch.utils.tensorboard.SummarWriter object or mira.topics.Writer object
            Tracks per-batch loss metrics during training. The tensorboard object
            writes tensorboard log files, which can be visualized using TensorBoard.
            The MIRA Writer just tracks values in an in-memory dictionary.
        log_every : int, default = 10
            Average the batch metrics every "log_every" steps.
        reinit : boolean : default = True
            If fitting an already-trained model, whether to re-initialize weights with
            random values or keep the existing weights.

        Returns
        -------
        self : object
            Fitted topic model

        .. note::

            Fitting a topic model usually takes a 2-5 minutes for the expression model
            and 5-10 minutes for the accessibility model for a typical experiment
            (5K to 40K cells).

            Tuning the topic model hyperparameters, however, can take longer.
            Finding the best number of topics significantly increases
            the interpretability of the model and its faithfullness to the underlying
            biology and is well worth the wait. To learn about topic model tuning, 
            see :ref:`mira.topics.TopicModelTuner`.

        '''
        for _ in self._fit(writer = writer, reinit = reinit, log_every = log_every,
            features = features, highly_variable = highly_variable, dataset = dataset):
            pass

        return self

    @adi.wraps_modelfunc(tmi.fit, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'], requires_adata = False)
    def _internal_fit(self, writer = None, log_every = 10,*,
            features, highly_variable, dataset):

        return self._fit(training_bar = False, writer = writer, log_every = log_every,
            features = features, highly_variable = highly_variable, 
            dataset=dataset)


    def _run_encoder_fn(self, fn, dataset, batch_size = 256, bar = True, desc = 'Predicting latent vars'):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')

        data_loader = dataset.get_dataloader(self, training=False,
            batch_size=batch_size)

        results = []
        for batch in self.transform_batch(data_loader, bar = bar, desc = desc):

            results.append(fn(batch['endog_features'], batch['read_depth'], batch['covariates'], batch['extra_features']))

        results = np.vstack(results)
        return results


    def get_topic_feature_distribution(self):

        topics = np.abs(self._get_gamma())[np.newaxis, :] * self._score_features() + self._get_bias()[np.newaxis, :]
        topics = np.sqrt(np.exp(topics)/np.exp(topics).sum(-1, keepdims = True)) #softmax

        return topics

    def cluster_topics(self):

        linkage_matrix = linkage(self.get_topic_feature_distribution(), 
            method='ward', metric='euclidean')

        return linkage_matrix
    

    @adi.wraps_modelfunc(tmi.fetch_features, fill_kwargs=['dataset'], requires_adata = False)
    def _predict_topic_comps_direct_return(self, batch_size = 256, bar = True,*, dataset):

        return self._run_encoder_fn(self.encoder.sample_posterior, 
                dataset, batch_size = batch_size, bar = bar) #.mean(-1)
    
    

    @adi.wraps_modelfunc(tmi.fetch_features, tmi.add_topic_comps,
        fill_kwargs=['dataset'])
    def predict(self, batch_size = 256, bar = True, box_cox = 0.25, *, dataset):
        '''
        Predict the topic compositions of cells in the data. Adds the 
        topic compositions to the `.obsm` field of the adata object.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        batch_size : int>0, default=512
            Minibatch size to run cells through encoder network to predict 
            topic compositions. Set to highest value where tensors will fit in
            memory to increase speed.

        Returns
        -------
        adata : anndata.AnnData
            `.obsm['X_topic_compositions']` : np.ndarray[float] of shape (n_cells, n_topics)
                Topic compositions of cells
            `.obsm['X_umap_features']` : np.ndarray[float] of shape (n_cells, n_topics)
                ILR transformed topic embeddings for use with nearest neighbors graph
            `.obs['topic_1'] ... .obs['topic_N']` : np.ndarray[float] of shape (n_cells,)
                Columns for the activation of each topic.

        Examples
        --------

        .. code-block:: python

            >>> model.predict(adata)
            Predicting latent vars: 100%|█████████████████████████| 36/36 [00:03<00:00,  9.29it/s]
            INFO:mira.adata_interface.topic_model:Added key to obsm: X_topic_compositions
            INFO:mira.adata_interface.topic_model:Added cols: topic_1, topic_2, topic_3, 
            topic_4, topic_5
            >>> scanpy.pp.neighbors(adata, metric = "manhattan", use_rep = "X_umap_features")
            >>> scanpy.tl.umap(adata, min_dist 0.1)
            >>> scanpy.pl.umap(adata, color = model.topic_cols, **mira.pref.topic_umap(ncols = 3))

        .. image:: /_static/mira.topics.predict.png
            :width: 1200

        '''

        topic_expectations = self._run_encoder_fn(self.encoder.sample_posterior, 
                                    dataset, batch_size = batch_size, bar = bar) #.mean(-1)

        basis = ilr.gram_schmidt_basis(topic_expectations.shape[-1])
        ilr_transformed = ilr.centered_boxcox_transform(topic_expectations, a = box_cox).dot(basis)

        return dict(
            cell_topic_dists = topic_expectations,
            topic_feature_dists = self.get_topic_feature_distribution(),
            topic_feature_activations = self._score_features(),
            feature_names = self.features,
            umap_features = ilr_transformed,
            topic_dendogram = self.cluster_topics(),
        )

    @adi.wraps_modelfunc(tmi.fetch_features, 
        fill_kwargs=['dataset'])
    def _untransformed_Z(self, batch_size = 256, bar = True,*, dataset):

        return self._run_encoder_fn(self.encoder.untransformed_Z, 
                dataset, batch_size = batch_size, bar = bar)


    @adi.wraps_modelfunc(tmi.fetch_topic_comps_and_linkage_matrix, 
        partial(adi.add_obsm, add_key = 'X_umap_features'),
        fill_kwargs=['topic_compositions','linkage_matrix'])
    def get_hierarchical_umap_features(self, box_cox = 0.5,*, 
        topic_compositions, linkage_matrix):
        '''
        Leverage the hiearchical relationship between topic-feature distributions
        to prduce a balance matrix for isometric logratio transformation. The 
        function `get_umap_features` uses an arbitrary balance matrix that 
        does not account for the relationship between topics.

        The "UMAP features" are the transformation of the topic compositions that
        encode biological similarity in the high-dimensional space. Use those
        features to produce a joint KNN graph for downstream analysis using UMAP, 
        clustering, and pseudotime inference.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        box_cox : "log" or float between 0 and 1
            Constant for box-cox transformation of topic compositions

        Returns
        -------
        adata : anndata.AnnData
            `.obsm['X_umap_features']` : np.ndarray[float] of shape (n_cells, n_topics)
                Transformed topic compositions of cells
            `.uns['topic_dendogram']` : np.ndarray
                linkage matrix given by scipy.cluster.hierarchy.linkage of
                hiearchical relationship between topic-feature distributions.
                Hierarchy calculated by hellinger distance and ward linkage.

        Examples
        --------

        .. code-block:: python

            >>> model.get_hierarchical_umap_features(adata) # to make features
            INFO:mira.adata_interface.topic_model:Fetching key X_topic_compositions from obsm
            INFO:mira.adata_interface.core:Added key to obsm: X_umap_features
            INFO:mira.adata_interface.topic_model:Added key to uns: topic_dendogram
            >>> scanpy.pp.neighbors(adata, metric = "manhattan", use_rep = "X_umap_features") # to make KNN graph

        '''

        g_matrix = ilr.get_hierarchical_gram_schmidt_basis(
            topic_compositions.shape[-1], linkage_matrix)
        umap_features = ilr.centered_boxcox_transform(topic_compositions, a = box_cox).dot(g_matrix)

        return umap_features

    
    @adi.wraps_modelfunc(tmi.fetch_topic_comps, partial(adi.add_obsm, add_key = 'X_umap_features'),
        fill_kwargs=['topic_compositions', 'covariates','extra_features'])
    def get_umap_features(self, box_cox = 0.5, *, topic_compositions, covariates, extra_features):
        '''
        Predict transformed topic compositions for each cell to derive nearest-
        neighbors graph. Projects topic compositions to orthonormal space using
        isometric logratio transformation.

        The "UMAP features" are the transformation of the topic compositions that
        encode biological similarity in the high-dimensional space. Use those
        features to produce a joint KNN graph for downstream analysis using UMAP, 
        clustering, and pseudotime inference.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        box_cox : "log" or float between 0 and 1
            Constant for box-cox transformation of topic compositions

        Returns
        -------
        adata : anndata.AnnData
            `.obsm['X_umap_features']` : np.ndarray[float] of shape (n_cells, n_topics)
                Transformed topic compositions of cells

        Examples
        --------

        .. code-block:: python

            >>> model.get_umap_features(adata) # to make features
            INFO:mira.adata_interface.topic_model:Fetching key X_topic_compositions from obsm
            INFO:mira.adata_interface.core:Added key to obsm: X_umap_features
            >>> scanpy.pp.neighbors(adata, metric = "manhattan", use_rep = "X_umap_features") # to make KNN graph
        '''
        
        basis = ilr.gram_schmidt_basis(topic_compositions.shape[-1])
        return ilr.centered_boxcox_transform(topic_compositions, a = box_cox).dot(basis)


    def _evaluate_vae_loss(self, model, losses, 
        batch_size = 256, bar = False,*, dataset):

        self.eval()
        self._set_seeds()
        
        data_loader = dataset.get_dataloader(self, training=False, 
            batch_size = batch_size)
        
        results = []
        for batch in self.transform_batch(data_loader, bar = bar):
            with torch.no_grad():

                results.append(
                    list(map(lambda loss_fn : float(loss_fn(model, self.guide, **batch, anneal_factor = 1.)), 
                        losses))
                )

        return list(map(lambda x : sum(x)/len(dataset), list(zip(*results))))


    @adi.wraps_modelfunc(tmi.fetch_features, adi.return_output,
        fill_kwargs=['dataset'], requires_adata = False)
    def distortion_rate_loss(self, batch_size = 256, bar = False, 
            _beta_weight = 1.,*, dataset):
        
        return self._distortion_rate_loss(batch_size= batch_size, _beta_weight = _beta_weight,
            dataset = dataset)


    def _distortion_rate_loss(self, batch_size = 256, _beta_weight = 1., bar = False,*,dataset):

        self.eval()
        vae_loss, rate = self._evaluate_vae_loss(
                self.model, [TraceMeanField_ELBO().loss, TraceMeanFieldLatentKL().loss],
                dataset=dataset, batch_size = batch_size,
                bar = bar,
            )

        distortion = vae_loss - rate

        return distortion, rate * _beta_weight, {} #loss_vae/self.num_exog_features

    
    @adi.wraps_modelfunc(tmi.fetch_features, adi.return_output,
        fill_kwargs=['dataset'], requires_adata = False)
    def score(self, batch_size = 256, *, dataset):
        '''
        Get normalized ELBO loss for data. This method is only available on
        topic models that have not been loaded from disk.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        batch_size : int>0, default=512
            Minibatch size to run cells through encoder network to predict 
            topic compositions. Set to highest value where tensors will fit in
            memory to increase speed.

        Returns
        -------
        loss : float

        Examples
        --------

        .. code-block:: python

            >>> model.score(rna_data)
            0.11564

        Notes
        -----
        The `score` method is only available after training a topic model. After
        saving and writing to disk, this function will no longer work.

        Raises
        ------
        AttributeError
            If attempting to run after loading from disk or before training.

        '''
        self.eval()
        return self._evaluate_vae_loss(
                self.model, TraceMeanField_ELBO().loss,
                dataset=dataset, batch_size = batch_size
            )/self.num_exog_features

    
    def _run_decoder_fn(self, fn, latent_composition, covariates,
        batch_size = 256, bar = True, desc = 'Imputing features'):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()

        for start, end in self._iterate_batch_idx(len(latent_composition), batch_size, bar = True, desc = desc):

            yield fn(
                    torch.tensor(latent_composition[start : end], requires_grad = False).to(self.device),
                    torch.tensor(covariates[start : end].astype(np.float32), requires_grad = False).to(self.device) if self.num_covariates > 0 else None,
                ).detach().cpu().numpy()


    def _batched_impute(self, latent_composition, covariates, 
        batch_size = 256, bar = True):

        return self._run_decoder_fn(
                    partial(self.decoder, nullify_covariates = True), 
                    latent_composition, covariates,
                     batch_size= batch_size, bar = bar)
        

    @adi.wraps_modelfunc(tmi.fetch_topic_comps, adi.add_layer,
        fill_kwargs=['topic_compositions','covariates','extra_features'])
    def impute(self, batch_size = 256, bar = True, *, topic_compositions,
        covariates, extra_features):
        '''
        Impute the relative frequencies of features given the cells' topic
        compositions. The value given is *rho* (see manscript for details).

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        batch_size : int>0, default=512
            Minibatch size to run cells through encoder network to predict 
            topic compositions. Set to highest value where tensors will fit in
            memory to increase speed.

        Returns
        -------
        anndata.AnnData
            `.layers['imputed']` : np.ndarray[float] of shape (n_cells, n_features)
                Imputed relative frequencies of features

        Examples
        --------

        .. code-block::

            >>> model.impute(rna_data)
            >>> rna_data
            View of AnnData object with n_obs × n_vars = 18482 × 22293
                layers: 'imputed'
        '''
        return self.features, np.vstack([
            x for x  in self._batched_impute(topic_compositions, covariates,
                batch_size = batch_size, bar = bar)
        ])

    
    def _batched_batch_effect(self, latent_composition, covariates, 
        batch_size = 256, bar = True):

        return self._run_decoder_fn(self.decoder.get_batch_effect, 
                    latent_composition, covariates,
                     batch_size= batch_size, bar = bar)


    @adi.wraps_modelfunc(tmi.fetch_topic_comps, partial(adi.add_layer, add_layer = 'batch_effect'),
        fill_kwargs=['topic_compositions','covariates','extra_features'])
    def get_batch_effect(self, batch_size = 256, bar = True, *, topic_compositions,
        covariates, extra_features):
        '''
        Compute the estimated technical effects for each gene in each cell.

        Parameters
        ----------

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model
        batch_size : int>0, default=512
            Minibatch size to run cells through encoder network to predict 
            topic compositions. Set to highest value where tensors will fit in
            memory to increase speed.

        Returns
        -------
        anndata.AnnData
            `.layers['batch_effect']` : np.ndarray[float] of shape (n_cells, n_features)
                Estimated batch effects
        
        Examples
        --------

        .. code-block::

            >>> model.get_batch_effects(rna_data)
            >>> rna_data
            View of AnnData object with n_obs × n_vars = 18482 × 22293
                layers: 'batch_effect'

        '''

        if self.num_covariates == 0:
            raise ValueError('Cannot compute batch effect with no covariates.')

        return self.features, np.vstack([
            x for x  in self._batched_batch_effect(topic_compositions, covariates,
                batch_size = batch_size, bar = bar)
        ])


    @adi.wraps_modelfunc(tmi.fetch_topic_comps, partial(adi.add_obs_col, colname = 'softmax_denom'), 
        fill_kwargs = ['topic_compositions','covariates', 'extra_features'])
    def _get_softmax_denom(self, topic_compositions, covariates, extra_features,
            batch_size = 256, bar = True, include_batcheffects = True):

        return np.concatenate([
            x for x in self._run_decoder_fn(
                partial(self.decoder.get_softmax_denom, include_batcheffects = include_batcheffects), 
                topic_compositions, covariates,
                batch_size = batch_size, bar = bar, desc = 'Calculating softmax summary data')
        ])

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)


    def _get_save_data(self):
        return dict(
            cls_name = self.__class__.__name__,
            cls_bases = self.__class__.__bases__, 
            weights = self.state_dict(),
            params = self.get_params(),
            fit_params = dict(
                continuous_transformer = self.continuous_transformer,
                categorical_transformer = self.categorical_transformer,
                num_endog_features = self.num_endog_features,
                num_exog_features = self.num_exog_features,
                num_covariates = self.num_covariates,
                num_extra_features = self.num_extra_features,
                highly_variable = self.highly_variable,
                features = self.features,
                enrichments = self.enrichments,
            )
        )

    def save(self, filename):
        '''
        Save topic model.

        Parameters
        ----------
        filename : str
            File name to save topic model, recommend .pth extension
        '''
        torch.save(self._get_save_data(), filename)

    def _set_weights(self, fit_params, weights):

        for param, value in fit_params.items():
            setattr(self, param, value)
        
        self._get_weights(on_gpu = False, inference_mode = True,
                num_endog_features=self.num_endog_features,
                num_extra_features=self.num_extra_features,
                num_exog_features = self.num_exog_features,
                num_covariates= self.num_covariates
            )

        self.load_state_dict(weights)
        self.eval()
        self.to_cpu()
        return self

    def _score_features(self):
        score = np.sign(self._get_gamma()) * (self._get_beta() - self._get_bn_mean())/np.sqrt(self._get_bn_var() + self.decoder.bn.eps)
        return score

    def _get_topics(self):
        return self._score_features()
    
    def _get_beta(self):
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def _get_gamma(self):
        return self.decoder.bn.weight.cpu().detach().numpy()
    
    def _get_bias(self):
        return self.decoder.bn.bias.cpu().detach().numpy()

    def _get_bn_mean(self):
        return self.decoder.bn.running_mean.cpu().detach().numpy()

    def _get_bn_var(self):
        return self.decoder.bn.running_var.cpu().detach().numpy()

    def to_gpu(self):
        '''
        Move topic model to GPU device "cuda:0", if available.
        '''
        self.set_device('cuda:0')
    
    def to_cpu(self):
        '''
        Move topic model to CPU device "cpu", if available.
        '''
        self.set_device('cpu')

    def set_device(self, device):
        '''
        Move topic model to a new device.

        Parameters
        ----------
        device : str
            Name of device on which to allocate model
        '''
        logger.info('Moving model to device: {}'.format(device))
        self.device = device
        self = self.to(self.device)

    @property
    def topic_cols(self):
        '''
        Attribute, returns the names of the columns for the topics added by the
        `predict` method to an anndata object. Useful for quickly accessing
        topic columns for plotting.

        Examples
        --------

        .. code-block::

            >>> model.num_topics
            5
            >>> model.topic_cols
            ['topic_0', 'topic_1','topic_2','topic_3','topic_4']
            >>> sc.pl.umap(adata, color = model.topic_cols, **mira.pref.topic_umap())

        '''
        return ['topic_' + str(i) for i in range(self.num_topics)]
