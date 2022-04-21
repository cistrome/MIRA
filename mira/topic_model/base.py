from functools import partial
from scipy import interpolate
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm.auto import tqdm, trange
import numpy as np
import torch.distributions.constraints as constraints
import logging
from math import ceil
import time
from sklearn.base import BaseEstimator
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
import gc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import mira.topic_model.ilr_tools as ilr
from torch.utils.data import DataLoader
import time
logger = logging.getLogger(__name__)


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


class Decoder(torch.nn.Module):
    
    def __init__(self,*,num_exog_features, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_exog_features, bias = False)
        self.bn = nn.BatchNorm1d(num_exog_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        return F.softmax(self.bn(self.beta(self.drop(inputs))), dim=1)

    def get_softmax_denom(self, inputs):
        return self.bn(self.beta(inputs)).exp().sum(-1)


class ModelParamError(ValueError):
    pass

class OneCycleLR_Wrapper(torch.optim.lr_scheduler.OneCycleLR):

    def __init__(self, optimizer, **kwargs):
        max_lr = kwargs.pop('max_lr')
        super().__init__(optimizer, max_lr, **kwargs)


def encoder_layer(input_dim, output_dim, nonlin = True, dropout = 0.2):
    layers = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim)]
    if nonlin:
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_fc_stack(layer_dims = [256, 128, 128, 128], dropout = 0.2, skip_nonlin = True):
    return nn.Sequential(*[
        encoder_layer(input_dim, output_dim, nonlin= not ((i >= (len(layer_dims) - 2)) and skip_nonlin), dropout = dropout)
        for i, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:]))
    ])


class BaseModel(torch.nn.Module, BaseEstimator):

    @classmethod
    def _load_old_model(cls, filename):

        old_model = torch.load(filename)

        params = old_model['params'].copy()
        params['num_topics'] = params.pop('num_modules')
        
        fit_params = {}
        fit_params['num_exog_features'] = len(params['features'])
        fit_params['num_endog_features'] = params['highly_variable'].sum()
        fit_params['highly_variable'] = params.pop('highly_variable')
        fit_params['features'] = params.pop('features')
        
        model = cls(**params)
        model._set_weights(fit_params, old_model['model']['weights'])

        if 'pi' in old_model['model']:
            model.residual_pi = old_model['model']['pi']

        model.enrichments = {}
        
        return model

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

        data = torch.load(filename)

        model = cls(**data['params'])
        model._set_weights(data['fit_params'], data['weights'])

        return model

    def __init__(self,
            endogenous_key = None,
            exogenous_key = None,
            counts_layer = None,
            num_topics = 16,
            hidden = 128,
            num_layers = 3,
            num_epochs = 40,
            decoder_dropout = 0.2,
            encoder_dropout = 0.1,
            use_cuda = True,
            seed = 0,
            min_learning_rate = 1e-6,
            max_learning_rate = 1e-1,
            beta = 0.95,
            batch_size = 64,
            initial_pseudocounts = 50,
            nb_parameterize_logspace = True,
            embedding_size = None,
            kl_strategy = 'monotonic',
            reconstruction_weight = 1.,
            dataset_loader_workers = 0,
            ):
        '''
        Learns regulatory "topics" from single-cell multiomics data. Topics capture 
        patterns of covariance between gene or cis-regulatory elements. 
        Each cell is represented by a composition over topics, and each 
        topic corresponds with activations of co-regulated elements.

        One may use enrichment analysis of the topics to understand signaling 
        and transcription factor drivers of cell states, and embedding of 
        cell-topic distributions to visualize and cluster cells, 
        and to perform pseudotime trajectory inference.

        Parameters
        ----------
        endogenous_key : str, default=None
            Column in AnnData that marks features to be used for encoder neural network. 
            These features should prioritize elements that distinguish 
            between populations, like highly-variable genes.
        exogenous_key : str, default=None
            Column in AnnData that marks features to be used for decoder neural network. 
            These features should include all elements used for enrichment analysis of topics. 
            Commonly, this will be genes with a certain amout of expression that 
            are not necessarily highly variable. For accessibility data, all called peaks may be used.
        counts_layer : str, default=None
            Layer in AnnData that countains raw counts for modeling.
        num_topics : int, default=16
            Number of topics to learn from data.
        hidden : int, default=128
            Number of nodes to use in hidden layers of encoder network
        num_layers: int, default=3
            Number of layers to use in encoder network, including output layer
        num_epochs: int, default=40
            Number of epochs to train topic model. The One-cycle learning rate policy
            requires a pre-defined training length, and 40 epochs is 
            usually an overestimate of the optimal number of epochs to train for.
        decoder_dropout : float (0., 1.), default=0.2
            Dropout rate for the decoder network. Prevents node collapse.
        encoder_dropout : float (0., 1.), default=0.2
            Dropout rate for the encoder network. Prevents overfitting.
        use_cuda : boolean, default=True
            Try using CUDA GPU speedup while training.
        seed : int, default=None
            Random seed for weight initialization. 
            Enables reproduceable initialization of model.
        dataset_loader_workers : int>=0, default = 0
            Number of workers to use for dataset loading and preprocessing of each batch.
            Batches are loaded and transformed as a stream from the in-memory AnnData object.
            For 0, transformation occurs in the main process. For the accessibility topic model,
            three workers is recommended.**
        min_learning_rate : float, default=1e-6
            Start learning rate for One-cycle learning rate policy.
        max_learning_rate : float, default=1e-1
            Peak learning rate for One-cycle policy. 
        beta : float, default=0.95
            Momentum parameter for ADAM optimizer.
        batch_size : int, default=64
            Minibatch size for stochastic gradient descent while training. 
            Larger batch sizes train faster, but may produce less optimal models.
        initial_pseudocounts : int, default=50
            Initial pseudocounts allocated to approximated hierarchical dirichlet prior.
            More pseudocounts produces smoother topics, 
            less pseudocounts produces sparser topics. 
        nb_parameterize_logspace : boolean, default=True
            Parameterize negative-binomial distribution using log-space probability 
            estimates of gene expression. Is more numerically stable.
        embedding_size : int > 0 or None, default=None
            Number of nodes in first encoder neural network layer. Default of *None*
            gives an embedding size of `hidden`.
        kl_strategy : {'monotonic','cyclic'}, default='monotonic'
            Whether to anneal KL term using monotonic or cyclic strategies. Cyclic
            may produce slightly better models.

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

        Examples
        --------

        .. code-block:: python

            >>> model = mira.topics.ExpressionTopicModel(
            ...    exogenous_key = 'exog', 
            ...     endogenous_key = 'endog',
            ...     counts_layer = 'rawcounts',
            ...     num_topics = 15,
            ... )
            >>> model.fit(adata)
            >>> model.predict(adata)
                
        '''
        super().__init__()

        self.endogenous_key = endogenous_key
        self.exogenous_key = exogenous_key
        self.counts_layer = counts_layer
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

    def get_endog_fn(self):
        raise NotImplementedError()
                   
    def get_exog_fn(self):
        raise NotImplementedError()

    def get_rd_fn(self):
        
        def preprocess_read_depth(X):
            return np.array(X.sum(-1)).reshape((-1,1)).astype(np.float32)
        
        return preprocess_read_depth

    def get_training_sampler(self):
        
        return dict(batch_size = self.batch_size, 
            shuffle = True, drop_last=True,)
        
    def get_dataloader(self, dataset, training = False):

        if training:
            extra_kwargs = dict(
                shuffle = True, drop_last = True,
            )
            if self.dataset_loader_workers > 0:
                extra_kwargs.update(dict(
                    num_workers = self.dataset_loader_workers,
                    prefetch_factor = 5
                ))
        else:
            extra_kwargs = {}

        return DataLoader(
            dataset, 
            batch_size = self.batch_size, 
            **extra_kwargs,
            collate_fn= partial(tmi.collate_batch,
                preprocess_endog = self.get_endog_fn(), 
                preprocess_exog = self.get_exog_fn(), 
                preprocess_read_depth = self.get_rd_fn(),
            )
        )

    def _get_dataset_statistics(self, dataset):
        pass

    def _get_weights(self, on_gpu = True, inference_mode = False):
        
        try:
            del self.svi
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
        assert isinstance(self.reconstruction_weight, (int, float)) and self.reconstruction_weight > 0

        use_cuda = torch.cuda.is_available() and self.use_cuda and on_gpu
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        if not use_cuda:
            if not inference_mode:
                logger.warn('Cuda unavailable. Will not use GPU speedup while training.')
            else:
                logger.info('Moving model to CPU for inference.')

        self.decoder = Decoder(
            num_exog_features=self.num_exog_features, 
            num_topics=self.num_topics, 
            dropout = self.decoder_dropout
        )

        self.encoder = self.encoder_model(
            embedding_size = self.embedding_size,
            num_endog_features = self.num_endog_features, 
            num_topics = self.num_topics, 
            hidden = self.hidden, 
            dropout = self.encoder_dropout, 
            num_layers = self.num_layers
        )

        self.K = torch.tensor(self.num_topics, requires_grad = False)
        #self.eps = torch.tensor(5.0e-3, requires_grad=False)
        self.to(self.device)


    #,*, endog_features, exog_features, read_depth, anneal_factor = 1.
    def model(self):
        
        pyro.module("decoder", self.decoder)

        _alpha, _beta = self._get_gamma_parameters(self.initial_pseudocounts, self.num_topics)
        with pyro.plate("topics", self.num_topics):
            initial_counts = pyro.sample("a", dist.Gamma(self._to_tensor(_alpha), self._to_tensor(_beta)))

        theta_loc = self._get_prior_mu(initial_counts, self.K)
        theta_scale = self._get_prior_std(initial_counts, self.K)

        return theta_loc.to(self.device), theta_scale.to(self.device)


    def guide(self):

        #assert(self.initial_pseudocounts > self.num_topics), 'Initial counts must be greater than the number of topics.'

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(
            *self._get_gamma_moments(self.initial_pseudocounts, self.num_topics)
        )

        pseudocount_mu = pyro.param('pseudocount_mu', _counts_mu * torch.ones((self.num_topics,)).to(self.device))

        pseudocount_std = pyro.param('pseudocount_std', np.sqrt(_counts_var) * torch.ones((self.num_topics,)).to(self.device), 
                constraint = constraints.positive)

        pyro.module("encoder", self.encoder)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(pseudocount_mu, pseudocount_std))


    @staticmethod
    def _get_gamma_parameters(I, K):
        return 2., 2*K/I

    @staticmethod
    def _get_gamma_moments(I,K):
        return I/K, 0.5 * (I/K)**2

    @staticmethod
    def _get_lognormal_parameters_from_moments(m, v):
        m_squared = m**2
        mu = np.log(m_squared / np.sqrt(v + m_squared))
        var = np.log(v/m_squared + 1)

        return mu, var

    @staticmethod
    def _get_prior_mu(a, K):
        return a.log() - 1/K * torch.sum(a.log())

    @staticmethod
    def _get_prior_std(a, K):
        return torch.sqrt(1/a * (1-2/K) + 1/(K * a))

    
    def transform_batch(self, data_loader, bar = True, desc = ''):

        for batch in tqdm(data_loader, desc = desc) if bar else data_loader:
            yield {k : torch.tensor(v, requires_grad = False).to(self.device)
                for k, v in batch.items()}

    def _get_1cycle_scheduler(self, n_batches_per_epoch):
        
        return pyro.optim.lr_scheduler.PyroLRScheduler(OneCycleLR_Wrapper, 
            {'optimizer' : Adam, 'optim_args' : {'lr' : self.min_learning_rate, 'betas' : (self.beta, 0.999)}, 'max_lr' : self.max_learning_rate, 
            'steps_per_epoch' : n_batches_per_epoch, 'epochs' : self.num_epochs, 'div_factor' : self.max_learning_rate/self.min_learning_rate,
            'cycle_momentum' : False, 'three_phase' : False, 'verbose' : False})

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


    @property
    def highly_variable(self):
        return self._highly_variable

    @highly_variable.setter
    def highly_variable(self, h):
        assert(isinstance(h, (list, np.ndarray)))
        h = np.ravel(np.array(h))
        assert(h.dtype == bool)
        assert(len(h) == self.num_exog_features)
        self._highly_variable = h

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, f):
        assert(isinstance(f, (list, np.ndarray)))
        f = np.ravel(np.array(f))
        assert(len(f) == self.num_exog_features)
        self._features = f

    def _instantiate_model(self,*, features, highly_variable):

        assert(isinstance(self.num_epochs, int) and self.num_epochs > 0)
        assert(isinstance(self.batch_size, int) and self.batch_size > 0)
        assert(isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0)
        if self.max_learning_rate is None:
            self.max_learning_rate = self.min_learning_rate
        else:
            assert(isinstance(self.max_learning_rate, float) and self.max_learning_rate > 0)

        self.enrichments = {}
        self.num_endog_features = highly_variable.sum()
        self.num_exog_features = len(features)
        self.features = features
        self.highly_variable = highly_variable
        
        self._get_weights()

    def _step(self, batch, anneal_factor):

        return {
            'loss' : float(self.svi.step(**batch, anneal_factor = anneal_factor))/self.batch_size,
            'anneal_factor' : anneal_factor
            }

    @adi.wraps_modelfunc(fetch = tmi.fit_adata, 
        fill_kwargs=['features','highly_variable','dataset'])
    def get_learning_rate_bounds(self, num_epochs = 6, eval_every = 10, 
        lower_bound_lr = 1e-6, upper_bound_lr = 1,*,
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
        num_epochs : int, default=6
            Number of epochs to run the LRRT, may be decreased for larger datasets.
        eval_every : int, default=10,
            Aggregate this number of batches per evaluation of the objective loss.
            Larger numbers give lower variance estimates of model performance.
        lower_bound_lr : float, default=1e-6
            Start learning rate of LRRT
        upper_bound_lr : float, default=1 
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
            features = features, highly_variable = highly_variable
        )

        data_loader = self.get_dataloader(dataset, training=True)

        self._get_dataset_statistics(dataset)

        n_batches = len(dataset)//self.batch_size
        eval_steps = ceil((n_batches * num_epochs)/eval_every)

        learning_rates = np.exp(
                np.linspace(np.log(lower_bound_lr), 
                np.log(upper_bound_lr), 
                eval_steps+1)
            )

        self.learning_rates = learning_rates

        def lr_function(e):
            return learning_rates[e]/learning_rates[0]

        scheduler = pyro.optim.LambdaLR({'optimizer': Adam, 
            'optim_args': {'lr': learning_rates[0], 'betas' : (0.95, 0.999)}, 
            'lr_lambda' : lr_function})

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        batches_complete, step_loss = 0,0
        learning_rate_losses = []
        
        try:

            t = trange(eval_steps-2, desc = 'Learning rate range test', leave = True)
            _t = iter(t)

            for epoch in range(num_epochs + 1):

                self.train()
                for batch in self.transform_batch(data_loader, bar = False):

                    step_loss += self._step(batch, 1.)['loss']
                    batches_complete+=1
                    
                    if batches_complete % eval_every == 0 and batches_complete > 0:
                        scheduler.step()
                        learning_rate_losses.append(step_loss/(self.batch_size * self.num_exog_features))
                        step_loss = 0.0
                        try:
                            next(_t)
                        except StopIteration:
                            break

        except ValueError:
            logger.error('\nGradient overflow from too high learning rate, stopping test early.')

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


    def plot_learning_rate_bounds(self, figsize = (10,7), ax = None):
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
        ax.plot(np.exp(x_fit), y_spline, color = 'grey', label = '')
        ax.axvline(self.min_learning_rate, color = 'black', label = 'Min Learning Rate')
        ax.axvline(self.max_learning_rate, color = 'red', label = 'Max Learning Rate')

        legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, fontsize='large', bbox_to_anchor=(1.0, 1.05))
        ax.legend(**legend_kwargs)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(xlabel = 'Learning Rate', ylabel = 'Loss', xscale = 'log')
        return ax


    @adi.wraps_modelfunc(tmi.fit_adata, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'])
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
            )

        return self


    def _fit(self, writer = None, training_bar = True, reinit = True,*,
            dataset, features, highly_variable):
        
        if reinit:
            self._instantiate_model(
                features = features, highly_variable = highly_variable, 
            )

        self._get_dataset_statistics(dataset)

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4, convergence_check=False)

        n_batches = len(dataset)//self.batch_size
        n_observations = len(dataset)//self.batch_size
        data_loader = self.get_dataloader(dataset, training=True)

        scheduler = self._get_1cycle_scheduler(n_batches)
        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())
        self.training_loss = []
        
        anneal_fn = partial(self._get_cyclic_KL_factor if self.kl_strategy == 'cyclic' else self._get_monotonic_kl_factor, 
            n_epochs = self.num_epochs, n_batches_per_epoch = n_batches)

        step_count = 0
        t = trange(self.num_epochs, desc = 'Epoch 0', leave = True) if training_bar else range(self.num_epochs)
        _t = iter(t)
        epoch = 0
        while True:
            
            self.train()
            running_loss = 0.0
            for batch in self.transform_batch(data_loader, bar = False):
                
                anneal_factor = anneal_fn(step_count)

                try:
                    metrics = self._step(batch, anneal_factor)

                    if not writer is None:
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, step_count)

                    running_loss+=metrics['loss']
                    step_count+=1

                except ValueError:
                    raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate. Try setting a lower learning rate.')

                if epoch < self.num_epochs:
                    scheduler.step()
            
            epoch_loss = running_loss/(n_observations * self.num_exog_features)
            self.training_loss.append(epoch_loss)
            recent_losses = self.training_loss[-5:]

            if training_bar:
                t.set_description("Epoch {} done. Recent losses: {}".format(
                    str(epoch + 1),
                    ' --> '.join('{:.3e}'.format(loss) for loss in recent_losses)
                ))

            try:
                next(_t)
            except StopIteration:
                pass

            if early_stopper(recent_losses[-1]) and epoch > self.num_epochs:
                break

            epoch+=1

            yield epoch, epoch_loss

        self.set_device('cpu')
        self.eval()
        return self


    @adi.wraps_modelfunc(tmi.fit_adata, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'])
    def fit(self, writer = None, reinit = True,*,
        features, highly_variable, dataset):
        '''
        Initializes new weights, then fits model to data.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility features to model

        Returns
        -------
        self : object
            Fitted topic model

        .. note::

            Fitting a topic model usually takes a 2-5 minutes for the expression model
            and 5-10 minutes for the accessibility model for a typical experiment
            (5K to 40K cells).

            Optimizing the topic model hyperparameters, however, can take much longer.
            We recommend running topic model tuners overnight, which is usually sufficient
            training time. Finding the best number of topics significantly increases
            the interpretability of the model and its faithfullness to the underlying
            biology and is well worth the wait.

            To learn about topic model tuning, see :ref:`mira.topics.TopicModelTuner`.
        '''
        for _ in self._fit(writer = writer, reinit = reinit, 
            features = features, highly_variable = highly_variable, dataset = dataset):
            pass

        return self

    @adi.wraps_modelfunc(tmi.fit_adata, adi.return_output,
        fill_kwargs=['features','highly_variable','dataset'])
    def _internal_fit(self, writer = None, *,features, highly_variable, dataset):

        return self._fit(training_bar = False, writer = writer,
            features = features, highly_variable = highly_variable, 
            dataset=dataset)


    def _run_encoder_fn(self, fn, dataset, batch_size = 512, bar = True, desc = 'Predicting latent vars'):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')

        data_loader = self.get_dataloader(dataset, training=False)

        results = []
        for batch in self.transform_batch(data_loader, bar = bar, desc = desc):

            results.append(fn(batch['endog_features'], batch['read_depth']))

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


    @adi.wraps_modelfunc(tmi.fetch_features, tmi.add_topic_comps,
        fill_kwargs=['dataset'])
    def predict(self, batch_size = 512,*, dataset):
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

        return dict(
            cell_topic_dists = self._run_encoder_fn(self.encoder.topic_comps, dataset, batch_size = batch_size),
            topic_feature_dists = self.get_topic_feature_distribution(),
            topic_feature_activations = self._score_features(),
            feature_names = self.features,
            topic_dendogram = self.cluster_topics(),
        )


    @adi.wraps_modelfunc(tmi.fetch_topic_comps_and_linkage_matrix, partial(adi.add_obsm, add_key = 'X_umap_features'),
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
        fill_kwargs=['topic_compositions'])
    def get_umap_features(self, box_cox = 0.5, *, topic_compositions):
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


    def _get_elbo_loss(self, batch_size = 512, *, dataset):

        try:
            self.svi
        except AttributeError:
            raise AttributeError('This function only works after training, not before training or after loading from disk.')

        self.eval()
        running_loss = 0
        
        data_loader = self.get_dataloader(dataset, training=False)

        for batch in self.transform_batch(data_loader, bar = False):
            running_loss += float(self.svi.evaluate_loss(**batch, anneal_factor = 1.0))

        return running_loss
    
    @adi.wraps_modelfunc(tmi.fetch_features, adi.return_output,
        fill_kwargs=['dataset'])
    def score(self, batch_size = 512, *, dataset):
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
        return self._get_elbo_loss(
                dataset=dataset, batch_size = batch_size
            )/(len(dataset) * self.num_exog_features)

    
    def _run_decoder_fn(self, fn, latent_composition, batch_size = 512, bar = True, desc = 'Imputing features'):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()

        for start, end in self._iterate_batch_idx(len(latent_composition), batch_size, bar = True, desc = desc):

            yield fn(
                    torch.tensor(latent_composition[start:end], requires_grad = False).to(self.device)
                ).detach().cpu().numpy()


    def _batched_impute(self, latent_composition, batch_size = 512, bar = True):

        return self._run_decoder_fn(self.decoder, latent_composition, 
                    batch_size= batch_size, bar = bar)
        

    @adi.wraps_modelfunc(tmi.fetch_topic_comps, adi.add_layer,
        fill_kwargs=['topic_compositions'])
    def impute(self, batch_size = 512, bar = True, *, topic_compositions):
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
            x for x  in self._batched_impute(topic_compositions, batch_size = batch_size, bar = bar)
        ])


    @adi.wraps_modelfunc(tmi.fetch_topic_comps, partial(adi.add_obs_col, colname = 'softmax_denom'), 
        fill_kwargs = ['topic_compositions'])
    def _get_softmax_denom(self, topic_compositions, batch_size = 512, bar = True):
        return np.concatenate([
            x for x in self._run_decoder_fn(self.decoder.get_softmax_denom, topic_compositions, 
                        batch_size = batch_size, bar = bar, desc = 'Calculating softmax summary data')
        ])

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def _get_save_data(self):
        return dict(
            weights = self.state_dict(),
            params = self.get_params(),
            fit_params = dict(
                num_endog_features = self.num_endog_features,
                num_exog_features = self.num_exog_features,
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
        
        self._get_weights(on_gpu = False, inference_mode = True)

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