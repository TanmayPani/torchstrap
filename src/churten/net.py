"""Neural net base class
"""
import fnmatch
from collections.abc import Mapping, Sized, Iterable
from typing import Callable, Optional
from functools import partial
from collections import OrderedDict
from contextlib import contextmanager
#import os
#import sys
import pickle
import tempfile
import warnings

import numpy as np
from sklearn.base import BaseEstimator

import torch
from torch.nn import Module

from tensordict import TensorDictBase
from tensordict import TensorClass

from torchdata.nodes import BaseNode

from .callbacks import EpochTimer, PrintLog, PassThroughScoring
from .history import History

from .optimizer import optimizer_setter
from .optimizer import GradientTransformations

from .utils import _TorchLoadUnpickler
from .utils import _identity
from .utils import _infer_predict_nonlinearity
from .utils import _check_f_arguments
from .utils import check_is_fitted
from .utils import get_map_location
from .utils import params_for
from .utils import to_device, to_tensor, to_numpy
from .utils import FirstStepAccumulator
from .utils import TeeGenerator

from .data import TabularDataset

INTMAX = 4294967295
# pylint: disable=too-many-instance-attributes
class NeuralNet(BaseEstimator):
    # pylint: disable=anomalous-backslash-in-string
    """NeuralNet base class.
    """
    prefixes_ = ['iterator_train', 'iterator_valid', 'callbacks', 'dataset', 'compile', "train_split"]

    cuda_dependent_attributes_ = []

    # This attribute keeps track of which initialization method is being used.
    # It should not be changed manually.
    init_context_ = None

    _modules = []
    _criteria = []
    _optimizers = []

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        module : Module | type[Module],
        criterion : Callable,
        optimizer : GradientTransformations | type[GradientTransformations],
        lr : float = 0.01,
        max_epochs : int = 10,
        batch_size : int = 128,
        dataset : Optional[TensorClass] = None,
        iterator_train : Optional[BaseNode] = None,
        iterator_valid : Optional[BaseNode] = None,
        train_split : float = 0.5,
        callbacks = None,
        predict_nonlinearity = 'auto',
        warm_start = False,
        verbose = 1,
        device = 'cpu',
        compile = False,
        use_caching = 'auto',
        torch_load_kwargs = None,
        **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.iterator_train = iterator_train
        self.iterator_valid = iterator_valid
        self.dataset = dataset
        self.train_split = train_split
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.compile = compile
        self.use_caching = use_caching
        self.torch_load_kwargs = torch_load_kwargs

        history = kwargs.pop('history', None)
        initialized = kwargs.pop('initialized_', False)
        virtual_params = kwargs.pop('virtual_params_', dict())

        self._params_to_validate = set(kwargs.keys())
        vars(self).update(kwargs)

        self.history_ = history
        self.initialized_ = initialized
        self.virtual_params_ = virtual_params

    @property
    def history(self):
        return self.history_

    @history.setter
    def history(self, value):
        self.history_ = value

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassThroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassThroughScoring(
                name='valid_loss',
            )),
            ('print_log', PrintLog()),
        ]

    def get_default_callbacks(self):
        return self._default_callbacks

    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in ``method_name`` with
        parameters specified in ``cb_kwargs``.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_epoch_begin
        * on_epoch_end
        * on_batch_begin
        * on_batch_end

        """
        getattr(self, method_name)(self, **cb_kwargs)
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)

    # pylint: disable=unused-argument
    def on_train_begin(self, net, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_train_end(self, net, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_epoch_begin(self, net, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_batch_begin(self, net, **kwargs):
        self.history.new_batch()

    def on_batch_end(self, net, **kwargs):
        pass

    def on_grad_computed(
            self, net, named_parameters, batch=None, training=False, **kwargs):
        pass

    def _yield_callbacks(self):
        """Yield all callbacks set on this instance including
        a set whether its name was set by the user.
        """
        print_logs = []
        for item in self.get_default_callbacks() + (self.callbacks or []):
            if isinstance(item, (tuple, list)):
                named_by_user = True
                name, cb = item
            else:
                named_by_user = False
                cb = item
                if isinstance(cb, type):  # uninitialized:
                    name = cb.__name__
                else:
                    name = cb.__class__.__name__
            if isinstance(cb, PrintLog) or (cb == PrintLog):
                print_logs.append((name, cb, named_by_user))
            else:
                yield name, cb, named_by_user
        yield from print_logs

    def _callbacks_grouped_by_name(self):
        """Group callbacks by name and collect names set by the user."""
        callbacks, names_set_by_user = OrderedDict(), set()
        for name, cb, named_by_user in self._yield_callbacks():
            if named_by_user:
                names_set_by_user.add(name)
            callbacks[name] = callbacks.get(name, []) + [cb]
        return callbacks, names_set_by_user

    def _uniquely_named_callbacks(self):
        """Make sure that the returned dict of named callbacks is unique
        w.r.t. to the callback name. User-defined names will not be
        renamed on conflict, instead an exception will be raised. The
        same goes for the event where renaming leads to a conflict.
        """
        grouped_cbs, names_set_by_user = self._callbacks_grouped_by_name()
        for name, cbs in grouped_cbs.items():
            if len(cbs) > 1 and name in names_set_by_user:
                raise ValueError("Found duplicate user-set callback name "
                                 "'{}'. Use unique names to correct this."
                                 .format(name))

            for i, cb in enumerate(cbs):
                if len(cbs) > 1:
                    unique_name = '{}_{}'.format(name, i+1)
                    if unique_name in grouped_cbs:
                        raise ValueError("Assigning new callback name failed "
                                         "since new name '{}' exists already."
                                         .format(unique_name))
                else:
                    unique_name = name
                yield unique_name, cb

    def initialize_callbacks(self):
        """Initializes all callbacks and save the result in the
        ``callbacks_`` attribute.
        """
        callbacks_ = []

        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        for name, cb in self._uniquely_named_callbacks():
            # check if callback itself is changed
            param_callback = getattr(self, 'callbacks__' + name, Dummy)
            # callback itself was set
            cb = param_callback if param_callback is not Dummy else cb
            # below: check for callback params
            # don't set a parameter for non-existing callback
            params = self.get_params_for('callbacks__{}'.format(name))
            if (cb is None): 
                if params:
                    raise ValueError("Trying to set a parameter for callback {} "
                                 "which does not exist.".format(name))
                else:
                    continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            
            cb.initialize()
            callbacks_.append((name, cb))

        # pylint: disable=attribute-defined-outside-init
        self.callbacks_ = callbacks_
        return self

    def initialized_instance(self, instance_or_cls, kwargs):
        """Return an instance initialized with the given parameters
        """
        #print(kwargs, instance_or_cls)
        is_init = callable(instance_or_cls) and not isinstance(instance_or_cls, type)
        #print(is_init, callable(instance_or_cls) , isinstance(instance_or_cls, type)
        if is_init:
            if not kwargs:
                return instance_or_cls
            else:
                return type(instance_or_cls)(**kwargs)

        return instance_or_cls(**kwargs)

    def initialize_criterion(self):
        """Initializes the criterion.
        """
        kwargs = self.get_params_for('criterion')
        criterion = self.initialized_instance(self.criterion, kwargs)
        # pylint: disable=attribute-defined-outside-init
        self.criterion_ = criterion
        return self

    def initialize_module(self):
        """Initializes the module.
        """
        kwargs = self.get_params_for('module')
        module = self.initialized_instance(self.module, kwargs)
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

    def _is_virtual_param(self, key):
        return any(fnmatch.fnmatch(key, pat) for pat in self.virtual_params_)

    def _virtual_setattr(self, param, val):
        setattr(self, param, val)

    def _register_virtual_param(self, param_patterns, fn=_virtual_setattr):
        if not isinstance(param_patterns, list):
            param_patterns = [param_patterns]
        for pattern in param_patterns:
            self.virtual_params_[pattern] = fn

    def _apply_virtual_params(self, virtual_kwargs):
        for pattern, fn in self.virtual_params_.items():
            for key, val in virtual_kwargs.items():
                if not fnmatch.fnmatch(key, pattern):
                    continue
                fn(self, key, val)

    def initialize_virtual_params(self):
        self.virtual_params_ = {}

    def initialize_optimizer(self):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.
        """
        named_parameters = self.get_all_learnable_params()
        args, kwargs = self.get_params_for_optimizer(
            'optimizer', named_parameters)

        # pylint: disable=attribute-defined-outside-init
        self.optimizer_ = self.optimizer(*args, **kwargs)
        return self

    def initialize_history(self):
        """Initializes the history."""
        if self.history_ is None:
            self.history_ = History()
        else:
            self.history_.clear()
        return self

    def _format_reinit_msg(self, name, kwargs=None, triggered_directly=True):
        """Returns a message that informs about re-initializing a compoment.
        """
        msg = "Re-initializing {}".format(name)
        if triggered_directly and kwargs:
            msg += (" because the following parameters were re-set: {}"
                    .format(', '.join(sorted(kwargs))))
        msg += "."
        return msg

    @contextmanager
    def _current_init_context(self, name):
        try:
            self.init_context_ = name
            yield
        finally:
            self.init_context_ = None

    def _initialize_virtual_params(self):
        # this init context is for consistency and not being used at the moment
        with self._current_init_context('virtual_params'):
            self.initialize_virtual_params()
            return self

    def _initialize_callbacks(self):
        # this init context is for consistency and not being used at the moment
        with self._current_init_context('callbacks'):
            if self.callbacks == "disable":
                self.callbacks_ = []
                return self
            self.initialize_callbacks()
            return self

    def _initialize_criterion(self, reason=None):
        # _initialize_criterion and _initialize_module share the same logic
        with self._current_init_context('criterion'):
            kwargs = {}
            for criterion_name in self._criteria:
                kwargs.update(self.get_params_for(criterion_name))

            has_init_criterion = any(
                isinstance(getattr(self, criterion_name + '_', None), Callable)
                for criterion_name in self._criteria)

            # check if a re-init message is required
            if kwargs or reason or has_init_criterion:
                if self.initialized_ and self.verbose:
                    if reason:
                        # re-initialization was triggered indirectly
                        msg = reason
                    else:
                        # re-initialization was triggered directly
                        msg = self._format_reinit_msg("criterion", kwargs)
                    print(msg)

            self.initialize_criterion()

            # deal with device
            for name in self._criteria:
                criterion = getattr(self, name + '_')
                if isinstance(criterion, torch.nn.Module):
                    criterion = to_device(criterion, self.device)
                    criterion = self.torch_compile(criterion, name=name)
                    setattr(self, name + '_', criterion)

            return self

    def _initialize_module(self, reason=None):
        # _initialize_criterion and _initialize_module share the same logic
        with self._current_init_context('module'):
            kwargs = {}
            for module_name in self._modules:
                kwargs.update(self.get_params_for(module_name))

            has_init_module = any(
                isinstance(getattr(self, module_name + '_', None), torch.nn.Module)
                for module_name in self._modules)

            if kwargs or reason or has_init_module:
                if self.initialized_ and self.verbose:
                    if reason:
                        # re-initialization was triggered indirectly
                        msg = reason
                    else:
                        # re-initialization was triggered directly
                        msg = self._format_reinit_msg("module", kwargs)
                    print(msg)

            self.initialize_module()

            # deal with device
            for name in self._modules:
                module = getattr(self, name + '_')
                if isinstance(module, torch.nn.Module):
                    module = to_device(module, self.device)
                    module = self.torch_compile(module, name=name)
                    setattr(self, name + '_', module)

            return self

    # pylint: disable=unused-argument
    def torch_compile(self, module, name=None):
        """Compile torch modules
        """
        if not self.compile:
            return module

        # Whether torch.compile is available (PyTorch 2.0 and up)
        torch_compile_available = hasattr(torch, 'compile')
        if not torch_compile_available:
            raise ValueError(
                "Setting compile=True but torch.compile is not available. Please "
                f"check that your installed PyTorch version ({torch.__version__}) "
                "supports torch.compile (requires v1.14, v2.0 or higher)")

        params = self.get_params_for('compile')
        module_compiled = torch.compile(module, **params)
        return module_compiled

    def get_all_learnable_params(self):
        """Yield the learnable parameters of all modules
        """
        seen = set()
        for name in self._modules + self._criteria:
            module = getattr(self, name + '_')
            named_parameters = getattr(module, 'named_parameters', None)
            if not named_parameters:
                continue

            for param_name, param in named_parameters():
                if param in seen:
                    continue

                seen.add(param)
                yield param_name, param

    def _initialize_optimizer(self, reason=None):
        with self._current_init_context('optimizer'):
            if self.initialized_ and self.verbose:
                if reason:
                    # re-initialization was triggered indirectly
                    msg = reason
                else:
                    # re-initialization was triggered directly
                    msg = self._format_reinit_msg("optimizer", triggered_directly=False)
                print(msg)

            self.initialize_optimizer()

            # register the virtual params for all optimizers
            for name in self._optimizers:
                param_pattern = [name + '__param_groups__*__*', name + '__*']
                if name == 'optimizer':  # 'lr' is short for optimizer__lr
                    param_pattern.append('lr')
                setter = partial(
                    optimizer_setter,
                    optimizer_attr=name + '_',
                    optimizer_name=name,
                )
                self._register_virtual_param(param_pattern, setter)
            return self

    def _initialize_history(self):
        # this init context is for consistency and not being used at the moment
        with self._current_init_context('history'):
            self.initialize_history()
            return self

    def _initialize_random_state(self):
        if self.seed is not None:
            self.np_random_state = np.random.default_rng(seed=self.seed)
            #2**32-1, the maximum value of 32-bit int used by the leagcy mt1997 generators used by np.random.RandomState
            #torch_seed = self.np_random_state.integers(INTMAX, size=1).tolist()
            #disable_cudnn = True if "cuda" in self.device else False
            #set_global_random_seed(torch_seed[0], make_cuda_deterministic=disable_cudnn)

    def initialize(self):
        """Initializes all of its components and returns self."""
        self.check_training_readiness()
        
        if not self.initialized_:
            self._initialize_random_state()
            
        self._initialize_virtual_params()
        self._initialize_callbacks()
        self._initialize_module()
        self._initialize_criterion()
        self._initialize_optimizer()
        self._initialize_history()

        self._validate_params()

        self.initialized_ = True
        return self

    def check_training_readiness(self):
        """Check that the net is ready to train"""
        is_trimmed_for_prediction = getattr(self, '_trimmed_for_prediction', False)
        if is_trimmed_for_prediction:
            msg = (
                "The net's attributes were trimmed for prediction, thus it cannot "
                "be used for training anymore"
            )
            raise Exception(msg)

    def _set_training(self, training=True):
        """Set training/evaluation mode on all modules and criteria that are torch modules.
        """
        for module_name in self._modules + self._criteria:
            module = getattr(self, module_name + '_')
            if isinstance(module, torch.nn.Module):
                module.train(training)

    def check_is_fitted(self, attributes=None, *args, **kwargs):
        """Checks whether the net is initialized
        """
        attributes = (
            attributes or [module + '_' for module in self._modules] or ['module_']
        )
        check_is_fitted(self, attributes, *args, **kwargs)

    def trim_for_prediction(self):
        """Remove all attributes not required for prediction.
        """
        # pylint: disable=protected-access
        if getattr(self, '_trimmed_for_prediction', False):
            return

        self.check_is_fitted()
        # pylint: disable=attribute-defined-outside-init
        self._trimmed_for_prediction = True
        self._set_training(False)

        if isinstance(self.callbacks, list):
            self.callbacks.clear()
        self.callbacks_.clear()

        self.train_split = None
        self.iterator_train = None
        self.history.clear()

        attrs_to_trim = self._optimizers[:] + self._criteria[:]

        for name in attrs_to_trim:
            setattr(self, name + '_', None)
            if hasattr(self, name):
                setattr(self, name, None)


    def fit(self, X, y=None, sample_weights=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y=y, sample_weights=sample_weights, **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self

    def fit_loop(self, X, y=None, sample_weights=None, epochs=None, **fit_params):
        #self.check_data(X, y, sample_weights)
        self.check_training_readiness()
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(X, y=y, sample_weights=sample_weights, **fit_params)
        print("total size: ", len(dataset_train.dataset), 
              ", train size: ", len(dataset_train), 
              ", valid size: ", len(dataset_valid)
              )

        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = self.get_iterator(dataset_valid, training=False) if dataset_valid is not None else None

        #dataset = self.get_dataset(X, y=y, sample_weights=sample_weights)
        #iterator_train, iterator_valid = self.get_split_dataloaders(dataset)

        for iepoch in range(epochs):
            self.notify("on_epoch_begin", **on_epoch_kwargs)

            #print("epoch: ",  iepoch)

            self.run_single_epoch(iterator_train, training = True, prefix="train", step_fn=self._train_step, **fit_params)
            self.run_single_epoch(iterator_valid, training = False, prefix="valid", step_fn=self._validation_step, **fit_params)
            
            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self

    def run_single_epoch(self, iterator, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation."""

        #print(f"running training {training}" )

        if iterator is None:
            return

        batch_count = 0 
        for batch in iterator:
            #if batch_count > 1:
            #    break
            #worker_info
            #print(f"Seed for worker#{torch.utils.data.get_worker_info().id}(): {torch.initial_seed()}")
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (len(batch[0]) if isinstance(batch, (tuple, list))
                          else len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def _validation_step(self, batch, **fit_params):
        return self.validation_step(batch, **fit_params)

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        X, y, sample_weights, _ = batch.unpack_data()
        with torch.no_grad():
            y_pred = self.infer(X, **fit_params)
            loss = self.get_loss(y_pred, y, sample_weights=sample_weights, X=X, training=True)
        return {"loss" : loss, "y_pred" : y_pred}

    def _train_step(self, batch, **fit_params):
        step_accumulator = self.get_first_step_accumulator()

        def step_fn():
            self._zero_grad_optimizer()
            step = self.train_step(batch, **fit_params)
            step_accumulator.store_step(step)

            self.notify("on_grad_computed", named_parameters=TeeGenerator(self.get_all_learnable_params()), batch=batch, training=True)

            return step["loss"]
        
        self._step_optimizer(step_fn)
        return step_accumulator.get_step()

    def train_step(self, batch : Batch, **fit_params):
        self._set_training(True)
        X, y, sample_weights, _ = batch.unpack_data()
        #X, y, sample_weights, indices = batch.unpack_data()
        #print(indices[0:10])
        y_pred = self.infer(X, **fit_params)
        loss = self.get_loss(y_pred, y, sample_weights=sample_weights, X=X, training=True)
        loss.backward()
        return {"loss" : loss, "y_pred" : y_pred}

    def _zero_grad_optimizer(self, set_to_none=None):
        """Zero out the gradient of all optimizers.
        """
        for name in self._optimizers:
            optimizer = getattr(self, name + '_')
            if set_to_none is None:
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=set_to_none)

    def _step_optimizer(self, step_fn):
        """Perform a ``step`` call on all optimizers.
        """
        for name in self._optimizers:
            optimizer = getattr(self, name + '_')
            if step_fn is None:
                optimizer.step()
            else:
                optimizer.step(step_fn)

    def get_first_step_accumulator(self):
        return FirstStepAccumulator()

    # pylint: disable=unused-argument
    def get_loss(self, 
                 y_pred : torch.Tensor, 
                 y_true : torch.Tensor, 
                 sample_weights : Optional[torch.Tensor]=None, 
                 **kwargs):
        """Return the loss for this batch.
        """
        y_true = to_tensor(y_true, device=self.device)
        sample_weights = to_tensor(sample_weights, device=self.device)
        losses = self.criterion_(y_pred, y_true)
        if getattr(self, "criterion__reduction", None) == "none":
            if sample_weights is not None:
                n_samples = y_true.shape[0]
                return torch.dot(sample_weights.squeeze_(), losses.squeeze_()).div_(n_samples)
            return torch.mean(losses)
        return losses


    def predict(self, X, training=False):
        return self.predict_proba(X, training=training)
   
    def predict_proba(self, X, training=False):
        """Return the output of the module's forward method as a numpy
        array.
        """
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=training):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def forward_iter(self, X, training=False, device='cpu'):
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, training=training)
            yield to_device(yp, device=device)

    def evaluation_step(self, batch : Batch, training : bool = False, grad_enabled : bool = False) -> torch.Tensor | tuple[torch.Tensor]:
        self.check_is_fitted()
        X, _, _, _ = batch.unpack_data()

        with torch.set_grad_enabled(grad_enabled):
            self._set_training(training)
            return self.infer(X)

    def infer(self, x, **kwargs):
        dtype = kwargs.pop("to_dtype", torch.float32)
        x_tensor = to_tensor(x, device=self.device, dtype=dtype)
        if isinstance(x_tensor, Mapping):
            kwargs.update(x_tensor)
            return self.module_(**kwargs)
        if isinstance(x_tensor, (list, tuple)):
            return self.module_(*x_tensor, **kwargs)
        
        return self.module_(x_tensor, **kwargs)

    def _get_predict_nonlinearity(self):
        """Return the nonlinearity to be applied to the prediction
        """
        self.check_is_fitted()
        nonlin = self.predict_nonlinearity
        if nonlin is None:
            nonlin = _identity
        elif nonlin == 'auto':
            nonlin = _infer_predict_nonlinearity(self)
        if not callable(nonlin):
            raise TypeError("predict_nonlinearity has to be a callable, 'auto' or None")
        return nonlin
 
    def get_iterator(self, dataset : Dataset, training=False):
        """Get an iterator that allows to loop over the batches of the
        given data.
        """
        if training:
            kwargs = self.get_params_for('iterator_train')
            iterator = self.iterator_train or DataLoader
        else:
            kwargs = self.get_params_for('iterator_valid')
            iterator = self.iterator_valid or DataLoader

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        if kwargs['batch_size'] == -1:
            if isinstance(dataset, Sized):
                kwargs['batch_size'] = len(dataset)
            else:
                raise ValueError("Can't pass no batch_size (default = -1) for not Map-style dataset!")

        if "num_workers" in kwargs:
            if kwargs["num_workers"] > 0 and hasattr(self, "np_random_state"):
                seed : int = self.np_random_state.integers(INTMAX, size=1).tolist()[0]
                torch_generator = torch.Generator()
                torch_generator.manual_seed(seed)
                kwargs["generator"] = torch_generator
                if "worker_init_fn" not in kwargs:
                    kwargs["worker_init_fn"] = partial(seed_worker, make_cuda_deterministic="cuda" in self.device)

        return iterator(dataset, **kwargs)

    def get_split_datasets(self, X, y=None, sample_weights=None, **kwargs) -> tuple[Subset | Dataset, Optional[Subset]]:
        dataset = self.get_dataset(X, y=y, sample_weights=sample_weights)

        if not self.train_split:
            print("No split given...")
            return dataset, None

        if callable(self.train_split) and not isinstance(self.train_split, type):
            print("Given predifined split...")
            return self.train_split(dataset, y=y)

        print("Given ValidSplit instance or Class...")
            
        random_state=None
        if hasattr(self, "np_random_state"):
            random_state = self.np_random_state.integers(INTMAX, size=1).tolist()[0]
            #print(f"Splitter random_state will be set to {random_state}")

        kwargs = self.get_params_for("train_split")
        if "random_state" not in kwargs:
                kwargs["random_state"] = random_state

        if isinstance(self.train_split, (float, int)):
            kwargs["cv"] = self.train_split
            splitter = ValidSplit(**kwargs)
            return splitter(dataset, y=y)

        if isinstance(self.train_split, type):
            splitter = self.train_split(**kwargs)
            return splitter(dataset, y=y)

        return dataset, None

    def get_dataset(self, X, y=None, sample_weights=None, **kwargs):
        """Get a dataset that contains the input data and is passed to
        the iterator.
        """
        if isinstance(X, Dataset):
            return X

        dataset = self.dataset

        is_initialized = not callable(dataset)

        ds_kwargs = self.get_params_for('dataset')
        ds_kwargs.update(kwargs)
        if ds_kwargs and is_initialized:
            raise TypeError("Trying to pass an initialized Dataset while "
                            "passing Dataset arguments ({}) is not "
                            "allowed.".format(kwargs))

        if is_initialized:
            return dataset

        if issubclass(dataset, TorchDataset):
            return dataset(X, targets=y, sample_weights=sample_weights, **ds_kwargs)

        raise TypeError(f"Can only handle datasets inheriting from datasets.TorchDataset provided in this package. Can't handle datasets of type {dataset}")

    def _get_params_for(self, prefix):
        return params_for(prefix, self.__dict__)

    def get_params_for(self, prefix):
        """Collect and return init parameters for an attribute.
        """
        return self._get_params_for(prefix)

    def _get_params_for_optimizer(self, prefix, named_parameters):
        kwargs = self.get_params_for(prefix)
        params = list(named_parameters)
        pgroups = []

        for pattern, group in kwargs.pop('param_groups', []):
            matches = [i for i, (name, _) in enumerate(params) if
                       fnmatch.fnmatch(name, pattern)]
            if matches:
                p = [params.pop(i)[1] for i in reversed(matches)]
                pgroups.append({'params': p, **group})

        if params:
            pgroups.append({'params': [p for _, p in params]})

        args = (pgroups,)

        # 'lr' is an optimizer param that can be set without the 'optimizer__'
        # prefix because it's so common
        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr
        return args, kwargs

    def get_params_for_optimizer(self, prefix, named_parameters):
        """Collect and return init parameters for an optimizer.
        """
        args, kwargs = self._get_params_for_optimizer(prefix, named_parameters)
        return args, kwargs
    
    def _get_param_names(self):
        return [k for k in self.__dict__ if not k.endswith('_')]

    def _get_params_callbacks(self, deep=True):
        """sklearn's .get_params checks for `hasattr(value,
        'get_params')`. This returns False for a list. But our
        callbacks reside within a list. Hence their parameters have to
        be retrieved separately.
        """
        params = {}
        if not deep:
            return params

        callbacks_ = getattr(self, 'callbacks_', [])
        for key, val in callbacks_:
            name = 'callbacks__' + key
            params[name] = val
            if val is None:  # callback deactivated
                continue
            for subkey, subval in val.get_params().items():
                subname = name + '__' + subkey
                params[subname] = subval
        return params

    def get_params(self, deep=True, **kwargs):
        params = super().get_params(deep=deep, **kwargs)
        # Callback parameters are not returned by .get_params, needs
        # special treatment.
        params_cb = self._get_params_callbacks(deep=deep)
        params.update(params_cb)

        # don't include the following attributes
        to_exclude = {'_modules', '_criteria', '_optimizers'}
        return {key: val for key, val in params.items() if key not in to_exclude}

    def _validate_params(self):
        """Check argument names passed at initialization.
        """
        # warn about usage of iterator_valid__shuffle=True, since this
        # is almost certainly not what the user wants
        if 'iterator_valid__shuffle' in self._params_to_validate:
            if hasattr(self, "iterator_valid__shuffle"):
                warnings.warn(
                    "You set iterator_valid__shuffle=True; this is most likely not "
                    "what you want because the values returned by predict and "
                    "predict_proba will be shuffled.",
                    UserWarning)

        # check for wrong arguments
        unexpected_kwargs = []
        missing_dunder_kwargs = []
        for key in sorted(self._params_to_validate):
            if key.endswith('_'):
                continue

            # see https://github.com/skorch-dev/skorch/pull/590 for
            # why this must be sorted
            for prefix in sorted(self.prefixes_, key=lambda s: (-len(s), s)):
                if key == prefix:
                    # e.g. someone did net.set_params(callbacks=[])
                    break
                if key.startswith(prefix):
                    if not key.startswith(prefix + '__'):
                        missing_dunder_kwargs.append((prefix, key))
                    break
            else:  # no break means key didn't match a prefix
                unexpected_kwargs.append(key)

        msgs = []
        if unexpected_kwargs:
            tmpl = ("__init__() got unexpected argument(s) {}. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
            msg = tmpl.format(', '.join(sorted(unexpected_kwargs)))
            msgs.append(msg)

        for prefix, key in sorted(missing_dunder_kwargs, key=lambda tup: tup[1]):
            tmpl = "Got an unexpected argument {}, did you mean {}?"
            suffix = key[len(prefix):].lstrip('_')
            suggestion = prefix + '__' + suffix
            msgs.append(tmpl.format(key, suggestion))

        valid_vals_use_caching = ('auto', False, True)
        if self.use_caching not in valid_vals_use_caching:
            msgs.append(
                f"Incorrect value for use_caching used ('{self.use_caching}'), "
                f"use one of: {', '.join(map(str, valid_vals_use_caching))}"
            )

        if msgs:
            full_msg = '\n'.join(msgs)
            raise ValueError(full_msg)

    def set_params(self, **kwargs):
        """Set the parameters of this class.
        """
        normal_params, cb_params, special_params = {}, {}, {}
        virtual_params = {}

        for key, val in kwargs.items():
            if self._is_virtual_param(key):
                virtual_params[key] = val
            elif key.startswith('callbacks'):
                cb_params[key] = val
                self._params_to_validate.add(key)
            elif any(key.startswith(prefix) for prefix in self.prefixes_):
                special_params[key] = val
                self._params_to_validate.add(key)
            elif '__' in key:
                special_params[key] = val
                self._params_to_validate.add(key)
            else:
                normal_params[key] = val

        self._apply_virtual_params(virtual_params)
        super().set_params(**normal_params)

        for key, val in special_params.items():
            if key.endswith('_'):
                raise ValueError(
                    "Something went wrong here. Please open an issue on "
                    "https://github.com/skorch-dev/skorch/issues detailing what "
                    "caused this error.")
            setattr(self, key, val)

        if cb_params:
            # callbacks need special treatmeant since they are list of tuples
            self._initialize_callbacks()
            self._set_params_callback(**cb_params)
            vars(self).update(cb_params)

        # If the net is not initialized or there are no special params, we can
        # exit as this point, because the special_params have been set as
        # attributes and will be applied by initialize() at a later point in
        # time.
        if not self.initialized_ or not special_params:
            return self

        # if net is initialized, checking kwargs is possible
        self._validate_params()

        ######################################################
        # Below: Re-initialize parts of the net if necessary #
        ######################################################

        # if there are module params, reinit module, criterion, optimizer
        # if there are criterion params, reinit criterion, optimizer
        # optimizer params don't need to be checked, as they are virtual
        reinit_module = False
        reinit_criterion = False
        reinit_optimizer = False

        msg_module : str = ""
        msg_criterion : str = ""
        msg_optimizer : str = ""

        component_names = {key.split('__', 1)[0] for key in special_params}
        for prefix in component_names:
            if (prefix in self._modules) or (prefix == 'compile'):
                reinit_module = True
                reinit_criterion = True
                reinit_optimizer = True

                module_params = {k: v for k, v in special_params.items()
                                 if k.startswith(prefix)}
                msg_module = self._format_reinit_msg(
                    "module", module_params, triggered_directly=True)
                msg_criterion = self._format_reinit_msg(
                    "criterion", triggered_directly=False)
                msg_optimizer = self._format_reinit_msg(
                    "optimizer", triggered_directly=False)

                # if any module is modified, everything needs to be
                # re-initialized, no need to check any further
                break

            if prefix in self._criteria:
                reinit_criterion = True
                reinit_optimizer = True

                criterion_params = {k: v for k, v in special_params.items()
                                    if k.startswith(prefix)}
                msg_criterion = self._format_reinit_msg(
                    "criterion", criterion_params, triggered_directly=True)
                msg_optimizer = self._format_reinit_msg(
                    "optimizer", triggered_directly=False)

        if not (reinit_module or reinit_criterion or reinit_optimizer):
            raise ValueError("Something went wrong!")

        if reinit_module:
            self._initialize_module(reason=msg_module)
        if reinit_criterion:
            self._initialize_criterion(reason=msg_criterion)
        if reinit_optimizer:
            self._initialize_optimizer(reason=msg_optimizer)

        return self

    def _set_params_callback(self, **params):
        """Special handling for setting params on callbacks."""
        # model after sklearn.utils._BaseCompostion._set_params
        # 1. All steps
        if 'callbacks' in params:
            setattr(self, 'callbacks', params.pop('callbacks'))

        # 2. Step replacement
        names, _ = zip(*getattr(self, 'callbacks_'))
        for key in params.copy():
            name = key[11:]  # drop 'callbacks__'
            if '__' not in name and name in names:
                self._replace_callback(name, params.pop(key))

        # 3. Step parameters and other initilisation arguments
        for key in params.copy():
            name = key[11:]
            part0, part1 = name.split('__')
            kwarg = {part1: params.pop(key)}
            callback = dict(self.callbacks_).get(part0)
            if callback is not None:
                callback.set_params(**kwarg)
            else:
                raise ValueError(
                    "Trying to set a parameter for callback {} "
                    "which does not exist.".format(part0))

        return self

    def _replace_callback(self, name, new_val):
        # assumes `name` is a valid callback name
        callbacks_new = self.callbacks_[:]
        for i, (cb_name, _) in enumerate(callbacks_new):
            if cb_name == name:
                callbacks_new[i] = (name, new_val)
                break
        setattr(self, 'callbacks_', callbacks_new)

    def __getstate__(self):
        state = self.__dict__.copy()
        cuda_attrs = {}
        for prefix in self.cuda_dependent_attributes_:
            for key in state:
                if isinstance(key, str) and key.startswith(prefix):
                    cuda_attrs[key] = state[key]

        for k in cuda_attrs:
            state.pop(k)

        with tempfile.SpooledTemporaryFile() as f:
            pickle.dump(cuda_attrs, f)
            f.seek(0)
            state['__cuda_dependent_attributes__'] = f.read()

        return state

    def __setstate__(self, state):
        # get_map_location will automatically choose the
        # right device in cases where CUDA is not available.
        map_location = get_map_location(state['device'])
        load_kwargs = {'map_location': map_location}
        state['device'] = self._check_device(state['device'], map_location)
        torch_load_kwargs = state.get('torch_load_kwargs') or {"weights_only" : False}

        with tempfile.SpooledTemporaryFile() as f:
            unpickler = _TorchLoadUnpickler(
                f,
                map_location=map_location,
                torch_load_kwargs=torch_load_kwargs,
            )
            f.write(state['__cuda_dependent_attributes__'])
            f.seek(0)
            cuda_attrs = unpickler.load()
           # try:
           #     cuda_attrs = unpickler.load()
           # except pickle.UnpicklingError:
           #     f.seek(0)
           #     cuda_attrs = torch.load(f, **load_kwargs)

        state.update(cuda_attrs)
        state.pop('__cuda_dependent_attributes__')

        self.__dict__.update(state)

    def _register_attribute(
            self,
            name,
            attr,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """Add attribute name to prefixes_ and cuda_dependent_attributes_,
        keep track of modules, criteria, and optimizers.
        """
        name = name.rstrip('_')  # module_ -> module

        # Always copy the collections to avoid mutation
        if prefixes:
            self.prefixes_ = self.prefixes_[:] + [name]

        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = (
                self.cuda_dependent_attributes_[:] + [name + '_'])

        # make sure to not double register -- this should never happen, but
        # still better to check
        if (self.init_context_ == 'module') and (name not in self._modules):
            self._modules = self._modules[:] + [name]
        elif (self.init_context_ == 'criterion') and (name not in self._criteria):
            self._criteria = self._criteria[:] + [name]
        elif (self.init_context_ == 'optimizer') and (name not in self._optimizers):
            self._optimizers = self._optimizers[:] + [name]

    def _unregister_attribute(
            self,
            name,
            prefixes=True,
            cuda_dependent_attributes=True,
    ):
        """Remove attribute name from prefixes_, cuda_dependent_attributes_ and the
        _modules/_criteria/_optimizers list if applicable.
        """
        name = name.rstrip('_')  # module_ -> module

        # copy the lists to avoid mutation
        if prefixes:
            self.prefixes_ = [p for p in self.prefixes_ if p != name]

        if cuda_dependent_attributes:
            self.cuda_dependent_attributes_ = [
                a for a in self.cuda_dependent_attributes_ if a != name + '_']

        if name in self._modules:
            self._modules = [p for p in self._modules if p != name]
        if name in self._criteria:
            self._criteria = [p for p in self._criteria if p != name]
        if name in self._optimizers:
            self._optimizers = [p for p in self._optimizers if p != name]

    def _check_settable_attr(self, name, attr):
        """Check whether this attribute is valid for it to be settable.

        E.g. for it to work with set_params.

        """
        if (self.init_context_ is None) and isinstance(attr, torch.nn.Module):
            msg = ("Trying to set torch compoment '{}' outside of an initialize method."
                   " Consider defining it inside 'initialize_module'".format(name))
            raise Exception(msg)

        if (self.init_context_ is None) and isinstance(attr, torch.optim.Optimizer):
            msg = ("Trying to set torch compoment '{}' outside of an initialize method."
                   " Consider defining it inside 'initialize_optimizer'".format(name))
            raise Exception(msg)

        if not name.endswith('_'):
            msg = ("Names of initialized modules or optimizers should end "
                   "with an underscore (e.g. '{}_')".format(name))
            raise Exception(msg)

    def __setattr__(self, name, attr):
        """Set an attribute on the net
        """
        is_known = name in self.prefixes_ or name.rstrip('_') in self.prefixes_
        is_special_param = '__' in name
        first_init = not hasattr(self, 'initialized_')
        is_torch_component = isinstance(attr, (torch.nn.Module, torch.optim.Optimizer))

        if not (is_known or is_special_param or first_init) and is_torch_component:
            self._check_settable_attr(name, attr)
            self._register_attribute(name, attr)
        super().__setattr__(name, attr)

    def __delattr__(self, name):
        # take extra precautions to undo the changes made in __setattr__
        self._unregister_attribute(name)
        super().__delattr__(name)

    def _get_module(self, name, msg):
        """Return the PyTorch module with the given name
        """
        try:
            module = getattr(self, name)
            if not hasattr(module, 'state_dict'):
                raise AttributeError
            return module
        except AttributeError:
            if name.endswith('_'):
                name = name[:-1]
            raise AttributeError(msg.format(name=name))

    def save_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            **kwargs):
        """Saves the module's parameters, history, and optimizer,
        not the whole object.
        """
        kwargs_module, kwargs_other = _check_f_arguments(
            'save_params',
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_criterion=f_criterion,
            f_history=f_history,
            **kwargs)

        if not kwargs_module and not kwargs_other:
            if self.verbose:
                print("Nothing to save")
            return

        msg_init = (
            "Cannot save state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...).")
        msg_module = (
            "You are trying to save 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module or Optimizer; make sure that it exists and check for typos.")

        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr.endswith('_') and not self.initialized_:
                self.check_is_fitted([attr], msg=msg_init)
            module = self._get_module(attr, msg=msg_module)
            torch.save(module.state_dict(), f_name)

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history.to_file(f_history)

    def _check_device(self, requested_device, map_device):
        """Compare the requested device with the map device and
        return the map device if it differs from the requested device
        along with a warning.
        """
        if requested_device is None:
            # user has set net.device=None, we don't know the type, use fallback
            msg = (
                f"Setting self.device = {map_device} since the requested device "
                f"was not specified"
            )
            warnings.warn(msg, UserWarning)
            return map_device

        type_1 = torch.device(requested_device)
        type_2 = torch.device(map_device)
        if type_1 != type_2:
            warnings.warn(
                'Setting self.device = {} since the requested device ({}) '
                'is not available.'.format(map_device, requested_device),
                UserWarning)
            return map_device
        # return requested_device instead of map_device even though we
        # checked for *type* equality as we might have 'cuda:0' vs. 'cuda:1'.
        return requested_device

    def load_params(
            self,
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            checkpoint=None,
            **kwargs):
        """Loads the the module's parameters, history, and optimizer,
        not the whole object.
        """
        torch_load_kwargs = self.torch_load_kwargs
        if torch_load_kwargs is None:
            torch_load_kwargs = {"weights_only" : False}

        def _get_state_dict(f_name):
            map_location = get_map_location(self.device)
            self.device = self._check_device(self.device, map_location)
            return torch.load(f_name, map_location=map_location, **torch_load_kwargs)

        kwargs_full = {}
        if checkpoint is not None:
            if not self.initialized_:
                self.initialize()
            if f_history is None and checkpoint.f_history is not None:
                self.history = History.from_file(checkpoint.f_history_)
            kwargs_full.update(**checkpoint.get_formatted_files(self))

        # explicit arguments may override checkpoint arguments
        kwargs_full.update(**kwargs)
        for key, val in [('f_params', f_params), ('f_optimizer', f_optimizer),
                         ('f_criterion', f_criterion), ('f_history', f_history)]:
            if val:
                kwargs_full[key] = val

        kwargs_module, kwargs_other = _check_f_arguments('load_params', **kwargs_full)

        if not kwargs_module and not kwargs_other:
            if self.verbose:
                print("Nothing to load")
            return

        # only valid key in kwargs_other is f_history
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            self.history = History.from_file(f_history)

        msg_init = (
            "Cannot load state of an un-initialized model. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...).")
        msg_module = (
            "You are trying to load 'f_{name}' but for that to work, the net "
            "needs to have an attribute called 'net.{name}_' that is a PyTorch "
            "Module or Optimizer; make sure that it exists and check for typos.")

        for attr, f_name in kwargs_module.items():
            # valid attrs can be 'module_', 'optimizer_', etc.
            if attr.endswith('_') and not self.initialized_:
                self.check_is_fitted([attr], msg=msg_init)
            module = self._get_module(attr, msg=msg_module)
            state_dict = _get_state_dict(f_name)
            module.load_state_dict(state_dict)

    def __repr__(self, N_CHAR_MAX=700):
        to_include = ['module']
        to_exclude = []
        parts = [str(self.__class__) + '[uninitialized](']
        if self.initialized_:
            parts = [str(self.__class__) + '[initialized](']
            to_include = ['module_']
            to_exclude = ['module__']

        for key, val in sorted(self.__dict__.items()):
            if not any(key.startswith(prefix) for prefix in to_include):
                continue
            if any(key.startswith(prefix) for prefix in to_exclude):
                continue

            val = str(val)
            if '\n' in val:
                val = '\n  '.join(val.split('\n'))
            parts.append('  {}={},'.format(key, val))

        parts.append(')')
        return '\n'.join(parts)
