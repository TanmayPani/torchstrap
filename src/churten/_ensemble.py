from functools import partial

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from torch.utils.data import SequentialSampler
from torch.utils.data import SubsetRandomSampler

from tensordict import TensorDictBase

from churten.data.sampler import undersample

from .history import History

from .callbacks import Callback
from .callbacks import EpochTimer
from .callbacks import PrintLog
from .callbacks import PassThroughScoring

from .optimizer import GradientTransformations

from .data import TabularDataset
from .data import train_test_multi_subset_samplers

from .utils import _infer_predict_nonlinearity
from .utils import get_map_location
from .utils import params_for

class StackedNeuralNetEnsemble(BaseEstimator):
    def __init__(
        self,
        module,
        optimizer,
        criterion,
        num_module_replicas = 1,
        num_dataset_replicas = 1,
        *,
        lr = 0.0001,
        max_epochs = 10,
        dataset = TabularDataset,
        samplers_fn = train_test_multi_subset_samplers,
        batch_size = 128,
        predict_nonlinearity = None,
        callbacks = {},
        device = "cpu",
        compile = False,
        **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_module_replicas = num_module_replicas
        self.num_dataset_replicas = num_dataset_replicas
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.samplers_fn = samplers_fn 
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.device = device
        self.compile = compile

        history = kwargs.pop('history', None)
        initialized = kwargs.pop('initialized_', False)

        vars(self).update(kwargs)

        self.history_ = history
        self.initialized_ = initialized

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

    def _initialize_callbacks(self):
        """Initializes all callbacks and save the result in the
        ``callbacks_`` attribute.
        """
        self.callbacks_ = []

        class Dummy:
            # We cannot use None as dummy value since None is a
            # legitimate value to be set.
            pass

        for name, cb in self.callbacks.items():
            # check if callback itself is changed
            param_callback = getattr(self, 'callbacks__' + name, Dummy)
            # callback itself was set
            cb = param_callback if param_callback is not Dummy else cb
            # below: check for callback params
            # don't set a parameter for non-existing callback
            params = self.get_params_for(f'callbacks__{name}')
           
            if (cb is None): 
                if params:
                    raise ValueError(f"Trying to set a parameter for callback {name}"
                                 "which does not exist.")
                else:
                    continue

            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            
            cb.initialize()
            self.callbacks_.append((name, cb))

        # pylint: disable=attribute-defined-outside-init
        return self

    def _initialize_module(self):
        kwargs = self.get_params_for("module")

        self.num_total_module_replicas = self.num_module_replicas*self.num_dataset_replicas

        for key, val in kwargs.items():
            if not isinstance(val, list):
                kwargs[key] = [val]


