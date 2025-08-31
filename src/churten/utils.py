from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from functools import partial
import io
from itertools import tee
import pathlib
import pickle
import warnings
import random
import os

import numpy as np
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.utils import _safe_indexing as safe_indexing
from sklearn.utils.validation import check_is_fitted as sk_check_is_fitted
from sklearn.model_selection import train_test_split

import torch
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import PackedSequence
#from torch.utils.data.dataset import Subset

class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'

def set_global_random_seed(seed : int, make_cuda_deterministic=False, verbose=True):
    if verbose:
        print (f"Setting pytorch, numpy and python global seeds to: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if make_cuda_deterministic:
        set_cudnn_deterministic(verbose=verbose)

def set_cudnn_deterministic(verbose=True):
    if verbose:
        print("Disabling use of non-deterministic methods by cuDNN.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

def params_for(prefix, kwargs):
    """Extract parameters that belong to a given sklearn module prefix from
    """
    if not prefix.endswith('__'):
        prefix += '__'
    return {key[len(prefix):]: val for key, val in kwargs.items()
            if key.startswith(prefix)}

def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, PackedSequence))

def to_device(X, device, dtype=None):
    """Generic function to modify the device type of the tensor(s) or module.
    """
    if (device, dtype) == (None, None):
        return X

    if isinstance(X, Mapping):
        # dict-like but not a dict
        return type(X)(**{key: to_device(val, device) for key, val in X.items()})

    # PackedSequence class inherits from a namedtuple
    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)

    if isinstance(X, torch.distributions.distribution.Distribution):
        return X

    return X.to(device=device, dtype=dtype)

# pylint: disable=unused-argument
def noop(*args, **kwargs):
    """No-op function that does nothing and returns ``None``.

    This is useful for defining scoring callbacks that do not need a
    target extractor.
    """

# pylint: disable=not-callable
#def to_tensor(X, device, accept_sparse=False):
def to_tensor(X, device, dtype=None, seq_to_tensor=False):
    """Turn input data to torch tensor.
    """
    to_tensor_ = partial(to_tensor, device=device, dtype=dtype, seq_to_tensor=seq_to_tensor)

    if is_torch_data_type(X):
        return to_device(X, device, dtype=dtype)
    #if TORCH_GEOMETRIC_INSTALLED and is_geometric_data_type(X):
    #    return to_device(X, device)
    #if hasattr(X, 'convert_to_tensors'):
        # huggingface transformers BatchEncoding
    #    return X.convert_to_tensors('pt')
    if isinstance(X, Sequence):
        if any(isinstance(x, Sequence) for x in X):
            if seq_to_tensor:
                return 
        return torch.as_tensor(np.array(X), device=device, dtype=dtype)

    if isinstance(X, Mapping):
        return {key: to_tensor_(val) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    if np.isscalar(X):
        return torch.as_tensor(X, device=device, dtype=dtype)
    if isinstance(X, np.ndarray):
        return torch.as_tensor(X, device=device, dtype=dtype)
    #if sparse.issparse(X):
    #   if accept_sparse:
    #       return torch.sparse_coo_tensor(
    #           X.nonzero(), X.data, size=X.shape).to(device)
    #   raise TypeError("Sparse matrices are not supported. Set "
    #                   "accept_sparse=True to allow sparse matrices.")
    raise TypeError(f"Cannot convert this data of type {type(X)} to a torch tensor.")
def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy.
    """
    if isinstance(X, np.ndarray):
        return X

    if isinstance(X, Mapping):
        return {key: to_numpy(val) for key, val in X.items()}

    #if is_pandas_ndframe(X):
    #    return X.values

    if isinstance(X, (tuple, list)):
        return type(X)(to_numpy(x) for x in X)

    #if _is_slicedataset(X):
    #    return np.asarray(X)

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if hasattr(X, 'is_mps') and X.is_mps:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()

def get_map_location(target_device, fallback_device='cpu'):
    """Determine the location to map loaded data (e.g., weights)
    for a given target device (e.g. 'cuda').
    """
    if target_device is None:
        target_device = fallback_device

    map_location = torch.device(target_device)

    # The user wants to use CUDA but there is no CUDA device
    # available, thus fall back to CPU.
    if map_location.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn(
            'Requested to load data to CUDA but no CUDA devices '
            'are available. Loading on device "{}" instead.'.format(
                fallback_device,
            ), UserWarning)
        map_location = torch.device(fallback_device)
    return map_location


def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """Checks whether the net is initialized.

    Note: This calls ``sklearn.utils.validation.check_is_fitted``
    under the hood, using exactly the same arguments and logic. The
    only difference is that this function has an adapted error message
    and raises a ``skorch.exception.NotInitializedError`` instead of
    an ``sklearn.exceptions.NotFittedError``.

    """
    try:
        sk_check_is_fitted(estimator, attributes, msg=msg, all_or_any=all_or_any)
    except NotFittedError as exc:
        if msg is None:
            msg = ("This %(name)s instance is not initialized yet. Call "
                   "'initialize' or 'fit' with appropriate arguments "
                   "before using this method.")

        raise Exception(msg % {'name': type(estimator).__name__}) from exc


def _identity(x):
    """Return input as is, the identity operation"""
    return x

def _make_2d_probs(prob):
    """Create a 2d probability array from a 1d vector

    This is needed because by convention, even for binary classification
    problems, sklearn expects 2 probabilities to be returned per row, one for
    class 0 and one for class 1.

    """
    y_proba = torch.stack((1 - prob, prob), 1)
    return y_proba


def _sigmoid_then_2d(x):
    """Transform 1-dim logits to valid y_proba

    Sigmoid is applied to x to transform it to probabilities. Then
    concatenate the probabilities with 1 - these probabilities to
    return a correctly formed ``y_proba``. This is required for
    sklearn, which expects probabilities to be 2d arrays whose sum
    along axis 1 is 1.0.

    Parameters
    ----------
    x : torch.tensor
      A 1 dimensional float torch tensor containing raw logits.

    Returns
    -------
    y_proba : torch.tensor
      A 2 dimensional float tensor of probabilities that sum up to 1
      on axis 1.

    """
    prob = torch.sigmoid(x)
    return _make_2d_probs(prob)


def _infer_predict_nonlinearity(net):
    """Infers the correct nonlinearity to apply for this net

    The nonlinearity is applied only when calling
    :func:`~skorch.classifier.NeuralNetClassifier.predict` or
    :func:`~skorch.classifier.NeuralNetClassifier.predict_proba`.

    """
    # Implementation: At the moment, this function "dispatches" only
    # based on the criterion, not the class of the net. We still pass
    # the whole net as input in case we want to modify this at a
    # future point in time.
    if len(net._criteria) != 1:
        # don't know which criterion to consider, don't try to guess
        return _identity

    criterion = getattr(net, net._criteria[0] + '_')
    # unwrap optimizer in case of torch.compile being used
    criterion = getattr(criterion, '_orig_mod', criterion)

    if isinstance(criterion, CrossEntropyLoss):
        return partial(torch.softmax, dim=-1)

    if isinstance(criterion, BCEWithLogitsLoss):
        return _sigmoid_then_2d

    if isinstance(criterion, BCELoss):
        return _make_2d_probs

    return _identity

@contextmanager
def open_file_like(f, mode):
    """Wrapper for opening a file"""
    if isinstance(f, (str, pathlib.Path)):
        file_like = open(f, mode)
    else:
        file_like = f

    try:
        yield file_like
    finally:
        file_like.close()

class _TorchLoadUnpickler(pickle.Unpickler):
    """
    Subclass of pickle.Unpickler that intercepts 'torch.storage._load_from_bytes' calls
    and uses `torch.load(..., map_location=..., torch_load_kwargs=...)`.

    This way, we can use normal pickle when unpickling a skorch net but still benefit
    from torch.load to handle the map_location. Note that `with torch.device(...)` does
    not work for unpickling.

    """

    def __init__(self, *args, map_location, torch_load_kwargs, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_location = map_location
        self.torch_load_kwargs = torch_load_kwargs

    def find_class(self, module, name):
        # The actual serialized data for PyTorch tensors references
        # torch.storage._load_from_bytes internally. We intercept that call:
        if (module == 'torch.storage') and (name == '_load_from_bytes'):
            # Return a function that uses torch.load with our desired map_location
            def _load_from_bytes(b):
                return torch.load(
                    io.BytesIO(b),
                    map_location=self.map_location,
                    **self.torch_load_kwargs
                )
            return _load_from_bytes

        return super().find_class(module, name)

def split_indices(indices, splits, stratify=None):
    index_splits = []
    while len(splits) > 1:
        left_frac = splits[0]/np.sum(splits)
        right_frac = 1 - left_frac
        left, indices = train_test_split(indices, test_size=right_frac, stratify=stratify, shuffle=True)
        if stratify is not None:
            stratify = stratify[indices]
        index_splits.append(left)
        splits = splits[1:]
    index_splits.append(indices)

    return index_splits 

class FirstStepAccumulator:
    """Store and retrieve the train step data.

    This class simply stores the first step value and returns it.

    For most uses, ``skorch.utils.FirstStepAccumulator`` is what you
    want, since the optimizer calls the train step exactly
    once. However, some optimizerss such as LBFGSs make more than one
    call. If in that case, you don't want the first value to be
    returned (but instead, say, the last value), implement your own
    accumulator and make sure it is returned by
    ``NeuralNet.get_train_step_accumulator`` method.

    """
    def __init__(self):
        self.step = None

    def store_step(self, step):
        """Store the first step."""
        if self.step is None:
            self.step = step

    def get_step(self):
        """Return the stored step."""
        return self.step

class TeeGenerator:
    """Stores a generator and calls ``tee`` on it to create new generators
    when ``TeeGenerator`` is iterated over to let you iterate over the given
    generator more than once.

    """
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.gen, it = tee(self.gen)
        yield from it

def _check_f_arguments(caller_name, **kwargs):
    """Check file name arguments and return them
    """
    if kwargs.get('f_params') and kwargs.get('f_module'):
        raise TypeError("{} called with both f_params and f_module, please choose one"
                        .format(caller_name))

    kwargs_module = {}
    kwargs_other = {}
    keys_other = {'f_history', 'f_pickle'}
    for key, val in kwargs.items():
        if not key.startswith('f_'):
            raise TypeError(
                "{name} got an unexpected argument '{key}', did you mean 'f_{key}'?"
                .format(name=caller_name, key=key))

        if val is None:
            continue
        if key in keys_other:
            kwargs_other[key] = val
        else:
            # strip 'f_' prefix and attach '_', and normalize 'params' to 'module'
            # e.g. 'f_optimizer' becomes 'optimizer_', 'f_params' becomes 'module_'
            key = 'module_' if key == 'f_params' else key[2:] + '_'
            kwargs_module[key] = val
    return kwargs_module, kwargs_other
