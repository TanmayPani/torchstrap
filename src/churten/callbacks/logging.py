import sys
import time
import tempfile
from contextlib import suppress
from numbers import Number
from itertools import cycle
from pathlib import Path

import torch
import numpy as np
import tqdm
from tabulate import tabulate

from churten.utils import Ansi
from .callbacks import Callback

def filter_log_keys(keys, keys_ignored=None):
    """Filter out keys that are generally to be ignored.
    """
    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key

class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur``.
    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - (self.epoch_start_time_ or 0))

class PrintLog(Callback):
    """Print useful information from the model's history as a table.
    """
    def __init__(
            self,
            keys_ignored=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        #self.keys_ignored_.add('batches')
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable).

        """
        _row = row[key]

        _values = _row.mean(dim=0)
        _value_err, _value = torch.std_mean(_values, dim=0)

        value = _value.item()
        value_err = _value_err.item()

        return f"{value: .4f} +/- {value_err: .4f}"

#        if isinstance(value, bool) or value is None:
#            return '+' if value else ''
#
#        if not isinstance(value, Number):
#            return value
#
        # determine if integer value
#        is_integer = float(value).is_integer()
#        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

#        # if numeric, there could be a 'best' key
#        key_best = key + '_best'
#        if (key_best in row) and row[key_best]:
#            template = color + template + Ansi.ENDC.value
#        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.
        """
        sorted_keys = []

        # make sure 'epoch' comes first
        #if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
        #    sorted_keys.append('epoch')

        # ignore keys like *_best or event_*
        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        # add event_* keys
        #for key in sorted(keys):
        #    if key.startswith('event_') and (key not in self.keys_ignored_):
        #        sorted_keys.append(key)

        # make sure 'dur' comes last
        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        for key, color in zip(self._sorted_keys(row.keys()), colors):
            formatted = self.format_row(row, key, color=color)
            #if key.startswith('event_'):
            #    key = key[6:]
            yield key, formatted

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, verbose=True,**kwargs):
        data = net.history[-1]
        #verbose = net.verbose
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()


class ProgressBar(Callback):
    """Display a progress bar for each epoch.
    """
    def __init__(
            self,
            batches_per_epoch='auto',
            detect_notebook=True,
            postfix_keys=None
    ):
        self.batches_per_epoch = batches_per_epoch
        self.detect_notebook = detect_notebook
        self.postfix_keys = postfix_keys or ['train_loss', 'valid_loss']

    def in_ipynb(self):
        try:
            return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except NameError:
            return False

    def _use_notebook(self):
        return self.in_ipynb() if self.detect_notebook else False

    def _get_batch_size(self, net, training):
        name = 'iterator_train' if training else 'iterator_valid'
        net_params = net.get_params()
        #print(net_params)
        #print(hasattr(net, "batch_size"))
        return net_params.get(name + '__batch_size', net_params['batch_size'])

    def _get_batches_per_epoch_phase(self, net, dataset, training):
        if dataset is None:
            return 0
        batch_size = self._get_batch_size(net, training)
        return int(np.ceil(len(dataset) / batch_size))

    def _get_batches_per_epoch(self, net, dataset_train, dataset_valid):
        return (self._get_batches_per_epoch_phase(net, dataset_train, True) +
                self._get_batches_per_epoch_phase(net, dataset_valid, False))

    def _get_postfix_dict(self, net):
        postfix = {}
        for key in self.postfix_keys:
            try:
                postfix[key] = net.history[-1, 'batches', -1, key]
            except KeyError:
                pass
        return postfix

    # pylint: disable=attribute-defined-outside-init
    def on_batch_end(self, net, **kwargs):
        self.pbar_.set_postfix(self._get_postfix_dict(net), refresh=False)
        self.pbar_.update()

    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # Assume it is a number until proven otherwise.
        batches_per_epoch = self.batches_per_epoch

        if self.batches_per_epoch == 'auto':
            batches_per_epoch = self._get_batches_per_epoch(
                net, dataset_train, dataset_valid
            )
        elif self.batches_per_epoch == 'count':
            if len(net.history) <= 1:
                # No limit is known until the end of the first epoch.
                batches_per_epoch = None
            else:
                batches_per_epoch = len(net.history[-2, 'batches'])

        if self._use_notebook():
            self.pbar_ = tqdm.tqdm_notebook(total=batches_per_epoch, leave=False)
        else:
            self.pbar_ = tqdm.tqdm(total=batches_per_epoch, leave=False)

    def on_epoch_end(self, net, **kwargs):
        self.pbar_.close()

    def __getstate__(self):
        # don't save away the temporary pbar_ object which gets created on
        # epoch begin anew anyway. This avoids pickling errors with tqdm.
        state = self.__dict__.copy()
        state.pop('pbar_', None)
        return state
