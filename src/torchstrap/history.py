import json

import pickle
from collections import defaultdict
from functools import partial

from beartype.typing import Any, Callable, Optional
from beartype.door import is_bearable

from optree import register_pytree_node_class
from optree import tree_structure, tree_transpose, tree_transpose_map, tree_map
from optree import tree_flatten_one_level, tree_flatten, tree_unflatten
from optree import treespec_one_level, treespec_list, treespec_leaf

from torch import Tensor

from torchstrap.utils import open_file_like

MISSING = object()

def untensor(x):
    if is_bearable(x, Tensor):
        return x.tolist()
    return x

def list_list() -> list[list[Any]]:
    return [[]]

dict_list = partial(defaultdict, list)

def is_list(x):
    return is_bearable(x, hint=list)

@register_pytree_node_class(namespace="torchstrap.history")
class History(defaultdict):
    @property
    def num_epochs(self) -> int:
        return len(self.get("iepoch", []))

    @property
    def num_batches(self) -> int:
        return len(self["batches"].get("ibatch", [[]])[-1])

    def new_epoch(self):
        self["iepoch"].append(self.num_epochs+1)
        for key in self["batches"].keys():
            self["batches"][key].append([])
        
    def append(self, key : str, val : Any):
        self[key].append(val)
    
    def extend(self, key : str, val : list):
        self[key].extend(val)

    def new_batch(self):
        self["batches"]["ibatch"][-1].append(self.num_batches+1)

    def append_batch(self, key : str, val : Any):
        self["batches"][key][-1].append(val)

    def extend_batch(self, key : str, val : list):
        self["batches"][key][-1].extend(val)

    def row_slice(self, ie : int | slice, ib : int | slice | None =  None):
        row = {k : v[ie] for k, v in self.items()}

        ib =  ib if isinstance(ib, slice) else slice(ib)
        if isinstance(ie, slice):
            for key, epochs in row["batches"].items():
                for epoch, batch in enumerate(epochs):
                    row["batches"][key][epoch] = batch[ib]

        row["batches"] = {k : v[ib] for k, v in row["batches"].items()}

        return row

    def __getitem__(self, index : int | str | slice):
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            return self.row_slice(index)

    def to_defaultdict(self):
        return dict_list(self)
    
    def to_dict(self):
        res =  dict(**self)
        res["batches"] = dict(**res["batches"])

        return tree_map(untensor, res) 

    def to_file(self, f):
        with open_file_like(f, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4)
    
    @classmethod
    def from_file(cls, f):
        with open_file_like(f, 'r') as fp:
            return cls(json.load(fp))

    def to_list(self, func: Optional[Callable]=None):
        return tree_transpose_map(
            func or (lambda x: x), self, 
            inner_treespec=treespec_list([treespec_leaf()]*self.num_epochs), 
            is_leaf = is_list, 
            namespace="torchstrap.history",
        )

    def __missing__(self, key):
        self[key] = list() if key != "batches" else defaultdict(list_list)
        if self.num_epochs > 1:
            self[key].extend([MISSING for _ in range(self.num_epochs - 1)])
        return self[key]
    
    def __tree_flatten__(self):
        children, metadata = tree_flatten(
            dict_list(self), is_leaf=is_list,
        )
        return children, metadata, None

    @classmethod
    def __tree_unflatten__(cls, metadata, children):
        return tree_unflatten(metadata, children)


