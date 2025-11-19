import json
import pickle
from collections import defaultdict
from functools import partial

from beartype.typing import Any, Callable, Optional
from beartype.door import is_bearable

from optree import register_pytree_node_class
from optree import tree_structure, tree_transpose, tree_transpose_map
from optree import tree_flatten_one_level, tree_flatten, tree_unflatten
from optree import treespec_one_level, treespec_list, treespec_leaf

MISSING = object() 

def list_list() -> list[list[Any]]:
    return [[]]

dict_list = partial(defaultdict, list)

def is_list(x):
    return is_bearable(x, hint=list)

@register_pytree_node_class(namespace="churten.history")
class History(defaultdict):
    def __init__(self):
        super().__init__(list)

    def new_epoch(self):
        self["iepoch"].append(self.num_epochs+1)

        if "batches" in self:
            for key in self["batches"].keys():
                self["batches"][key].append([])
        
    def append(self, key : str, val : Any):
        self[key].append(val)
    
    def extend(self, key : str, val : list):
        self[key].extend(val)

    def new_batch(self):
        if "batches" not in self:
            self["batches"] = defaultdict(list_list)
        
        self["batches"]["ibatch"][-1].append(self.num_batches+1)

    def append_batch(self, key : str, val : Any):
        self["batches"][key][-1].append(val)

    def extend_batch(self, key : str, val : list):
        self["batches"][key][-1].extend(val)

    def to_defaultdict(self):
        return dict_list(self)

    def to_list(self, func: Optional[Callable]=None):
        return tree_transpose_map(
            func or (lambda x: x), self, 
            inner_treespec=treespec_list([treespec_leaf()]*self.num_epochs), 
            is_leaf = is_list, 
            namespace="churten.history",
        )

    def __missing__(self, key):
        if self.num_epochs < 2:
            return super().__missing__(key)
        
        default = super().__missing__(key)
        default.extend([MISSING for _ in range(self.num_epochs - 1)])
        return default
    
    def __tree_flatten__(self):
        children, metadata = tree_flatten(
            dict_list(self), is_leaf=is_list,
        )
        return children, metadata, None

    @classmethod
    def __tree_unflatten__(cls, metadata, children):
        return tree_unflatten(metadata, children)

    @property
    def num_epochs(self) -> int:
        return len(self.get("iepoch", []))

    @property
    def num_batches(self) -> int:
        return len(self["batches"].get("ibatch", [[]])[-1])

