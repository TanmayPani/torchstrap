from copy import deepcopy
from typing import Callable, Union, Protocol, Optional, Any
from typing import runtime_checkable
from collections.abc import Iterator, Mapping, Sequence

import torch

from torch import Tensor, device

from torch.utils import _pytree

from torch.func import functional_call
from torch.func import vmap, grad_and_value

from torch.nn import Parameter
from torch.nn import ParameterDict
from torch.nn import Module

import torch.nn.functional as F

from churten.optimizer import GradientTransformations
from churten.utils import params_for

@runtime_checkable
class TensorCallable(Protocol):
    def __call__(
            self,
            #tensor : Tensor | Sequence[Tensor] | Mapping[str, Tensor],
            *args : Any, 
            **kwargs : Any,
    ) -> Tensor | tuple[Tensor, ...]:
        ...

type ModuleLike = Union[Module, type[Module], TensorCallable, type[TensorCallable]]

@torch.no_grad
def init_to_zero(module):
    for param in module.parameters():
        param.zero_()


class Lambda(Module):
    def __init__(
        self, fn : TensorCallable
    ):
        super().__init__()
        self.forward = fn

def initialize_layer(layer : ModuleLike, *args, **kwargs) -> Module:
    if isinstance(layer, Module):
        return layer
    
    if isinstance(layer, TensorCallable) and not isinstance(layer, type):
        return Lambda(layer)

    if isinstance(layer, type) and issubclass(layer, Module):
        return layer(*args, **kwargs)

    return Lambda(layer(*args, **kwargs))

   
class Layer(Module):
    def __init__(self,
                 module : ModuleLike,
                 input_size : int,
                 output_size : int, 
                 dropout_prob : Optional[float] = None, 
                 batch_norm : Optional[type[Module]] = None, 
                 activation : Optional[ModuleLike] = None,
                 **kwargs, 
                ):
        super().__init__()

        layer_kwargs = params_for("layer", kwargs)

        self.layeritos = torch.nn.ModuleDict()

        if dropout_prob is not None:
            self.layeritos["dropout"] = torch.nn.Dropout(p=dropout_prob)
        
        if batch_norm is not None:
            layer_kwargs["bias"] = False

        self.layeritos["layer"] = initialize_layer(module, input_size, output_size, **layer_kwargs)

        if batch_norm is not None:
            batch_norm_kwargs = params_for("batch_norm", kwargs)
            self.layeritos["batch_norm"] = initialize_layer(batch_norm, output_size, **batch_norm_kwargs)

        if activation is not None:
            activation_kwargs = params_for("activation", kwargs)
            activation_args = activation_kwargs.pop("args", ())
            self.layeritos["activation"] = initialize_layer(activation, *activation_args, **activation_kwargs)
     
    def forward(self, x, training=False):
        #x = F.dropout(x, p = self.dropout_p, training=training)
        for _, module in self.layeritos.items():
            x = module(x)
        return x
 
        
class MLP(Module):
    def __init__(self,
                 layer : ModuleLike | list[ModuleLike] = torch.nn.Linear, 
                 activation : Optional[ModuleLike | list[Optional[ModuleLike]]] = torch.nn.ReLU, 
                 layer_sizes=[10, 1], 
                 output_activation : Optional[ModuleLike] = None,
                 input_transform : Optional[ModuleLike]=None,
                 **kwargs):
        
        super().__init__()
        
        nlayers = len(layer_sizes)-1 
        assert nlayers > 0

        if isinstance(layer, (list)):
            assert len(layer) == nlayers
            layers = layer
        else:
            layers = [layer]*nlayers

        if isinstance(activation, (list)):
            assert len(activation) >= nlayers-1
            activations = activation
        else:
            activations = [activation]*nlayers
        activations[-1] = output_activation


        for key, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == nlayers
            else:
                kwargs[key] = [value]*nlayers

        list_kwargs = [{key:val[i] for key, val in kwargs.items()} for i in range(nlayers)]


        self.layers = torch.nn.ModuleList()
        if input_transform is not None:
            input_transform_kwargs = params_for("input_transform", kwargs)
            input_transform_args = input_transform_kwargs.pop("args", ())
            self.layers.append(initialize_layer(input_transform, *input_transform_args, **input_transform_kwargs))
            
        for ilayer, (input_size, output_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(Layer(layers[ilayer], input_size, output_size, activation=activations[ilayer], **(list_kwargs[ilayer])))    

    def forward(self, x, training=False):
        self.train(mode=training)
        #print(x[0])
        for layer in self.layers:
            x = layer(x)
        return x

class DNN(Module):
    def __init__(self, layer_sizes=[100, 1], dropout_prob=None, do_batch_norm=False, output_activation=None, input_transform : Optional[torch.nn.Module]=None, **kwargs):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        if input_transform is not None:
            self.layers.append(input_transform)

        self.layer_sizes = layer_sizes[:-1]
        self.output_size = layer_sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Linear(input_size, output_size))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())
            
        self.layers.append(torch.nn.Linear(self.layer_sizes[-1], self.output_size))
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)
    
    def forward(self, x, training=False):
        #print(training)
        self.train(mode=training)
        #print(x[0])
        #print(x)
        for layer in self.layers:
            x = layer(x)
            #print(x)
        return x
    
class Conv1dNN(Module):
    def __init__(self, sizes, output_activation=None, dropout_prob=None, do_batch_norm=False, **kwargs):
        super().__init__()
        
        assert len(sizes) > 1
        self.layers = torch.nn.ModuleList()
        self.layer_sizes = sizes[:-1]
        self.output_size = sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Conv1d(input_size, output_size, kernel_size=1))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())
            
        self.layers.append(torch.nn.Conv1d(self.layer_sizes[-1], self.output_size, kernel_size=1))
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SumPooling(Module):
    def __init__(self, dim, mask_dim=1):
        super().__init__()
        self.dim = dim
        self.mask_dim = mask_dim
    
    def _resize_mask(self, mask, size, dim):
        return mask.select(dim, 0).unsqueeze_(dim).expand(size)
        
    def forward(self, x, mask=None):
        if mask is not None:
            _new_mask = self._resize_mask(mask, x.size(), self.mask_dim)
            x = torch.where(_new_mask, x, torch.zeros_like(x))
        return torch.sum(x, dim=self.dim)
    
class PFN(Module):
    def __init__(self, phi_sizes, f_sizes, output_activation=None,
                 phi_dropout_prob=None, phi_do_batch_norm=False, 
                f_dropout_prob=None, f_do_batch_norm=False):
        super().__init__()
        
        self.phi_sizes = phi_sizes
        self.f_sizes = [phi_sizes[-1]] + f_sizes
        
        assert len(self.phi_sizes) > 1 and len(self.f_sizes) > 1
        
        self.phi_layers = torch.nn.ModuleList()
        self.phi_layers.append(Conv1dNN(phi_sizes, output_activation=torch.nn.ReLU(), dropout_prob=phi_dropout_prob, do_batch_norm=phi_do_batch_norm))
        
        self.pooling_layer = SumPooling(-1)
        
        self.f_layers = torch.nn.ModuleList()
        self.f_layers.append(DNN(self.f_sizes, output_activation=output_activation, dropout_prob=f_dropout_prob, do_batch_norm=f_do_batch_norm))
                
    def forward(self, x, mask):
        for layer in self.phi_layers:
            x = layer(x)
        x = self.pooling_layer(x, mask)
        for layer in self.f_layers:
            x = layer(x)
        return x
        
class JetNN(Module):
    def __init__(self, phi_sizes, f_sizes, jet_dnn_sizes, output_dnn_sizes, dropout_prob=None, do_batch_norm=False):
        super().__init__()
        self.pfn = PFN(phi_sizes, f_sizes, 
                       phi_dropout_prob=dropout_prob, phi_do_batch_norm=do_batch_norm, 
                       f_dropout_prob=dropout_prob, f_do_batch_norm=do_batch_norm,
                       output_activation=torch.nn.ReLU()
                    )  
        self.jet_dnn = DNN(jet_dnn_sizes, 
                           dropout_prob=dropout_prob, do_batch_norm=do_batch_norm,
                           output_activation=torch.nn.ReLU()
                        )
        self.output_dnn_sizes = [jet_dnn_sizes[-1] + f_sizes[-1]] + output_dnn_sizes
        self.output_dnn = DNN(self.output_dnn_sizes, 
                              dropout_prob=dropout_prob, do_batch_norm=do_batch_norm
                            )        
        
    def forward(self, jet, constits, mask):
        jet_dnn_output = self.jet_dnn(jet)
        pfn_output = self.pfn(constits, mask)
        output = torch.cat([jet_dnn_output, pfn_output], dim=-1)
        output = self.output_dnn(output)
        return output
        
        
