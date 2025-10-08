from functools import partial
from math import pi

from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import torch 
from torch.func import vmap, functional_call
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from torch.distributions import Categorical

from churten.ensemble import Ensemble
from churten.optimizer import Adam

def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = torch.linspace(0, 1, n_samples)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * pi
    signs = torch.randint(0, 2, (n_samples,)) * 2 - 1
    labels = (signs > 0).to(torch.int8)

    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return points, labels

def make_classifier_module(*layer_sizes, device = "cpu", dtype = torch.float32):
    layer_sizes, output_size = layer_sizes[:-1], layer_sizes[-1]
    module = Sequential()
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        module.extend(
            Sequential(
                Linear(in_size, out_size), 
                ReLU(),
            )
        )
    module.append(Linear(layer_sizes[-1], output_size))
    return module.to(device=device, dtype=dtype)

def make_batched_indices(seed, *, dataset_size, sample_size=()):
    dist = Categorical(torch.ones(dataset_size))
    return dist.sample(sample_size)

def parallel_batch_iterator(
    X, y, *, 
    num_replicas = 1,
    batch_size = 32,
    num_batches = 100,
):
    total_samples = batch_size*num_batches

    indices = vmap(
        make_batched_indices, 
        randomness="different"
    )(
        torch.arange(num_replicas), 
        dataset_size=X.shape[0], 
        sample_size=(total_samples,),
    )

    for i in range(0, total_samples, batch_size):
        batch_indices = indices[:, i:i+batch_size]
        yield (
            X[batch_indices.ravel()].reshape(*(batch_indices.shape), *(X.shape[1:])), 
            y[batch_indices.ravel()].reshape(*(batch_indices.shape), 1), 
            None,
        )

def predict_fn(model, params, buffers, inputs):
    return sigmoid(functional_call(model, (params, buffers), inputs))

def predict_on_mesh(ensemble, width=1.5, steps=50):
    with torch.inference_mode():
        xs = torch.linspace(-width, width, steps=steps, device=ensemble.device)
        ys = torch.linspace(-width, width, steps=steps, device=ensemble.device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    
        points = torch.stack([xx.ravel(), yy.ravel()], dim=1).expand(ensemble.num_replicas, xx.numel(), 2)
        fpred = vmap(partial(predict_fn, ensemble._base_model))
        z = fpred(ensemble._params_dict, ensemble._buffers_dict, points)

        z_mean = z.mean(dim=0).reshape_as(xx)

        return xx.detach().cpu(), yy.detach().cpu(), z_mean.detach().cpu()
    
def plot_predictions(ax, xx, yy, z):
    return ax.imshow(
        z, 
        extent=(
            xx.min(), xx.max(), 
            yy.min(), yy.max(),
        ), 
        origin="lower", 
        cmap=LinearSegmentedColormap.from_list(
            "blueorange", 
            ["xkcd:darkblue", "white", "xkcd:orangered"],
            #["tab:blue", "white", "tab:orange"],
        ), 
        vmin=0, vmax=1, 
        aspect="equal",
    )


def plot_spirals(ax, points, labels):
    return ax.scatter(
        points[:, 0], 
        points[:, 1], 
        c=labels,
        cmap = ListedColormap([
            #"xkcd:darkblue", "xkcd:orangered", 
            "tab:blue", "tab:orange",
        ]),
        edgecolors = "white",
    )

if __name__ == "__main__":
    device = "cuda"
    num_replicas = 100
    num_samples = 100
    batch_size = 32
    num_batches = 200

    torch.manual_seed(0)
    
    points, labels = make_spirals(num_samples, noise_std=0.05)

    data_iterator = parallel_batch_iterator(
        points, 
        labels, 
        num_replicas=num_replicas, 
        batch_size=batch_size, 
        num_batches=num_batches,
    )

    print("Initialized dataset ...")
    
    ensemble = Ensemble(
        make_classifier_module,
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        model_init_args=(2, 512, 512, 1),
        device = device,
        model_init_randomness="different",
    )

    print("Initialized ensemble for bootstrapping ...")
    
    optimizer = Adam(lr=1e-3, batch_size=(num_replicas,), device=device)
    optimizer.init(ensemble._params_dict)
    ensemble.train(True)
    losses = ensemble.fit_step(optimizer, data_iterator)
    
    print("Training ensemble with bootstrapping done ...")

    dy, y = torch.std_mean(losses.detach_().cpu(), dim=0)
    x = torch.arange(num_batches)

    fig0, ax0 = plt.subplots()
    ax0.set_title("Cross entropy loss vs # minibatch iterations", weight="bold")
    ax0.set_xlabel("minibatch iterations", weight="bold")
    ax0.set_ylabel("loss", weight="bold")
    ax0.plot(x, y, "-", label="loss")
    ax0.fill_between(x, y-dy, y+dy, alpha=0.2, label=r"$\Delta$(loss)")
    ax0.legend()
    fig0.savefig("loss.png")


    xx, yy, z = predict_on_mesh(ensemble)

    fig = plt.figure()
    #fig.suptitle("Predictions from bootstraped ensemble", weight="bold")
    ax = fig.add_subplot()
    #fig.subplots_adjust(top=0.8)
    ax.set_title(f"Predictions from bootstrap ({num_replicas} resamples)", weight="bold", y=0.95, pad = 30)

    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    
    ax.set_xlabel("x", weight="bold")
    ax.set_ylabel("y", weight="bold")

    sc = plot_spirals(ax, points, labels)
    im = plot_predictions(ax, xx, yy, z)
    ax.legend(
        sc.legend_elements()[0], 
        ["0", "1"], 
        title = "label",
        title_fontproperties = FontProperties(weight="bold"), 
        loc="lower right",
        #frameon=False,
        alignment = "center",
    )
    fig.colorbar(im, ax=ax, label="mean prediction")
    fig.savefig("predictions.png")
    
    show_plot = True
    if show_plot:
        try:
            plt.show()
        except KeyboardInterrupt:
            exit()



