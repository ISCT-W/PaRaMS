from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import torch
from jax import random as jax_random
from scipy.optimize import linear_sum_assignment

rngmix = lambda rng, x: jax_random.fold_in(rng, hash(x))


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = jnp.asarray(params[k])
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = jnp.take(w, jnp.array(perm[p]), axis=axis)
    return w


def vit_permutation_spec_MLP(num_layers: int = 12) -> PermutationSpec:
    """
    Permutation spec for ViT. num_layers=12 for B/32 and B/16, numl_layers=24 for L/14.
    :param num_layers: Number of resblocks.
    :return: perm spec
    """
    assert num_layers >= 1
    perm_spec = {}
    for layer_idx in range(num_layers):
        perm_spec.update({
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight": (f"P_{layer_idx}", None),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight": (None, f"P_{layer_idx}"),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias": (f"P_{layer_idx}",),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias": (None,),
        })

    return permutation_spec_from_axes_to_perm(perm_spec)


def apply_permutation(ps: PermutationSpec, perm, params):
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def parameter_rearrangement(rng, ps: PermutationSpec, params_a, params_b, max_iter=200, obj='mismatching',
                            tolerance=10):
    """
    MLP Parameter rearrangement.
    :param tolerance:
    :param rng: rng.
    :param ps: Permutation Spec.
    :param params_a: state dict of defender.
    :param params_b: state dict of free-rider.
    :param max_iter: iterations to run (For multi-layer MLP).
    :param obj: Match or mismatch parameters. Mismatch for adaptive.
    :param tolerance: Tolerance in optimization.
    :return: permutation matrix.
    """
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    best_iter = 0
    similarity = []
    loss_interp, acc_interp = [], []

    for iteration in range(max_iter):
        print(f'Searching P for the {iteration} times...')
        progress = False
        for p_ix in jax_random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = jnp.asarray(params_a[wk])
                w_b = jnp.asarray(get_permuted_param(ps, perm, wk, params_b, except_axis=axis))
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            if obj == 'mismatching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=False)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL < oldL + 1e-12:
                    perm[p] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration
            elif obj == 'matching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=True)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL > oldL + 1e-12:
                    perm[p] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration
            else:
                raise ValueError("Unknown matching objective!")

        if not progress and iteration - best_iter >= tolerance:
            break

    return {k: torch.tensor(np.array(v)) for k, v in perm.items()}, similarity, loss_interp, acc_interp


def apply_attention_qkvw_scaling(
        params: dict,
        layer_idx: int,
        scale_min: float = 0.5,
        scale_max: float = 20.0,
        rng=None,
        mode="uniform"
):
    """
    Separately scales Q,K and V,W_out for the specified attention layer.

    Q,K are scaled by alpha:
        Q -> alpha * Q
        K -> (1/alpha) * K
    V,W_out are scaled by gamma:
        V -> gamma * V
        W_out -> (1/gamma) * W_out

    :param params: The model state dict.
    :param layer_idx: The index of the Transformer layer to scale.
    :param scale_min: Minimum scaling coefficient.
    :param scale_max: Maximum scaling coefficient.
    :param rng: Optional random generator.
    :param mode: Sampling mode for alpha,gamma. "log_uniform", "beta", or "uniform".
    :return: The updated params with scaled Q,K,V,W_out (and their biases).
    """

    key_in_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight"
    key_in_proj_bias = f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias"
    key_out_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight"

    W_in_proj = params[key_in_proj_weight]
    b_in_proj = params[key_in_proj_bias]
    W_out_proj = params[key_out_proj_weight]

    d_k = W_in_proj.shape[0] // 3

    Q_weight = W_in_proj[0: d_k, :]
    K_weight = W_in_proj[d_k: 2 * d_k, :]
    V_weight = W_in_proj[2 * d_k: 3 * d_k, :]

    Q_bias = b_in_proj[0: d_k]
    K_bias = b_in_proj[d_k: 2 * d_k]
    V_bias = b_in_proj[2 * d_k: 3 * d_k]

    def sample_factors(num, mode_str, low, high, gen):
        """
        Samples an array of length 'num' from [low, high],
        according to the specified 'mode_str'.
        """
        if mode_str == "log_uniform":
            # log-uniform
            if gen is None:
                return torch.exp(torch.empty(num).uniform_(np.log(low), np.log(high)))
            else:
                return torch.exp(
                    np.log(low) + (np.log(high) - np.log(low)) * torch.rand(num, generator=gen)
                )
        elif mode_str == "beta":
            # Beta(0.5, 0.5) => more concentration near edges
            a, b = 0.5, 0.5
            if gen is None:
                beta_samples = np.random.beta(a, b, size=num)
                return torch.tensor(low + (high - low) * beta_samples, dtype=torch.float32)
            else:
                dist = torch.distributions.Beta(a, b)
                beta_samples = dist.sample((num,))
                return low + (high - low) * beta_samples
        elif mode_str == "uniform":
            # plain uniform
            if gen is None:
                return torch.empty(num).uniform_(low, high)
            else:
                return low + (high - low) * torch.rand(num, generator=gen)
        else:
            raise ValueError("Unsupported mode. Use 'log_uniform', 'beta', or 'uniform'.")

    alpha = sample_factors(d_k, mode, scale_min, scale_max, rng)
    gamma = sample_factors(d_k, mode, scale_min, scale_max, rng)

    for i in range(d_k):
        Q_weight[i, :] *= alpha[i]
        Q_bias[i] *= alpha[i]

        K_weight[i, :] *= 1.0 / alpha[i]
        K_bias[i] *= 1.0 / alpha[i]

        V_weight[i, :] *= gamma[i]
        V_bias[i] *= gamma[i]
        W_out_proj[:, i] *= 1.0 / gamma[i]

    W_in_proj[0: d_k, :] = Q_weight
    W_in_proj[d_k: 2 * d_k, :] = K_weight
    W_in_proj[2 * d_k: 3 * d_k, :] = V_weight

    b_in_proj[0: d_k] = Q_bias
    b_in_proj[d_k: 2 * d_k] = K_bias
    b_in_proj[2 * d_k: 3 * d_k] = V_bias

    params[key_in_proj_weight] = W_in_proj
    params[key_in_proj_bias] = b_in_proj
    params[key_out_proj_weight] = W_out_proj

    return params


def evaluate_model(model, data_loader, criterion):
    " Useless currently! "
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to('cpu'), target.to('cpu')
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return total_loss / total, correct / total


def interpolate_params(params_a, params_b, alpha):
    " Construct interpolation model of a and b with a scaling factor: alpha. Used in CTL check. "
    return {name: (1 - alpha) * params_a[name] + alpha * params_b[name] for name in params_a}
