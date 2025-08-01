from collections import defaultdict
from typing import NamedTuple, Dict, Tuple, Optional, Union

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


def parameter_rearrangement(
    rng, 
    ps: PermutationSpec, 
    params_a: Dict[str, torch.Tensor], 
    params_b: Dict[str, torch.Tensor], 
    max_iter: int = 200, 
    obj: str = 'mismatching',
    tolerance: int = 1
) -> Tuple[Dict[str, torch.Tensor], list, list, list]:
    """
    MLP Parameter rearrangement to find optimal permutation that disrupts model merging.
    
    This function uses the Hungarian algorithm to find permutations that either maximize 
    similarity (matching) or minimize similarity (mismatching) between MLP parameters.
    
    Args:
        rng: JAX random key for reproducible random operations
        ps (PermutationSpec): Permutation specification defining which axes to permute
        params_a (Dict[str, torch.Tensor]): State dict of the defender model (victim)
        params_b (Dict[str, torch.Tensor]): State dict of the free-rider model (pretrained)
        max_iter (int): Maximum iterations to run optimization (default: 200)
        obj (str): Optimization objective - 'mismatching' for defense, 'matching' for alignment
        tolerance (int): Early stopping tolerance - stop if no improvement for this many iterations
    
    Returns:
        Tuple containing:
        - Dict[str, torch.Tensor]: Optimal permutation matrices for each parameter group
        - list: Similarity scores across iterations (currently empty)
        - list: Loss interpolation values (currently empty) 
        - list: Accuracy interpolation values (currently empty)
        
    Note:
        Uses Hungarian algorithm (linear_sum_assignment) to solve the assignment problem
        optimally at each iteration.
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
        params: Dict[str, torch.Tensor],
        layer_idx: int,
        scale_min: float = 0.5,
        scale_max: float = 20.0,
        rng: Optional[torch.Generator] = None,
        mode: str = "uniform"
) -> Dict[str, torch.Tensor]:
    """
    Applies diagonal scaling to attention parameters to disrupt model merging.
    
    This is the original and most computationally efficient attention scaling method.
    It uses independent scalar scaling for each dimension of the attention parameters,
    creating a diagonal transformation matrix.
    
    Mathematical formulation (per dimension):
    - Q_i -> α_i * Q_i, K_i -> (1/α_i) * K_i  
    - V_i -> γ_i * V_i, W_out_i -> (1/γ_i) * W_out_i
    
    Key properties:
    - Attention output remains functionally equivalent: (Q'K'^T)V'W_O' = QK^T VW_O
    - Each dimension scaled independently (diagonal transformation)
    - Fastest computational complexity: O(d) scaling operations
    - Good balance between efficiency and protection
    - Preserves independence of attention dimensions
    
    Differences from other scaling modes:
    - **Diagonal**: Uses independent scalar scaling per dimension (this function)
    - **Symmetric**: Uses symmetric positive definite matrices (stronger protection)
    - **Nonsymmetric**: Uses full invertible matrices (strongest protection)
    
    Args:
        params (Dict[str, torch.Tensor]): The model state dictionary to modify
        layer_idx (int): The index of the transformer layer to scale (0-based)
        scale_min (float): Minimum scaling coefficient (default: 0.5)
        scale_max (float): Maximum scaling coefficient (default: 20.0)
        rng (Optional[torch.Generator]): Random generator for reproducible scaling
        mode (str): Sampling distribution - "uniform", "log_uniform", or "beta"
    
    Returns:
        Dict[str, torch.Tensor]: Updated parameters with diagonal attention scaling applied
        
    Note:
        - Most computationally efficient scaling method
        - Recommended for most practical applications
        - Provides good protection with minimal computational overhead
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
    # gamma = sample_factors(d_k, mode, scale_min, scale_max, rng)

    for i in range(d_k):
        Q_weight[i, :] *= alpha[i]
        Q_bias[i] *= alpha[i]

        K_weight[i, :] *= 1.0 / alpha[i]
        K_bias[i] *= 1.0 / alpha[i]

        # V_weight[i, :] *= gamma[i]
        # V_bias[i] *= gamma[i]
        # W_out_proj[:, i] *= 1.0 / gamma[i]
        V_weight[i, :] *= alpha[i]
        V_bias[i] *= alpha[i]
        W_out_proj[:, i] *= 1.0 / alpha[i]

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


def apply_attention_qkvw_blockdiag_nonsym(
        params: Dict[str, torch.Tensor],
        layer_idx: int,
        n_heads: int = 12,
        rng: Optional[torch.Generator] = None,
        scale_min: float = 0.5,
        scale_max: float = 20.0,
        eye_eps: float = 1e-2
) -> Dict[str, torch.Tensor]:
    """
    Applies per-head block-diagonal nonsymmetric transformation to attention parameters.
    
    This advanced scaling method uses nonsymmetric invertible matrices for each attention head,
    providing stronger protection against model merging while maintaining functional equivalence.
    
    Mathematical formulation:
    - W_Q' = A × W_Q        (Query transformation)
    - W_K' = A^(-T) × W_K   (Key transformation with inverse transpose)  
    - W_V' = A × W_V        (Value transformation, B = A)
    - W_O' = W_O × A^(-1)   (Output transformation)
    
    Where A is a block-diagonal matrix: A = block_diag(A_1, A_2, ..., A_n_heads)
    Each A_h is a random nonsymmetric invertible matrix for head h.
    
    Key properties:
    - Maintains attention output: QK^T and VW_O computations unchanged
    - Nonsymmetric matrices provide maximum parameter space coverage
    - Each head gets independent random transformation
    - Stronger defense compared to diagonal or symmetric variants
    
    Args:
        params (Dict[str, torch.Tensor]): Model state dictionary to modify
        layer_idx (int): Transformer layer index (0-based)
        n_heads (int): Number of attention heads (default: 12)
        rng (Optional[torch.Generator]): Random generator for reproducibility
        scale_min (float): Minimum singular value scale (default: 0.5)
        scale_max (float): Maximum singular value scale (default: 20.0)
        eye_eps (float): Identity regularization term for invertibility (default: 1e-2)
    
    Returns:
        Dict[str, torch.Tensor]: Updated parameters with nonsymmetric block-diagonal scaling
        
    Note:
        This is the most general form of attention scaling, offering maximum flexibility
        and strongest protection against model merging attacks.
    """
    
    base = f"model.visual.transformer.resblocks.{layer_idx}.attn"
    W_in = params[f"{base}.in_proj_weight"].clone()  # [3d_k, d_model]
    b_in = params[f"{base}.in_proj_bias"].clone()  # [3d_k]
    W_out = params[f"{base}.out_proj.weight"].clone()  # [d_model, d_model]

    d_k = W_in.shape[0] // 3
    head_dim = d_k // n_heads

    Q_w, K_w, V_w = W_in[:d_k], W_in[d_k:2 * d_k], W_in[2 * d_k:]
    Q_b, K_b, V_b = b_in[:d_k], b_in[d_k:2 * d_k], b_in[2 * d_k:]

    # Build block-diagonal A, A^(-1), A^(-T)
    g = torch.Generator(device=W_in.device) if rng is None else rng
    A_blocks, Ainv_blocks, AinvT_blocks = [], [], []

    log_min, log_max = np.log(scale_min), np.log(scale_max)

    for _ in range(n_heads):
        # Generate random full matrix + eps*I for invertibility and nonsymmetry
        A_h = torch.randn(head_dim, head_dim, generator=g, device=W_in.device)
        A_h += eye_eps * torch.eye(head_dim, device=W_in.device)
        
        # Scale singular values to desired range
        u = torch.rand(head_dim, generator=g, device=W_in.device)
        s = torch.exp(u * (log_max - log_min) + log_min)  # log-uniform sampling
        A_h = A_h @ torch.diag(s)  # Modify singular value range

        A_inv_h = torch.inverse(A_h)
        A_invT_h = A_inv_h.T
        A_blocks.append(A_h)
        Ainv_blocks.append(A_inv_h)
        AinvT_blocks.append(A_invT_h)

    A = torch.block_diag(*A_blocks)
    A_inv = torch.block_diag(*Ainv_blocks)
    A_inv_T = torch.block_diag(*AinvT_blocks)

    # Apply transformations
    Q_w, Q_b = A @ Q_w, A @ Q_b
    K_w, K_b = A_inv_T @ K_w, A_inv_T @ K_b  # Note: using A^(-T)
    V_w, V_b = A @ V_w, A @ V_b
    W_out = W_out @ A_inv  # Right multiply

    # Write back to parameters
    W_in[:d_k], W_in[d_k:2 * d_k], W_in[2 * d_k:] = Q_w, K_w, V_w
    b_in[:d_k], b_in[d_k:2 * d_k], b_in[2 * d_k:] = Q_b, K_b, V_b
    params[f"{base}.in_proj_weight"] = W_in
    params[f"{base}.in_proj_bias"] = b_in
    params[f"{base}.out_proj.weight"] = W_out
    return params


def apply_attention_qkvw_blockdiag_sym(
        params: Dict[str, torch.Tensor],
        layer_idx: int,
        n_heads: int = 12,
        rng: Optional[torch.Generator] = None,
        scale_min: float = 0.05,
        scale_max: float = 20.0
) -> Dict[str, torch.Tensor]:
    """
    Applies per-head block-diagonal symmetric transformation to attention parameters.
    
    This scaling method uses symmetric positive definite matrices for each attention head,
    providing a balance between protection strength and mathematical elegance.
    
    Mathematical formulation:
    - W_Q' = A × W_Q        (Query transformation)
    - W_K' = A^(-1) × W_K   (Key transformation with inverse)
    - W_V' = A × W_V        (Value transformation, B = A)  
    - W_O' = W_O × A^(-1)   (Output transformation)
    
    Where A is a block-diagonal matrix: A = block_diag(A_1, A_2, ..., A_n_heads)
    Each A_h is a symmetric positive definite matrix: A_h = U × diag(s) × U^T
    
    Key properties:
    - Maintains attention output: QK^T and VW_O computations unchanged
    - Symmetric matrices ensure numerical stability
    - Each head gets independent random transformation
    - Intermediate protection level between diagonal and nonsymmetric variants
    
    Args:
        params (Dict[str, torch.Tensor]): Model state dictionary to modify
        layer_idx (int): Transformer layer index (0-based)
        n_heads (int): Number of attention heads (default: 12)
        rng (Optional[torch.Generator]): Random generator for reproducibility
        scale_min (float): Minimum eigenvalue scale (default: 0.05)
        scale_max (float): Maximum eigenvalue scale (default: 20.0)
    
    Returns:
        Dict[str, torch.Tensor]: Updated parameters with symmetric block-diagonal scaling
        
    Note:
        Symmetric transformations provide good protection while maintaining numerical
        stability and theoretical interpretability.
    """
    
    base = f"model.visual.transformer.resblocks.{layer_idx}.attn"
    W_in = params[f"{base}.in_proj_weight"].clone()  # [3d_k, d_model]
    b_in = params[f"{base}.in_proj_bias"].clone()  # [3d_k]
    W_out = params[f"{base}.out_proj.weight"].clone()  # [d_model, d_model]

    d_k = W_in.shape[0] // 3  # full embed_dim
    head_dim = d_k // n_heads  # per-head dim

    Q_w, K_w, V_w = W_in[:d_k], W_in[d_k:2 * d_k], W_in[2 * d_k:]
    Q_b, K_b, V_b = b_in[:d_k], b_in[d_k:2 * d_k], b_in[2 * d_k:]

    # Build block-diagonal A and its inverse
    g = torch.Generator(device=W_in.device) if rng is None else rng
    blocks, inv_blocks = [], []

    log_min, log_max = np.log(scale_min), np.log(scale_max)

    for _ in range(n_heads):
        # Generate random orthogonal matrix U via QR decomposition
        U, _ = torch.linalg.qr(torch.randn(head_dim, head_dim,
                                           generator=g, device=W_in.device))
        # Generate random eigenvalues in log-uniform distribution
        diag = torch.exp(
            torch.rand(head_dim, generator=g, device=W_in.device)
            * (log_max - log_min) + log_min
        )
        # Construct symmetric positive definite matrix: A_h = U × diag × U^T
        A_h = U @ torch.diag(diag) @ U.T
        Ainv_h = U @ torch.diag(1.0 / diag) @ U.T
        blocks.append(A_h)
        inv_blocks.append(Ainv_h)

    A = torch.block_diag(*blocks)  # [d_k, d_k]
    A_inv = torch.block_diag(*inv_blocks)  # [d_k, d_k]

    # Apply transformations
    Q_w, Q_b = A @ Q_w, A @ Q_b
    K_w, K_b = A_inv @ K_w, A_inv @ K_b
    V_w, V_b = A @ V_w, A @ V_b
    W_out = W_out @ A_inv  # Right multiply all columns

    # Write back to parameters
    W_in[:d_k], W_in[d_k:2 * d_k], W_in[2 * d_k:] = Q_w, K_w, V_w
    b_in[:d_k], b_in[d_k:2 * d_k], b_in[2 * d_k:] = Q_b, K_b, V_b
    params[f"{base}.in_proj_weight"] = W_in
    params[f"{base}.in_proj_bias"] = b_in
    params[f"{base}.out_proj.weight"] = W_out
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


def params(
    model_state_dict: dict,
    pretrained_state_dict: dict = None,
    num_layers: int = 12,
    enable_permutation: bool = True,
    enable_scaling: bool = True,
    perm_config: dict = None,
    scaling_config: dict = None,
    rng_seed: int = 0
):
    """
    Unified PaRaMS defense interface that applies parameter-level defenses to protect against model merging.
    
    This function combines two defense modules:
    1. MLP Parameter Rearrangement: Rearranges MLP parameters to disrupt merging
    2. Attention Head Scaling: Scales Q,K,V,W_out parameters in attention layers with multiple variants
    
    Attention Scaling Variants:
    - 'diagonal': Simple per-dimension diagonal scaling (fastest, basic protection)
      Uses independent scalar multipliers: Q' = diag(α)Q, K' = diag(1/α)K
      
    - 'symmetric': Block-diagonal symmetric positive definite matrices (balanced protection)
      Uses per-head symmetric matrices: Q' = AQ, K' = A⁻¹K where A = UΛU^T
      
    - 'nonsymmetric': Block-diagonal general invertible matrices (strongest protection)
      Uses per-head nonsymmetric matrices: Q' = AQ, K' = A⁻ᵀK with full rank A
    
    Args:
        model_state_dict (dict): The model's state dictionary to protect
        pretrained_state_dict (dict, optional): Pretrained model state dict (required for permutation)
        num_layers (int): Number of transformer layers (12 for ViT-B, 24 for ViT-L)
        enable_permutation (bool): Whether to apply MLP parameter rearrangement
        enable_scaling (bool): Whether to apply attention head scaling
        perm_config (dict, optional): Configuration for permutation module
            - max_iter (int): Maximum optimization iterations (default: 200)
            - obj (str): 'mismatching' for defense, 'matching' for alignment (default: 'mismatching')
            - tolerance (int): Early stopping tolerance (default: 10)
        scaling_config (dict, optional): Configuration for scaling module
            - scaling_type (str): 'diagonal', 'symmetric', or 'nonsymmetric' (default: 'diagonal')
            - scale_min (float): Minimum scaling factor (default: 0.05)
            - scale_max (float): Maximum scaling factor (default: 20.0)
            - mode (str): Sampling mode for diagonal - 'uniform', 'log_uniform', 'beta' (default: 'uniform')
            - n_heads (int): Number of attention heads for block methods (default: 12)
            - eye_eps (float): Identity regularization for nonsymmetric (default: 1e-2)
        rng_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Protected model state dictionary
        
    Raises:
        ValueError: If permutation is enabled but pretrained_state_dict is not provided
        ValueError: If unknown scaling_type is specified
        
    Examples:
        Basic usage with diagonal scaling:
        >>> protected_params = params(
        ...     model.state_dict(),
        ...     pretrained.state_dict(),
        ...     enable_permutation=True,
        ...     enable_scaling=True
        ... )
        
        Using symmetric block-diagonal scaling:
        >>> protected_params = params(
        ...     model.state_dict(),
        ...     pretrained.state_dict(),
        ...     enable_scaling=True,
        ...     scaling_config={
        ...         'scaling_type': 'symmetric',
        ...         'n_heads': 12,
        ...         'scale_min': 0.1,
        ...         'scale_max': 10.0
        ...     }
        ... )
        
        Maximum protection with nonsymmetric scaling:
        >>> protected_params = params(
        ...     model.state_dict(),
        ...     pretrained.state_dict(),
        ...     enable_permutation=True,
        ...     enable_scaling=True,
        ...     scaling_config={
        ...         'scaling_type': 'nonsymmetric',
        ...         'n_heads': 12,
        ...         'eye_eps': 1e-3
        ...     }
        ... )
    """
    
    # Default configurations
    if perm_config is None:
        perm_config = {
            'max_iter': 200,
            'obj': 'mismatching',
            'tolerance': 1
        }
    
    if scaling_config is None:
        scaling_config = {
            'scale_min': 0.05,
            'scale_max': 20.0,
            'mode': 'uniform',
            'scaling_type': 'diagonal',  # 'diagonal', 'symmetric', 'nonsymmetric'
            'n_heads': 12,  # Number of attention heads
            'eye_eps': 1e-2  # For nonsymmetric scaling
        }
    
    # Validate inputs
    if enable_permutation and pretrained_state_dict is None:
        raise ValueError("pretrained_state_dict is required when enable_permutation=True")
    
    # Clone the model parameters to avoid modifying the original
    protected_params = {name: param.clone() for name, param in model_state_dict.items()}
    
    # Apply MLP Parameter Rearrangement
    if enable_permutation:
        print("Applying MLP parameter rearrangement...")
        
        # Extract MLP parameters for both models
        victim_mlp_params = {name: param.clone() for name, param in model_state_dict.items() 
                           if 'mlp.c' in name}
        pretrained_mlp_params = {name: param.clone() for name, param in pretrained_state_dict.items() 
                               if 'mlp.c' in name}
        
        # Generate permutation specification
        perm_spec = vit_permutation_spec_MLP(num_layers=num_layers)
        
        # Apply parameter rearrangement
        rng = jax_random.PRNGKey(rng_seed)
        permutation, _, _, _ = parameter_rearrangement(
            rng, perm_spec, victim_mlp_params, pretrained_mlp_params,
            max_iter=perm_config['max_iter'],
            obj=perm_config['obj'],
            tolerance=perm_config['tolerance']
        )
        
        # Apply permutation to MLP parameters
        permuted_mlp_params = apply_permutation(perm_spec, permutation, victim_mlp_params)
        
        # Convert back to torch tensors and update protected parameters
        for k, v in permuted_mlp_params.items():
            protected_params[k] = torch.tensor(np.array(v))
    
    # Apply Attention Head Scaling
    if enable_scaling:
        scaling_type = scaling_config.get('scaling_type', 'diagonal')
        print(f"Applying attention head scaling ({scaling_type})...")
        
        # Generate random generator for scaling
        rng_gen = torch.Generator()
        rng_gen.manual_seed(rng_seed)
        
        # Select scaling function based on type
        if scaling_type == 'diagonal':
            scaling_func = apply_attention_qkvw_scaling
        elif scaling_type == 'symmetric':
            scaling_func = apply_attention_qkvw_blockdiag_sym
        elif scaling_type == 'nonsymmetric':
            scaling_func = apply_attention_qkvw_blockdiag_nonsym
        else:
            raise ValueError(f"Unknown scaling_type: {scaling_type}. Choose from 'diagonal', 'symmetric', 'nonsymmetric'")
            
        # Apply scaling to each transformer layer
        for layer_idx in range(num_layers):
            if scaling_type == 'diagonal':
                protected_params = scaling_func(
                    protected_params,
                    layer_idx=layer_idx,
                    scale_min=scaling_config['scale_min'],
                    scale_max=scaling_config['scale_max'],
                    rng=rng_gen,
                    mode=scaling_config.get('mode', 'uniform')
                )
            elif scaling_type == 'symmetric':
                protected_params = scaling_func(
                    protected_params,
                    layer_idx=layer_idx,
                    n_heads=scaling_config.get('n_heads', 12),
                    rng=rng_gen,
                    scale_min=scaling_config['scale_min'],
                    scale_max=scaling_config['scale_max']
                )
            elif scaling_type == 'nonsymmetric':
                protected_params = scaling_func(
                    protected_params,
                    layer_idx=layer_idx,
                    n_heads=scaling_config.get('n_heads', 12),
                    rng=rng_gen,
                    scale_min=scaling_config['scale_min'],
                    scale_max=scaling_config['scale_max'],
                    eye_eps=scaling_config.get('eye_eps', 1e-2)
                )
    
    print(f"PaRaMS protection applied successfully! (Permutation: {enable_permutation}, Scaling: {enable_scaling})")
    return protected_params
