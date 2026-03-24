# mHC Implementation Pattern

Use this reference when porting mHC into a new decoder-only model.

## Contents

- Shapes
- Mapping formulas
- Generic module sketch
- Decoder block integration
- Model-level integration
- Initialization rules
- Notebook-vs-production note
- Common failure modes

## Shapes

- Base hidden state: `[B, S, D]`
- Expanded residual state: `[B, S, N, D]`
- Flattened residual state: `[B, S, N * D]`
- Mapping logits size: `N^2 + 2N`
- `pre_mapping`: `[B, S, N]`
- `post_mapping`: `[B, S, N]`
- `residual_mapping`: `[B, S, N, N]`

## Mapping formulas

Given flattened residual states `x_flat`:

- `pre_logits = proj(norm(x_flat))[..., :N] * alpha[0] + bias[:N]`
- `post_logits = proj(norm(x_flat))[..., N:2N] * alpha[1] + bias[N:2N]`
- `res_logits = proj(norm(x_flat))[..., 2N:] * alpha[2] + bias[2N:]`
- `pre_mapping = sigmoid(pre_logits)`
- `post_mapping = 2 * sigmoid(post_logits)`
- `residual_mapping = sinkhorn(res_logits.reshape(B, S, N, N))`

Use `float32` inside the mapping path and cast back to the model dtype only when combining with hidden states.

## Generic module sketch

```python
class ManifoldHyperConnection(nn.Module):
    def __init__(self, config):
        self.num_streams = config.mhc_expansion_rate
        self.flat_hidden_size = config.hidden_size * self.num_streams
        self.mapping_size = self.num_streams * self.num_streams + 2 * self.num_streams
        self.norm = RMSNorm(self.flat_hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(self.flat_hidden_size, self.mapping_size, bias=False)
        self.alpha = nn.Parameter(torch.empty(3))
        self.bias = nn.Parameter(torch.empty(self.mapping_size))

    def compute_mappings(self, hidden_states):
        flat = hidden_states.reshape(batch, seq, self.flat_hidden_size)
        logits = self.proj(self.norm(flat)).to(torch.float32)
        ...

    def prepare_layer_input(self, hidden_states):
        pre, post, residual = self.compute_mappings(hidden_states)
        layer_input = torch.einsum("bsn,bsnd->bsd", pre.to(hidden_states.dtype), hidden_states)
        residual_states = torch.einsum("bsnm,bsmd->bsnd", residual.to(hidden_states.dtype), hidden_states)
        return layer_input, residual_states, post

    def merge_layer_output(self, residual_states, layer_output, post):
        post_states = post.to(layer_output.dtype).unsqueeze(-1) @ layer_output.unsqueeze(-2)
        return residual_states + post_states
```

## Decoder block integration

Keep the original path intact:

```python
if not self.use_mhc:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(...)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states
```

Add the mHC path in the same block:

```python
layer_input, residual_states, post = self.attn_mhc.prepare_layer_input(hidden_states)
layer_input = self.input_layernorm(layer_input)
attn_output, _ = self.self_attn(hidden_states=layer_input, ...)
hidden_states = self.attn_mhc.merge_layer_output(residual_states, attn_output, post)

layer_input, residual_states, post = self.mlp_mhc.prepare_layer_input(hidden_states)
layer_input = self.post_attention_layernorm(layer_input)
mlp_output = self.mlp(layer_input)
hidden_states = self.mlp_mhc.merge_layer_output(residual_states, mlp_output, post)
```

Instantiate separate mHC modules for attention and MLP.

## Model-level integration

At the model boundary:

```python
hidden_states = inputs_embeds
position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

if self.use_mhc:
    hidden_states = hidden_states.unsqueeze(-2).repeat(1, 1, self.mhc_expansion_rate, 1)

for layer in self.layers:
    hidden_states = layer(hidden_states, position_embeddings=position_embeddings, ...)

if self.use_mhc:
    hidden_states = hidden_states.mean(dim=-2)  # or sum(dim=-2)

hidden_states = self.norm(hidden_states)
```

Expand after embeddings, reduce before final norm, and keep task heads unchanged.

## Initialization rules

Use a dedicated reset method for the mHC module and call it from model `_init_weights`.

- Fill `alpha` with `mhc_gating_factor_init`, typically `0.01`.
- Set `pre` bias to `-log(N - 1)` so `sigmoid(pre_bias) ~= 1 / N`. If `N == 1`, use a large positive value such as `10.0`.
- Leave `post` bias at `0` so `2 * sigmoid(0) = 1`.
- Initialize the residual bias matrix near identity, for example diagonal `4.0` and off-diagonal `-4.0`, then apply Sinkhorn.

This starts the model close to "keep each stream, add one layer output" behavior.

## Notebook-vs-production note

The notebook uses a fused RMSNorm trick:

- scale by RMSNorm gamma before the projection
- divide by the residual RMS after the projection

For a bias-free projection this is mathematically equivalent to `proj(RMSNorm(x))`. Use the simpler form first unless you are explicitly optimizing the mapping kernel.

## Common failure modes

- Attention shape errors: expanded `[B, S, N, D]` leaked into attention instead of reduced `[B, S, D]`.
- Generation or cache breakage: cache metadata or RoPE inputs were derived from the expanded tensor.
- Immediate loss spikes: initialization is not near identity or reduction uses `sum` where checkpoint scale expects `mean`.
- Non-stochastic residual matrices: Sinkhorn runs in low precision or lacks row and column normalization clamps.
