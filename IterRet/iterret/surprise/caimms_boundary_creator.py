import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import the core stateful pipeline
from .boundary_creator import StatefulSurpriseBoundary


def _extract_layer_keys(past_key_values) -> List[torch.Tensor]:
    """Return per-layer Key tensors, shape (batch, n_kv_heads, seq, head_dim).

    Handles the API drift across transformers versions:
    - >= ~4.5x / 5.x: a ``Cache`` object (e.g. ``DynamicCache``) that is NOT
      subscriptable; keys live at ``.layers[i].keys``.
    - intermediate: a ``Cache`` exposing a ``.key_cache`` list.
    - legacy: a tuple/list of ``(key, value)`` pairs, one per layer.
    """
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return [layer.keys for layer in layers]
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        return list(key_cache)
    return [layer[0] for layer in past_key_values]

class CAIMMSBoundaryEmitter(nn.Module):
    """
    A lightweight wrapper that wraps around a causal LM (e.g., Qwen3).
    It intercepts logits and past_key_values to compute Surprise-Based Episodic Segmentation
    (as described in C-AIMMS Section 1.3.1).
    
    Unlike standard EM-LLM models, this emitter DOES NOT store the KV-cache of episodes.
    Instead, it emits the start/end absolute token indices of new episodes so that the
    C-AIMMS orchestrator can encode them into HETREP (Hypergraph, Canvas, Vector Store).
    """
    def __init__(
        self, 
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct",
        batch_size: int = 1,
        surprisal_threshold_gamma: float = 1.5,
        n_local: int = 4096,
        n_init: int = 128,
        similarity_refinement: bool = True,
        refine_with_buffer: bool = True,
        similarity_metric: str = 'modularity',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()

        # 1. Load the base model. fp16 matmul isn't implemented on CPU, so only
        #    use it on accelerators (CUDA/MPS) and fall back to fp32 on CPU. Load
        #    then .to(device) rather than device_map= (which is an accelerate
        #    feature that doesn't take a plain torch.device / "mps").
        if isinstance(device, str):
            device = torch.device(device)
        # bf16 on CUDA: Qwen-family activations can overflow fp16. MPS keeps
        # fp16 (bf16 support there is partial); CPU needs fp32.
        compute_dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type, torch.float32)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=compute_dtype,
        )
        self.base_model.to(device)
        self.device = device
        
        # 2. Initialize the Stateful Surprise Boundary Pipeline
        self.boundary_tracker = StatefulSurpriseBoundary(
            batch_size=batch_size,
            surprisal_threshold_gamma=surprisal_threshold_gamma,
            n_local=n_local,
            n_init=n_init,
            similarity_refinement=similarity_refinement,
            similarity_metric=similarity_metric,
            device=device,
            dtype=torch.float32
        )
        
        self.last_boundary_idx = 0
        self.disable_boundaries = False

        # A queue of newly detected episodic bounds: (start_idx, end_idx)
        self.new_episodes: List[Tuple[int, int]] = []

        # Last logit row from the previous forward() call. logits[-1] of one call
        # is the distribution that predicts the FIRST token of the next call, so
        # carrying it keeps the EM-LLM surprise alignment correct across chunk
        # seams (see forward()). None at the start of a fresh stream.
        self._prev_last_logit: Optional[torch.Tensor] = None
        
    def get_new_episodes(self) -> List[Tuple[int, int]]:
        """
        Consumes and returns the list of newly detected episode boundaries.
        The C-AIMMS orchestrator should call this after generation steps to 
        extract chunks for HETREP encoding.
        """
        episodes = self.new_episodes.copy()
        self.new_episodes.clear()
        return episodes

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None, 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        # 1. Standard Forward Pass to get logits and K_states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True, 
            return_dict=True,
            **kwargs
        )
        
        if self.disable_boundaries:
            return outputs
        
        logits = outputs.logits
        loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.base_model.config.vocab_size), shift_labels.view(-1))

            # EM-LLM surprise (arXiv:2407.09450): the surprise of token x_t is
            #   -log P(x_t | x_<t)
            # i.e. how unexpected the RECEIVED token was given everything before
            # it. A causal LM's logits[t] is the distribution over the *next*
            # token, P(.|x_0..x_t), so the distribution that actually predicted
            # x_t is logits[t-1]. Shift the logits right by one and gather the
            # real tokens (em_labels = labels): then surprisal[t] = -log P(x_t |
            # x_<t), aligned to token position t. The previous call's last logit
            # predicted x_0 of this call, so carrying it keeps chunk seams exact;
            # at the very start of a stream x_0 is unconditioned (no surprise).
            surprise_logits = torch.empty_like(logits)
            surprise_logits[:, 1:, :] = logits[:, :-1, :]
            if self._prev_last_logit is not None:
                surprise_logits[:, :1, :] = self._prev_last_logit
            else:
                # Stream start: x_0 has no predictor. Its own logits are a
                # harmless placeholder -- a boundary at token 0 is meaningless
                # (it is absorbed into the first event downstream).
                surprise_logits[:, :1, :] = logits[:, :1, :]
            self._prev_last_logit = logits[:, -1:, :].detach()

            em_labels = labels
            boundary_logits = surprise_logits
        else:
            # Pure generation: the "received" token isn't known ahead of time,
            # so fall back to the model's own most-likely token, scored in place.
            em_labels = torch.argmax(logits, dim=-1)
            boundary_logits = logits

        # 2. Extract K_states purely for mathematical similarity refinement
        K_states_for_refinement = None
        if self.boundary_tracker.similarity_refinement and outputs.past_key_values is not None:
            K_states_for_refinement = []
            refine_st_layer = 20

            layer_keys = _extract_layer_keys(outputs.past_key_values)
            num_layers = len(layer_keys)

            input_length = input_ids.shape[-1]
            uncommitted_length = self.boundary_tracker.global_remainder_ed - self.boundary_tracker.global_remainder_st
            required_k_len = uncommitted_length + input_length

            for layer_idx in range(min(refine_st_layer, num_layers-1), num_layers):
                k_state = layer_keys[layer_idx]
                # float32: the refinement builds a Gram matrix via einsum(K, K);
                # in fp16 those key dot-products overflow to inf and the modularity
                # scores become garbage (boundaries collapse to one giant event).
                k_state_chunk = k_state[..., -required_k_len:, :].float()
                K_states_for_refinement.append(k_state_chunk)
                
        # 3. Step the boundary pipeline (on surprise-aligned logits, so the
        #    surprisal it computes is -log P(x_t | x_<t), not -log P(x_t | x_<=t))
        divide = self.boundary_tracker.step(
            logits=boundary_logits,
            em_labels=em_labels,
            K_states_for_refinement=K_states_for_refinement
        )
        
        # 4. Extract Episode Indices (DO NOT STORE KV TENSORS)
        for b in range(divide.shape[0]):
            boundary_indices = torch.where(divide[b] > 0)[0]
            if len(boundary_indices) > 0:
                for b_idx in boundary_indices:
                    absolute_b_idx = self.boundary_tracker.global_remainder_st + b_idx.item()
                    
                    if absolute_b_idx > self.last_boundary_idx:
                        # Append the slice indices for the new episode
                        self.new_episodes.append((self.last_boundary_idx, absolute_b_idx))
                        self.last_boundary_idx = absolute_b_idx
                        
        # 5. Shift boundary history to discard committed tokens
        num_tokens_to_remove = self.last_boundary_idx - self.boundary_tracker.global_remainder_st
        if num_tokens_to_remove > 0:
            self.boundary_tracker.shift_history(num_tokens_to_remove)
                
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
