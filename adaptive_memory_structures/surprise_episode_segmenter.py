"""
Surprise-based episode segmentation for HypergraphMemory.

Ports the token-level "surprise spike + graph-modularity refinement" boundary
detector from `Suprise Boundary Creator Module/boundary_creator.py` (and its
`similarity_refinement/` helper package) essentially unchanged — the surprisal
formula, the rolling-threshold spike test, and the modularity/conductance
boundary-snapping logic below are the same math, just inlined into one file so
they fit this directory's flat import style.

The one real adaptation is granularity: the original module streams raw
generation token-by-token and keeps a rolling multi-chunk history buffer. Here
we already have a fixed, already-written session transcript (a list of
`Page`s, each one atomic turn that can't be split mid-page), so `PageEpisodeSegmenter`
runs a single teacher-forced forward pass over the whole transcript, pools
token-level surprisal/Key-states to *per-page* values, and calls the same
`SurpriseBoundaryPipeline.compute_boundaries` / `refine_boundaries` statics
once as a one-shot batch (exc_length = number of pages) rather than through
`StatefulSurpriseBoundary`'s incremental streaming wrapper, which exists to
carry history across chunks we don't have here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from memory_structures import Page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# similarity_refinement/similarity.py -- ported verbatim
# ---------------------------------------------------------------------------

def modularity(A: torch.Tensor, communities: list[list[int]]) -> torch.Tensor:
    """
    Inputs:
        - A: Similarity matrix. Expected dimensions: (..., position, position)
        - communities: List of lists of position indices each defining a single community.
    Outputs:
        - Matrix of modularity values by layer and head. Dimensions: (layer, head)
    """
    ndims = len(A.shape)
    m = A.sum(dim=(-2, -1)) / 2

    k_i = A.sum(dim=-2)
    k_j = A.sum(dim=-1)
    expected_edges = torch.einsum('...i,...j->...ij', k_i, k_j)
    expected_edges /= (2 * m.unsqueeze(-1).unsqueeze(-1)) if ndims > 2 else (2 * m)

    Q = torch.zeros(A.shape[:-2]) if ndims > 2 else 0.0
    for community in communities:
        sub_A = A[..., community, :][..., :, community]
        sub_expected_edges = expected_edges[..., community, :][..., :, community]
        Q += (sub_A - sub_expected_edges).sum(dim=(-2, -1)).cpu()

    return Q / (4 * m.cpu())


def conductance(A: torch.Tensor, communities: list[list[int]]):
    conductance_vals = []
    total_vol = torch.sum(A, dim=(-2, -1))
    for community in communities:
        community_bar = [i for i in range(A.shape[-1]) if i not in community]
        cut_edges = torch.sum(A[..., community, :][..., :, community_bar], dim=(-2, -1))

        vol_S = torch.sum(A[..., community, :][..., :, community], dim=(-2, -1))
        vol_S_bar = total_vol - vol_S

        min_vol = torch.minimum(vol_S, vol_S_bar)
        min_vol[min_vol == 0] = 1e-15

        conductance_vals.append(cut_edges / min_vol)
        del cut_edges, vol_S, vol_S_bar

    conductance_vals = torch.stack(conductance_vals)
    if len(A.shape) > 2:
        min_conductance = conductance_vals[torch.argmin(conductance_vals.sum(dim=(-2, -1)))]
        max_conductance = conductance_vals[torch.argmax(conductance_vals.sum(dim=(-2, -1)))]
        mean_conductance = conductance_vals.mean(dim=0)
    else:
        min_conductance = conductance_vals.min()
        max_conductance = conductance_vals.max()
        mean_conductance = conductance_vals.mean()

    return min_conductance, max_conductance, mean_conductance, conductance_vals


def intra_inter_sim(A: torch.Tensor, communities: list[list[int]], return_mean: bool = True):
    inter, intra = [], []
    for community in communities:
        intra.append(torch.mean(A[..., community, :][..., :, community], dim=(-2, -1)))
        community_bar = [i for i in range(A.shape[-1]) if i not in community]
        inter.append(torch.mean(A[..., community, :][..., :, community_bar], dim=(-2, -1)))

    ratio = [i / j for i, j in zip(intra, inter)]

    if return_mean:
        intra = torch.mean(torch.stack(intra), dim=0)
        inter = torch.mean(torch.stack(inter), dim=0)
        ratio = torch.mean(torch.stack(ratio), dim=0)

    return ratio, intra, inter


def calc_adjacent_similarity_with_offset(A: torch.Tensor, first_indx: int, last_indx: int, sim_func=modularity):
    if first_indx > last_indx or first_indx > A.shape[0] or last_indx > A.shape[0]:
        raise ValueError(f'Problem with indices in similarity calculation: {first_indx}, {last_indx}, {A.shape}')
    T = A.shape[0]
    result = torch.zeros(last_indx - first_indx)
    for t in range(first_indx, last_indx):
        communities = [list(range(0, t)), list(range(t, T))]
        result[t - first_indx] = sim_func(A, communities)
    return result


# ---------------------------------------------------------------------------
# similarity_refinement/segmentation.py -- ported verbatim
# ---------------------------------------------------------------------------

def events_with_similarity_adjustment(events_base, A: torch.Tensor, similarity_metric: str = 'modularity',
                                       min_size: int = 0, offset: int = 0):
    events_temp = [0]
    events_base = [i + offset for i in events_base]
    events_base_ = [events_base[0]] if events_base[0] >= min_size or offset == 0 else []
    events_base = events_base_ + [events_base[i] for i in range(1, len(events_base)) if events_base[i] - events_base[i - 1] >= min_size]

    if similarity_metric == 'modularity':
        sim_func = modularity
    elif similarity_metric == 'conductance':
        sim_func = lambda a, c: conductance(a, c)[0]
    elif similarity_metric == 'intra_inter_sim':
        sim_func = lambda a, c: intra_inter_sim(a, c)[0]
    else:
        raise NotImplementedError(f'Similarity metric {similarity_metric} not implemented')

    for event in events_base:
        if event - events_temp[-1] > min_size:
            if event - offset > min_size:
                original_event_size = event - events_temp[-1]
                half_size = int(original_event_size / 2)
                start_from = max(0, events_temp[-1] - half_size)
                end_to = min(A.shape[0], event + half_size)
                first_indx_to_check = max(offset - start_from, events_temp[-1] - start_from)
                last_indx_to_check = event - start_from

                TI_LES = torch.clone(A[start_from:end_to, :][:, start_from:end_to])
                adj_mod = calc_adjacent_similarity_with_offset(TI_LES, first_indx_to_check, last_indx_to_check, sim_func=sim_func)
                if similarity_metric == 'conductance':
                    arg_mod = torch.argmin(adj_mod[min_size:])
                else:
                    arg_mod = torch.argmax(adj_mod[min_size:])
                events_temp.append(start_from + first_indx_to_check + min_size + arg_mod)
            else:
                events_temp.append(event)
        elif event - events_temp[-1] == min_size or offset == 0:
            events_temp.append(event)
        else:
            raise ValueError(f'Problem with event size: {event}')

    events_temp = [(i.item() if hasattr(i, 'item') else i) - (offset.item() if hasattr(offset, 'item') else offset) for i in events_temp[1:]]
    assert len(events_temp) == len(events_base), (
        f'Problem with refinement: does not have the same number of events: {len(events_temp)}, {len(events_base)}'
    )
    return events_temp


# ---------------------------------------------------------------------------
# boundary_creator.py -- ported near-verbatim (SurpriseBoundaryPipeline only;
# StatefulSurpriseBoundary's incremental history buffers aren't needed since
# PageEpisodeSegmenter processes a whole session transcript in one shot)
# ---------------------------------------------------------------------------

class SurpriseBoundaryPipeline:
    """Isolated surprise + graph-theoretic boundary refinement math."""

    @staticmethod
    def compute_surprisal(logits: torch.Tensor, em_labels: torch.Tensor) -> torch.Tensor:
        """Token surprisal from next-token logits and target labels."""
        if em_labels.dtype != torch.bool:
            prob = torch.softmax(logits, dim=-1)
            surprisal = -torch.log(torch.gather(prob, dim=-1, index=em_labels.unsqueeze(-1))).squeeze(-1)
            return surprisal
        return em_labels

    @staticmethod
    def compute_boundaries(
        exc_length: int,
        surprisal: torch.Tensor,
        global_remainder_surprisal: torch.Tensor,
        global_remainder_ed: int,
        n_local: int,
        n_init: int,
        surprisal_threshold_gamma: float,
        uniform_blocks: bool = False,
        max_block_size: int = 128,
    ) -> torch.Tensor:
        """Boolean boundary indicators from the surprise threshold over historical context."""
        if uniform_blocks:
            divide = torch.zeros(surprisal.shape, dtype=torch.bool, device=surprisal.device)
            divide[:, ::max_block_size] = True
            return divide

        if global_remainder_ed <= exc_length:
            std = torch.std(surprisal, dim=-1)
            mean = torch.mean(surprisal, dim=-1)
            divide = surprisal > surprisal_threshold_gamma * std.unsqueeze(-1) + mean.unsqueeze(-1)
        else:
            avg_st = max(global_remainder_ed - n_local, 0)
            avg_ed = max(global_remainder_ed - exc_length, n_init)

            history = global_remainder_surprisal[:, avg_st:avg_ed]
            std = torch.std(history, dim=-1)
            mean = torch.mean(history, dim=-1)
            divide = surprisal > surprisal_threshold_gamma * std.unsqueeze(-1) + mean.unsqueeze(-1)

        return divide

    @staticmethod
    def refine_boundaries(
        divide: torch.Tensor,
        exc_length: int,
        global_remainder_ed: int,
        global_remainder_len: int,
        global_block_divide: torch.Tensor,
        K_states_for_refinement: list[torch.Tensor],
        refine_with_buffer: bool,
        similarity_metric: str,
        min_block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Refine surprise boundaries using cosine-similarity-like Key-state graphs.

        K_states_for_refinement: per-layer Key tensors, each
        (batch, num_heads, full_seq_len, dim_head).
        """
        batch_size = divide.shape[0]

        if global_remainder_len >= 2 * exc_length and refine_with_buffer:
            last_divide = torch.zeros(batch_size)
            for u in range(batch_size):
                last_events = torch.where(global_block_divide[u, global_remainder_ed - 2 * exc_length: global_remainder_ed - exc_length] > 0)[0]
                if len(last_events) == 0:
                    last_divide[u] = exc_length
                else:
                    last_divide[u] = last_events[-1]
            offsets = exc_length - last_divide.int()
            max_offset = torch.max(offsets).item()
        else:
            offsets = torch.zeros(batch_size, dtype=torch.int)
            max_offset = 0

        slice_len = exc_length + max_offset

        K = torch.clone(K_states_for_refinement[0][:, :, -slice_len:, :]).unsqueeze(dim=1).to(device)
        for l in range(1, len(K_states_for_refinement)):
            layer_k = torch.clone(K_states_for_refinement[l][:, :, -slice_len:, :]).unsqueeze(dim=1).to(device)
            K = torch.cat((K, layer_k), dim=1)

        stacked_A = torch.einsum('blhtd,blhTd->btT', K, K).detach()
        del K

        try:
            stacked_A = stacked_A.to(device)
        except Exception as e:
            print(f'Tried casting stacked_A to device, but failed with error: {e}')

        for u in range(batch_size):
            events_sur = torch.where(divide[u] > 0)[0]
            if len(events_sur) > 0:
                off_u = offsets[u].item()
                A_u = stacked_A[u][max_offset - off_u:, max_offset - off_u:]

                events_sur_mod = events_with_similarity_adjustment(
                    events_sur, A_u, similarity_metric=similarity_metric,
                    min_size=min_block_size, offset=off_u,
                )

                divide[u] = torch.zeros_like(divide[u])
                divide[u][events_sur_mod] = True

        del stacked_A
        return divide


# ---------------------------------------------------------------------------
# New: page-granularity adaptation
# ---------------------------------------------------------------------------

def _extract_layer_keys(past_key_values: Any) -> list[torch.Tensor] | None:
    """
    Return a list of per-layer Key tensors, each
    (batch, num_kv_heads, seq_len, head_dim), from whatever shape HF's
    `use_cache=True` output takes -- this has changed across transformers
    versions:
      - newest (Cache object with `.layers`, each holding `.keys`/`.values`)
      - `DynamicCache` with `.to_legacy_cache()` -> tuple[(key, value), ...]
      - `DynamicCache` with `.key_cache`/`.value_cache` list attributes
      - legacy plain tuple[(key, value), ...] (older transformers, no Cache
        object at all)
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "layers"):
        return [layer.keys for layer in past_key_values.layers]
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        if legacy:
            return [kv[0] for kv in legacy]
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)
    try:
        return [kv[0] for kv in past_key_values]
    except (TypeError, IndexError):
        return None


class PageEpisodeSegmenter:
    """
    Splits a session's Pages into contiguous episode groups using the surprise
    + graph-modularity mechanism above, pooled to per-page granularity.

    Defaults mirror `CAIMMSBoundaryEmitter`'s (gamma=1.5, n_local=4096,
    n_init=128, similarity_refinement=True, min_block_size=8).
    """

    def __init__(
        self,
        gamma: float = 1.5,
        n_local: int = 4096,
        n_init: int = 128,
        min_block_size: int = 8,
        similarity_refinement: bool = True,
        similarity_metric: str = "modularity",
        refine_with_buffer: bool = True,
        refine_layer_fraction: float = 0.5,
    ):
        self.gamma = gamma
        self.n_local = n_local
        self.n_init = n_init
        self.min_block_size = min_block_size
        self.similarity_refinement = similarity_refinement
        self.similarity_metric = similarity_metric
        self.refine_with_buffer = refine_with_buffer
        self.refine_layer_fraction = refine_layer_fraction

    def segment(self, pages: list[Page], model: Any, tokenizer: Any) -> list[list[Page]]:
        """
        Return a list of contiguous page-groups (episodes), in original order.
        Falls back to "whole session = one episode" when there's no usable
        model/tokenizer or too few pages to meaningfully segment.
        """
        if not pages:
            return []
        if len(pages) < 2 or model is None or tokenizer is None:
            return [list(pages)]

        try:
            return self._segment_with_model(pages, model, tokenizer)
        except Exception:
            # Segmentation is a quality enhancement, not a correctness
            # requirement -- never let a tokenizer/model quirk break indexing.
            # Logged (not silent) so a real bug here doesn't masquerade as a
            # session that's simply never segmenting.
            logger.warning("PageEpisodeSegmenter: falling back to a single episode after an error", exc_info=True)
            return [list(pages)]

    def _segment_with_model(self, pages: list[Page], model: Any, tokenizer: Any) -> list[list[Page]]:
        device = next(model.parameters()).device

        input_ids_list: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for page in pages:
            text = page.to_text() or " "
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if ids.shape[0] == 0:
                ids = tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            input_ids_list.append(ids)
            spans.append((offset, offset + ids.shape[0]))
            offset += ids.shape[0]

        n_pages = len(pages)
        if offset < 2:
            return [list(pages)]

        full_ids = torch.cat(input_ids_list).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_ids=full_ids, use_cache=True, output_hidden_states=False)

        logits = out.logits
        pred_logits = logits[:, :-1, :]
        target_ids = full_ids[:, 1:]
        token_surprisal = SurpriseBoundaryPipeline.compute_surprisal(pred_logits, target_ids)
        # pad a zero for token 0, which has no predecessor to be surprising against
        token_surprisal_full = torch.cat(
            [torch.zeros(1, 1, device=token_surprisal.device, dtype=token_surprisal.dtype), token_surprisal],
            dim=-1,
        )

        page_surprisal = torch.zeros(1, n_pages, device=device)
        for i, (st, ed) in enumerate(spans):
            if ed > st:
                page_surprisal[0, i] = token_surprisal_full[0, st:ed].mean()

        divide = SurpriseBoundaryPipeline.compute_boundaries(
            exc_length=n_pages,
            surprisal=page_surprisal,
            global_remainder_surprisal=page_surprisal,
            global_remainder_ed=n_pages,
            n_local=self.n_local,
            n_init=self.n_init,
            surprisal_threshold_gamma=self.gamma,
        )

        if self.similarity_refinement:
            layer_keys = _extract_layer_keys(getattr(out, "past_key_values", None))
            if layer_keys:
                num_layers = len(layer_keys)
                st_layer = max(0, int(num_layers * (1 - self.refine_layer_fraction)))
                K_states_page = []
                for layer_idx in range(st_layer, num_layers):
                    key = layer_keys[layer_idx]  # (1, heads, seq_len, head_dim)
                    pooled = torch.zeros(1, key.shape[1], n_pages, key.shape[-1], device=device)
                    for i, (st, ed) in enumerate(spans):
                        if ed > st:
                            pooled[:, :, i, :] = key[:, :, st:ed, :].mean(dim=2)
                    K_states_page.append(pooled)

                if K_states_page:
                    divide = SurpriseBoundaryPipeline.refine_boundaries(
                        divide=divide,
                        exc_length=n_pages,
                        global_remainder_ed=n_pages,
                        global_remainder_len=n_pages,
                        global_block_divide=divide,
                        K_states_for_refinement=K_states_page,
                        refine_with_buffer=self.refine_with_buffer,
                        similarity_metric=self.similarity_metric,
                        min_block_size=self.min_block_size,
                        device=device,
                    )

        boundary_flags = divide[0].tolist()

        episodes: list[list[Page]] = []
        current: list[Page] = [pages[0]]
        for i in range(1, n_pages):
            if boundary_flags[i]:
                episodes.append(current)
                current = []
            current.append(pages[i])
        episodes.append(current)
        return episodes
