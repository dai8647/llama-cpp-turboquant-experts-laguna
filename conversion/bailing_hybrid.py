from __future__ import annotations

import math
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("BailingMoeV3Model", "BailingMoeV3ForCausalLM")
class BailingHybridModel(TextModel):
    """Ling 3.0 flash (inclusionAI): hybrid KDA + gated MLA MoE, `bailing_hybrid`.

    NOT the `bailingmoe2` stack (Ling 2.0) despite the shared family name -- only
    the MoE router carries over. The KDA block comes from Kimi Linear and the MLA
    block from Kimi's no-Q-compression variant, so conversion mirrors
    conversion/kimi_linear.py. The differences that matter here:

      * The attention module is `attention.`, not `self_attn.`. None of Kimi's
        tensor mappings match; bailing-hybrid entries were added alongside them.

      * `attention.g_proj.weight` exists on BOTH layer types with different
        shapes and different meanings: on KDA layers it is the full-rank output
        gate {n_embd, d_inner}, on MLA layers the head-wise attention gate
        {n_embd, n_head}. A name->enum table cannot express that, so the KDA one
        is renamed to `g_full_proj` here. Without this the loader silently binds
        the wrong tensor.

      * A_log is stored as +exp(A_log), NOT Kimi's -exp(A_log). config sets
        kda_safe_gate=true, which changes the decay to
            g = kda_lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias))
        so the sign lives in kda_lower_bound (-5.0), written as a KV below.
        Verified against fla ops/kda/gate.py (naive ref and Triton kernel agree).

      * `no_kda_lora: true` -> full-rank f_proj / g_proj, so SSM_F / SSM_G
        replace Kimi's SSM_F_{A,B} / SSM_G_{A,B} pairs.

    Config fields that look load-bearing and are NOT (verified by grepping
    modeling_bailing_moe_v3.py): expert_swiglu_limit_list and
    share_expert_swiglu_limit_list (populated with non-zero values for the last
    few layers, yet BailingMoeV3MLP.forward is a plain SwiGLU), use_qk_norm,
    linear_silu, group_norm_size, max_window_layers, mtp_use_kda, use_mla_nope,
    use_nGPT, scale_router_input, seq_aux. partial_rotary_factor is overwritten
    to 1.0 by the rotary module itself, so rotary_dim == qk_rope_head_dim.
    """

    model_arch = gguf.MODEL_ARCH.BAILING_HYBRID

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the MTP/nextn head is a real block in the checkpoint; include it unless
        # --no-mtp, matching glm/command_r. llama.cpp marks it TENSOR_SKIP.
        if (n_nextn := int(self.hparams.get("num_nextn_predict_layers", 0) or 0)) > 0 and not self.no_mtp:
            self.block_count = self.hparams["num_hidden_layers"] + n_nextn
            # tensor_map was built from the old block_count in super().__init__(),
            # so it must be rebuilt or every layer-42 tensor fails to map.
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def _is_kda_layer(self, bid: int) -> bool:
        """KDA everywhere except the last layer of each group, and the MTP head.

        Mirrors modeling_bailing_moe_v3.py:1006 --
            MLA if (layer_idx + 1) % layer_group_size == 0
                 or layer_idx >= num_hidden_layers // group_size * group_size
        The second clause is what puts the MTP head (layer 42) on MLA.
        """
        group = self.hparams["layer_group_size"]
        n_layer = self.hparams["num_hidden_layers"]
        is_mla = ((bid + 1) % group == 0) or (bid >= n_layer // group * group)
        return not is_mla

    def set_gguf_parameters(self):
        hparams = self.hparams

        # MLA KV cache requires the attention be converted to MQA (1 KV group).
        hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        assert hparams.get("no_kda_lora"), \
            "no_kda_lora is false: this checkpoint uses low-rank KDA gates, which " \
            "map to SSM_F_A/SSM_F_B (kimi-linear), not SSM_F/SSM_G"
        assert hparams.get("kda_safe_gate"), \
            "kda_safe_gate is false: llama.cpp's bailing-hybrid graph only implements " \
            "the safe-gate decay form"

        # Per-layer KV head count: 0 marks a KDA (recurrent) layer, which is how
        # llama.cpp tells the two branches apart.
        # NOTE: this array must be block_count long, NOT num_hidden_layers -- the
        # loader validates it against n_layer_all and rejects the model outright
        # if the MTP/nextn block has no entry. The MTP head is MLA, so it gets 1.
        _num_kv_heads = [0 if self._is_kda_layer(il) else 1 for il in range(self.block_count)]
        assert any(_num_kv_heads), "no MLA layers found -- layer_group_size indexing is wrong"
        assert len(_num_kv_heads) == self.block_count
        self.gguf_writer.add_head_count_kv(_num_kv_heads)
        logger.info(f"bailing-hybrid: {sum(1 for x in _num_kv_heads if x)} MLA / "
                    f"{sum(1 for x in _num_kv_heads if not x)} KDA layers")

        # ---- KDA ----
        self.gguf_writer.add_ssm_conv_kernel(hparams["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(hparams["head_dim"])
        self.gguf_writer.add_kda_lower_bound(float(hparams["kda_lower_bound"]))

        # ---- MLA ----
        # q_lora_rank is null (no Q compression), so add_q_lora_rank is skipped.
        assert hparams.get("q_lora_rank") is None, \
            "q_lora_rank is set: the graph builds a single wide q_proj and has no q_a/q_b path"
        kv_lora_rank     = hparams["kv_lora_rank"]
        qk_rope_head_dim = hparams["qk_rope_head_dim"]
        qk_nope_head_dim = hparams["qk_nope_head_dim"]
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        # ---- MoE (noaux_tc grouped top-k, bit-exact bailingmoe2) ----
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_group_count(hparams["n_group"])
        self.gguf_writer.add_expert_group_used_count(hparams["topk_group"])

        score = hparams.get("score_function", hparams.get("scoring_func"))
        assert score == "sigmoid", f"unexpected router score function {score!r}"
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        if (n_nextn := int(hparams.get("num_nextn_predict_layers", 0) or 0)) > 0 and not self.no_mtp:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # KDA conv1d: HF [d_inner, d_conv] -> numpy (1, d_inner, 1, d_conv),
        # which GGUF reverses into ggml ne = [d_conv, 1, d_inner, 1]. Memory
        # layout is preserved either way (d_conv changes fastest).
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
            elif data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # A_log is 1-D [n_head] here (kimi's is [1,H,1,1], solar_open2's
        # [1,1,64,1] -- third layout in three ports). Store +exp(A_log): the
        # negation lives in kda_lower_bound, unlike kimi which bakes in -exp().
        if name.endswith(".A_log"):
            data_torch = torch.exp(data_torch.float())
            data_torch = data_torch.reshape(1, 1, -1, 1)

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # llama.cpp asks for `blk.N.exp_probs_b.bias`, but `mlp.gate.expert_bias`
        # has no .weight/.bias suffix for map_tensor_name to strip, so it would be
        # written as a bare `blk.N.exp_probs_b` and the load fails on a missing
        # tensor. Same fix as bailingmoe/afmoe/grovemoe.
        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        # Disambiguate the two g_proj tensors (see the class docstring).
        if name.endswith(".attention.g_proj.weight"):
            assert bid is not None
            if self._is_kda_layer(bid):
                name = name.replace(".attention.g_proj.", ".attention.g_full_proj.")

        # merge the routed experts into one 3-D tensor per projection
        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for wid, tname in [("gate_proj", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                   ("down_proj", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                   ("up_proj",   gguf.MODEL_TENSOR.FFN_UP_EXP)]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    new_name = self.format_tensor_name(tname, bid)
                    yield from super().modify_tensors(data_torch, new_name, bid)
            return

        # MLA absorption needs kv_b split, with k_b transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")
            n_head_kv        = self.hparams["num_attention_heads"]
            v_head_dim       = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
