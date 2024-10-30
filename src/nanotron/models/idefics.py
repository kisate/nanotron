import torch

from typing import Dict, Optional, Union
from torch import nn

from nanotron import logging
from nanotron.config.config import Config
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models.base import NanotronModel
from nanotron.nn.layer_norm import TritonLayerNorm
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelEmbedding, TensorParallelRowLinear
from nanotron.distributed import dist
from nanotron.config import Idefics3VisionConfig, Idefics3Config
from nanotron.generation.generate_store import AttachableStore
from nanotron.random import RandomStates, branch_random_state
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method
from nanotron.models.llama import GLUActivation, LlamaModel, pad_to_right, Loss



logger = logging.get_logger(__name__)


class VisionEmbedding(nn.Module, AttachableStore):
    """
    Sharded implementation of the Idefics3VisionEmbeddings from huggingface for nanotron. Uses CLIPVit for megatron as a reference.
    """
    def __init__(self, tp_pg: dist.ProcessGroup, config: Idefics3Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.tp_pg = tp_pg
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_positions = self.num_patches

        self.position_embedding = TensorParallelEmbedding(
            num_embeddings=self.num_positions,
            embedding_dim=self.embed_dim,
            padding_idx=config.llama_config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> Dict[str, torch.Tensor]:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)

        embeddings = embeddings.transpose(0, 1)

        return {
            "embeddings": embeddings,
        }
        

class VisionCoreAttention(nn.Module):
    def __init__(self, config: Idefics3VisionConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads

        self.is_using_mup = config.is_using_mup
        self.checkpoint_attention = False

        self.dropout = config.attention_dropout

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size, q_length, n_local_q_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size, kv_length, n_local_kv_heads, inner_dim]
        value_states: torch.Tensor,  # [batch_size, kv_length, n_local_kv_heads, inner_dim]
    ):
        from flash_attn.flash_attn_interface import flash_attn_func

        # NOTE: this scale is for µTransfer,
        # in SP, we use sqrt(1/d_h)
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        causal = False
        dropout_rate = self.dropout if self.training else 0.0
        
        attn_output = flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            dropout_p=dropout_rate,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output 
    
class VisionSelfAttention(nn.Module, AttachableStore):
    def __init__(self, config: Idefics3VisionConfig, parallel_config: Optional[ParallelismArgs],
    tp_pg: dist.ProcessGroup):
        super().__init__()

        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup



        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk
        )

        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
        )

        self.attention = VisionCoreAttention(
            config,
            parallel_config=parallel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask
    ):
        from flash_attn import bert_padding
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )

        qkv_states = self.qkv_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
        q_length, batch_size, _ = qkv_states.shape

        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
            )
            key_states = (
                key_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
            value_states = (
                value_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]


        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # [2, batch_size, seq_length, num_heads, d_qk]
        key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # [batch_size, seq_length, 2, num_heads, d_qk]
        key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()

        # [batch_size, seq_length, num_heads, d_qk]
        key_states, value_states = torch.split(key_value_states, 1, dim=2)

        kv_length = key_states.shape[1]
        key_states = key_states.view(batch_size, kv_length, self.n_local_kv_heads, self.d_qk)
        value_states = value_states.view(batch_size, kv_length, self.n_local_kv_heads, self.d_v)

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )   

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}
    

class VisionMLP(nn.Module):
    def __init__(
            self,
            config: Idefics3VisionConfig,
            parallel_config: Optional[ParallelismArgs],
            tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        first_contiguous_chunks = (
            config.intermediate_size,  # shape of up_linear
        )
        self.fc1 = TensorParallelColumnLinear(
            config.hidden_size,
            config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=first_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.fc2 = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.act = torch.compile(lambda x: nn.functional.gelu(x, approximate="tanh"))

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.fc1(hidden_states)
        hidden_states = self.fc2(self.act(merged_states))
        return {"hidden_states": hidden_states}
    


class VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Idefics3VisionConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup
    ):
        super().__init__()

        self.self_attn = VisionSelfAttention(
            config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

        self.layer_norm1 = TritonLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        self.layer_norm2 = TritonLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )


        self.mlp = VisionMLP(
            config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE


    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        output = self.self_attn(hidden_states=hidden_states, sequence_mask=sequence_mask)

        hidden_states = output["hidden_states"]

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]

        hidden_states = hidden_states + residual

        return {
            "hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
        }
    

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        config: Idefics3Config,
        p2p: P2P,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.config = config
        self.p2p = p2p
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        self.embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=VisionEmbedding,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
            },
            module_input_keys={
                "pixel_values",
                "patch_attention_mask"
            },
            module_output_keys={
                "embeddings"
            }
        )

        self.encoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=VisionEncoderLayer,
                    module_kwargs={
                        "config": config.vision_config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                    },
                    module_input_keys={
                        "hidden_states",
                        "sequence_mask",
                    },
                    module_output_keys={
                        "hidden_states",
                        "sequence_mask"
                    }
                )

                for _ in range(config.vision_config.num_hidden_layers)
            ]
        )

        self.post_layernorm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonLayerNorm,
            module_kwargs={"normalized_shape": config.vision_config.hidden_size, "eps": config.vision_config.layer_norm_eps},
            module_input_keys={
                "input"
            },
            module_output_keys={
                "hidden_states"
            }
        )

    def forward(
        self,
        pixel_values: Union[torch.Tensor, TensorPointer],
        patch_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        batch_size = pixel_values.size(0)

        if patch_attention_mask is None:
            patch_size = self.config.vision_config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )

            patch_attention_mask = patch_attention_mask.to(pixel_values.device, dtype=torch.bool)

        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )["embeddings"]

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)

        hidden_encoder_states = {
            "hidden_states": hidden_states,
            "sequence_mask": patch_attention_mask,
        }

        for encoder_layer in self.encoder:
            hidden_encoder_states = encoder_layer(**hidden_encoder_states)

        hidden_states = hidden_encoder_states["hidden_states"]
        hidden_states = self.post_layernorm(input=hidden_states)
        
        return hidden_states
    
    def get_block_compute_costs(self):
        config = self.config.vision_config
        d_ff = config.intermediate_size
        d_qkv = config.hidden_size // config.num_attention_heads

        return {
            VisionEncoderLayer: 4 * config.num_attention_heads * d_qkv * config.hidden_size
            + 3 * d_ff * config.hidden_size
        }


class Idefics3MLP(nn.Module):
    def __init__(
        self,
        config: Idefics3VisionConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = torch.compile(GLUActivation(config.hidden_act))

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}
    
class Idefics3SimpleMLP(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        hidden_size = config.vision_config.hidden_size

        self.input_size = hidden_size * (config.scale_factor ** 2)
        self.output_size = hidden_size

        first_contiguous_chunks = (
            self.input_size,  # shape of up_linear
        )
        self.proj = TensorParallelColumnLinear(
            self.input_size,
            hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=first_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return {"hidden_states": hidden_states}
    

class Idefics3Connector(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projector = Idefics3SimpleMLP(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x
    
    def forward(self, hidden_states):
        hidden_states = self.pixel_shuffle(hidden_states, self.scale_factor)
        hidden_states = self.modality_projector(hidden_states)["hidden_states"]
        return {"hidden_states": hidden_states}

class Idefics3Model(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()

        self.config = config
        self.image_token_id = config.image_token_id

        self.llama = LlamaModel(
            config=config.llama_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

        self.vision_model = VisionTransformer(
            config=config,
            p2p=self.llama.p2p,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

        self.connector = PipelineBlock(
            p2p=self.llama.p2p,
            module_builder=Idefics3Connector,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
            },
            module_input_keys={
                "hidden_states"
            },
            module_output_keys={
                "hidden_states"
            }
        )

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        
    def inputs_merger(
            self,
            input_ids: Union[torch.Tensor, TensorPointer],
            inputs_embeds: Union[torch.Tensor, TensorPointer],
            image_hidden_states: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer] = None,
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        batch_size, num_images, num_channels, height, width = pixel_values.size()

        pixel_values = pixel_values.view(batch_size * num_images, num_channels, height, width)

        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
                )
        else:
            # Remove padding images from the mask/pP p
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) == patch_size * patch_size).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )["hidden_states"]

        # Modality projection & resampling
        image_hidden_states = self.connector(
            hidden_states=image_hidden_states
        )

        inputs_embeds = self.llama.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        inputs_embeds = self.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds["input_embeds"],
            image_hidden_states=image_hidden_states["hidden_states"],
        )

        outputs = self.llama.forward_with_hidden_states(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            input_mask=input_mask,
        )

        return outputs
    
    def get_block_compute_costs(self):
        llama_cost = self.llama.get_block_compute_costs()
        vision_cost = self.vision_model.get_block_compute_costs()

        return {**llama_cost, **vision_cost}
    

class Idefics3ForTraining(NanotronModel):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()

        self.model = Idefics3Model(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

        self.loss = PipelineBlock(
            p2p=self.model.llama.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": parallel_context.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer],
        pixel_attention_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        outputs = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

        loss = self.loss(
            sharded_logits=outputs["sharded_logits"],
            label_ids=label_ids,
            label_mask=label_mask,
        )

        return {"loss": loss}
    

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.llama_config.tie_word_embeddings is True:
            return ["model.llama.token_position_embeddings.pp_block.token_embedding.weight", "model.llama.lm_head.pp_block.weight"]
        else:
            return []