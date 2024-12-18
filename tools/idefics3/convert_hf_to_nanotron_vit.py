"""
HF_HUB_ENABLE_HF_TRANSFER=1 torchrun --nproc-per-node 1 tools/idefics3/convert_hf_to_nanotron_vit.py --nanotron-checkpoint-path nanotron-ckpt-vit --pretrained-model-name-or-path HuggingFaceM4/Idefics3-8B-Llama3
"""

import sys

sys.path.append(".venv/lib/python3.10/site-packages")

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# from tools.llama3.convert_hf_to_nanotron import copy_weights_from_hf_to_nanotron as copy_weights_from_hf_to_nanotron_llama
# from tools.llama3.convert_hf_to_nanotron import nanotron_config_from_hf_config as nanotron_config_from_hf_config_llama
from transformers import AutoModel, AutoModelForVision2Seq

from nanotron import logging
from nanotron.config.config import Config, GeneralArgs, LoggingArgs, ModelArgs, TokenizerArgs
from nanotron.config.models_config import ExistingCheckpointInit, Idefics3VisionConfig
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models.base import build_model
from nanotron.models.idefics import VisionTransformer, VisionTransformerNanotron
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize.weights import save_weights
from nanotron.trainer import mark_tied_parameters

logger = logging.get_logger(__name__)

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace Idefic3 Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def copy_weights_from_hf_to_nanotron_vision(
    nanotron_model: VisionTransformer, hf_model: AutoModel, nanotron_vision_config: Idefics3VisionConfig
):
    log_rank("Copying weights from Idefic3 ViT model to Nanotron model...", logger=logger, level=logging.INFO, rank=0)

    # Vision Embeddings
    log_rank("Copying Vision Embeddings...", logger=logger, level=logging.INFO, rank=0)

    assert (
        nanotron_model.embeddings.pp_block.patch_embedding.weight.shape
        == hf_model.embeddings.patch_embedding.weight.shape
    )

    assert (
        nanotron_model.embeddings.pp_block.patch_embedding.bias.shape == hf_model.embeddings.patch_embedding.bias.shape
    )

    assert (
        nanotron_model.embeddings.pp_block.position_embedding.weight.shape
        == hf_model.embeddings.position_embedding.weight.shape
    )

    with torch.no_grad():
        nanotron_model.embeddings.pp_block.patch_embedding.weight.copy_(hf_model.embeddings.patch_embedding.weight)

        nanotron_model.embeddings.pp_block.patch_embedding.bias.copy_(hf_model.embeddings.patch_embedding.bias)

        nanotron_model.embeddings.pp_block.position_embedding.weight.copy_(
            hf_model.embeddings.position_embedding.weight
        )

    log_rank("Copied Vision Embeddings", logger=logger, level=logging.INFO, rank=0)

    for i in tqdm(
        range(nanotron_vision_config.num_hidden_layers),
        desc="Copying Vision Layers",
        total=nanotron_vision_config.num_hidden_layers,
    ):
        assert (
            nanotron_model.encoder[i].pp_block.layer_norm1.weight.shape
            == hf_model.encoder.layers[i].layer_norm1.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.layer_norm1.weight.copy_(hf_model.encoder.layers[i].layer_norm1.weight)

        assert (
            nanotron_model.encoder[i].pp_block.layer_norm1.bias.shape
            == hf_model.encoder.layers[i].layer_norm1.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.layer_norm1.bias.copy_(hf_model.encoder.layers[i].layer_norm1.bias)

        tmp_qkv_proj = torch.cat(
            [
                hf_model.encoder.layers[i].self_attn.q_proj.weight,
                hf_model.encoder.layers[i].self_attn.k_proj.weight,
                hf_model.encoder.layers[i].self_attn.v_proj.weight,
            ],
            dim=0,
        )

        assert tmp_qkv_proj.shape == nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.weight.shape

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.weight.copy_(tmp_qkv_proj)

        tmp_qkv_proj_bias = torch.cat(
            [
                hf_model.encoder.layers[i].self_attn.q_proj.bias,
                hf_model.encoder.layers[i].self_attn.k_proj.bias,
                hf_model.encoder.layers[i].self_attn.v_proj.bias,
            ],
            dim=0,
        )

        assert tmp_qkv_proj_bias.shape == nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.bias.shape

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.bias.copy_(tmp_qkv_proj_bias)

        ## O

        assert (
            nanotron_model.encoder[i].pp_block.self_attn.o_proj.weight.shape
            == hf_model.encoder.layers[i].self_attn.out_proj.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.self_attn.o_proj.weight.copy_(
                hf_model.encoder.layers[i].self_attn.out_proj.weight
            )

        assert (
            nanotron_model.encoder[i].pp_block.self_attn.o_proj.bias.shape
            == hf_model.encoder.layers[i].self_attn.out_proj.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.self_attn.o_proj.bias.copy_(
                hf_model.encoder.layers[i].self_attn.out_proj.bias
            )

        # Layer Norm 2

        assert (
            nanotron_model.encoder[i].pp_block.layer_norm2.weight.shape
            == hf_model.encoder.layers[i].layer_norm2.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.layer_norm2.weight.copy_(hf_model.encoder.layers[i].layer_norm2.weight)

        assert (
            nanotron_model.encoder[i].pp_block.layer_norm2.bias.shape
            == hf_model.encoder.layers[i].layer_norm2.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.layer_norm2.bias.copy_(hf_model.encoder.layers[i].layer_norm2.bias)

        # MLP
        ## FC1

        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc1.weight.shape == hf_model.encoder.layers[i].mlp.fc1.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.mlp.fc1.weight.copy_(hf_model.encoder.layers[i].mlp.fc1.weight)

        assert nanotron_model.encoder[i].pp_block.mlp.fc1.bias.shape == hf_model.encoder.layers[i].mlp.fc1.bias.shape

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.mlp.fc1.bias.copy_(hf_model.encoder.layers[i].mlp.fc1.bias)

        ## FC2

        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc2.weight.shape == hf_model.encoder.layers[i].mlp.fc2.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.mlp.fc2.weight.copy_(hf_model.encoder.layers[i].mlp.fc2.weight)

        assert nanotron_model.encoder[i].pp_block.mlp.fc2.bias.shape == hf_model.encoder.layers[i].mlp.fc2.bias.shape

        with torch.no_grad():
            nanotron_model.encoder[i].pp_block.mlp.fc2.bias.copy_(hf_model.encoder.layers[i].mlp.fc2.bias)

    log_rank("Copied Vision Layers", logger=logger, level=logging.INFO, rank=0)

    # Post layer norm

    assert nanotron_model.post_layernorm.pp_block.weight.shape == hf_model.post_layernorm.weight.shape

    with torch.no_grad():
        nanotron_model.post_layernorm.pp_block.weight.copy_(hf_model.post_layernorm.weight)

    assert nanotron_model.post_layernorm.pp_block.bias.shape == hf_model.post_layernorm.bias.shape

    with torch.no_grad():
        nanotron_model.post_layernorm.pp_block.bias.copy_(hf_model.post_layernorm.bias)

    log_rank("Copied Post Layer Norm", logger=logger, level=logging.INFO, rank=0)


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Llama3-8B HF model
    log_rank(
        f"Loading pretrained Idefics3 model: {args.pretrained_model_name_or_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    hf_model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2"
    ).to(DEVICE)
    hf_config = hf_model.config
    hf_config_vision = hf_config.vision_config

    # Set Nanotron SigLIPConfig
    nanotron_vision_config = Idefics3VisionConfig(
        hidden_size=hf_config_vision.hidden_size,
        image_size=hf_config_vision.image_size,
        intermediate_size=hf_config_vision.intermediate_size,
        num_hidden_layers=hf_config_vision.num_hidden_layers,
        num_attention_heads=hf_config_vision.num_attention_heads,
        num_key_value_heads=hf_config_vision.num_attention_heads,
        num_channels=hf_config_vision.num_channels,
        patch_size=hf_config_vision.patch_size,
        hidden_act=hf_config_vision.hidden_act,
        layer_norm_eps=hf_config_vision.layer_norm_eps,
        attention_dropout=hf_config_vision.attention_dropout,
        is_using_mup=False,
    )

    # Init Idefics3 Nanotron model
    log_rank("Init empty Nanotron Idefics3 Model", logger=logger, level=logging.INFO, rank=0)

    nanotron_model = build_model(
        model_builder=lambda: VisionTransformerNanotron(
            config=nanotron_vision_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    sanity_check(root_module=nanotron_model)

    copy_weights_from_hf_to_nanotron_vision(
        nanotron_model=nanotron_model.model,
        hf_model=hf_model.model.vision_model,
        nanotron_vision_config=nanotron_vision_config,
    )

    log_rank("Copied weights from HF SigLIP model to Nanotron model!", logger=logger, level=logging.INFO, rank=0)

    nanotron_checkpoint_path = Path(args.nanotron_checkpoint_path)

    save_weights(
        model=nanotron_model,
        root_folder=nanotron_checkpoint_path,
        parallel_context=parallel_context,
    )

    # Store Config and Model Config files
    with open(nanotron_checkpoint_path / "config.yaml", "w") as f:
        config = Config(
            general=GeneralArgs(project="Nanotron", run="Idefics3"),
            parallelism=parallel_config,
            model=ModelArgs(
                init_method=ExistingCheckpointInit(nanotron_checkpoint_path),
                model_config=nanotron_vision_config,
            ),
            tokenizer=TokenizerArgs(nanotron_checkpoint_path),
        )
        log_rank("Saving config ...", logger=logger, level=logging.INFO, rank=0)
        yaml.dump(config.as_dict(), f)

    with open(nanotron_checkpoint_path / "model_config.json", "w") as f:
        log_rank("Saving model config ...", logger=logger, level=logging.INFO, rank=0)
        json.dump(asdict(nanotron_vision_config), f)

    log_rank(
        f"Checkpoint conversion finished, check {args.nanotron_checkpoint_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )


if __name__ == "__main__":
    _args = get_args()
    main(_args)
