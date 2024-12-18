"""
torchrun --nproc-per-node 1 tools/idefics3/generate_nanotron_predictions_vit.py --tp 1 --nanotron-checkpoint-path nanotron-ckpt-vit
"""

import argparse
import os
from pathlib import Path

import requests
import torch
from PIL import Image

# from sklearn.metrics import accuracy_score
from transformers import AutoProcessor

import nanotron.distributed as dist
from nanotron.config import Config, ParallelismArgs, get_config_from_file
from nanotron.models import build_model
from nanotron.models.idefics import VisionTransformerNanotron
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s the difference between these two images?"},
            {"type": "image"},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
        ],
    },
]


url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

SEQ_LENGTH = 512  # For truncating the TXT if GPU can't fit too many tokens

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory containing a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="Nanotron Parallelism")
    group.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree of the Nanotron Checkpoint")

    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    assert (
        parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        and parallel_config.tp_linear_async_communication is False
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    RANK = dist.get_rank(parallel_context.world_pg)

    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )

    model = build_model(
        model_builder=lambda: VisionTransformerNanotron(
            config=nanotron_config.model.model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,  # TODO Check with different parallelism if cpu is available
    )

    # torch.Size([484, 26, 768])

    mark_tied_parameters(model=model, parallel_context=parallel_context)
    sanity_check(root_module=model)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path))

    image_1 = Image.open(requests.get(url_1, stream=True).raw)
    image_2 = Image.open(requests.get(url_2, stream=True).raw)
    images = [image_1, image_2]

    # Using non-Idefics3 image size may break the pixel shuffle
    # For example, instead of 384 you should use either 364 or 404
    image_size = nanotron_config.model.model_config.image_size

    image_size = 364

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/Idefics3-8B-Llama3",
        size={"longest_edge": 2 * image_size},
        max_image_size={"longest_edge": image_size},
    )

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt").to(DEVICE)

    seq_length = inputs.input_ids.size(1)

    inputs = {
        "input_ids": inputs["input_ids"],
        "input_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"].bfloat16(),
        "pixel_attention_mask": inputs["pixel_attention_mask"],
    }

    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]

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
        pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

    patch_size = nanotron_config.model.model_config.patch_size
    patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
    patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
    patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) == patch_size * patch_size).bool()

    pixel_values = pixel_values.bfloat16()

    model.eval()

    with torch.no_grad():
        output = model.model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )["hidden_states"]

    print(output.shape)


if __name__ == "__main__":
    _args = get_args()
    main(_args)
