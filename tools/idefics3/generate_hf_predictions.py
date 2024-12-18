"""
torchrun --nproc-per-node 1 tools/idefics3/generate_hf_predictions.py --pretrained-model-name-or-path HuggingFaceM4/Idefics3-8B-Llama3
"""

import argparse
import os
from typing import List, Optional
from PIL import Image

import numpy as np
import requests
import torch


from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers.modeling_flash_attention_utils import _flash_attention_forward

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16

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


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def main(args):
    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()

    image_1 = Image.open(requests.get(url_1, stream=True).raw)
    image_2 = Image.open(requests.get(url_2, stream=True).raw)
    images = [image_1, image_2]

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # output = model(**inputs)

        output = model.model(use_cache=False, **inputs)

    logits = model.lm_head(output.last_hidden_state)

    predicted_tokens = [5, 27, 34]  # Index of the predictions to compare across models
    term_cols = int(os.get_terminal_size().columns / 3)

    for predicted_token in predicted_tokens:
        print("\n", "=" * term_cols, f"Predictions of token {predicted_token}", "=" * term_cols)
        next_tokens = torch.softmax(logits[0, predicted_token, :], -1)
        topk_next_tokens = torch.topk(next_tokens, 10)

        print(
            *[
                f"[HF Model] Next token: {idx.item()}, probability: {prob}"
                for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)
            ],
            sep="\n",
        )

    # Compute accuracy
    # predictions = np.argmax(output.logits.cpu(), axis=2).flatten().tolist()
    # labels = tokens.cpu().flatten()[1:].tolist()
    # print(f"\nAccuracy: {accuracy_score(labels, predictions)}")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
