# Doc on VLM training

## Installation

Required packages are mostly the same as for LLMs. Need to install PIL package to work with images: `pip install pillow`.

Some HF-related scripts may also require `pip install accelerate`.

`hf_transfer` may be installed to speed-up downloads from HF.

## Overview

VLM functionality uses [Idefics3](https://arxiv.org/pdf/2408.12637) architecture. It combines a CLIP-style model with a Llama-style model using their pixel shuffling technique. 

[`models/idefics.py`](/src/nanotron/models/idefics.py) contains code of the VLM implementation.

[`tools/idefics3.py`](/tools/idefics3/) contains HF/Nanotron conversion and simple evaluation scripts.

[`examples/vqa`](/examples/vqa/) contains a simple fine-tuning example using a small dataset that fits into RAM.

[`example/caption-pretrain`](/examples/caption-pretrain/) contains code that runs pretraining on a preprocessed/raw captioning dataset (LAION).

Training dataloader uses `datasets.IterableDataset` to load and preprocess the dataset step-by-step. It allows having different encoders for different datasets and is inspired by Megatron-Energon dataloader. Each dataset requires a sample encoder, that processes single samples, and a batch encoder that collates and encodes batches. 

Dataloader code is present in [`modular_dataloader`](/src/nanotron/modular_dataloader/), and your custom encoders can be added there. Currently dataloader only supports datasets stored in parquet.
