# Idefics3 Weight conversion tool
## Conversion
This directory contains the scripts to convert the Idefics3 checkpoints from HuggingFace to Nanotron and vice versa. Nanotron to HF conversion requires `accelerate`: `pip install accelerate` (otherwise empty HF model initialization is very slow).

- Convert from HuggingFace to Nanotron

`HF_HUB_ENABLE_HF_TRANSFER=1 torchrun --nproc-per-node 1 tools/idefics3/convert_hf_to_nanotron.py --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3 --pretrained-model-name-or-path HuggingFaceM4/Idefics3-8B-Llama3`
- Convert from Nanotron to HuggingFace

`torchrun --nproc-per-node 1 tools/idefics3/convert_nanotron_to_hf.py --huggingface-checkpoint-path idefics3_ckpt --pretrained-model-name-or-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3`

- Combine custom HF Llama and CLIP models in a Idefics3-like approach and save it as a Nanotron model:

`torchrun --nproc-per-node 1 tools/idefics3/build_nanotron_from_hf.py --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3 --pretrained-model-name-or-path-llama3 meta-llama/Meta-Llama-3-8B-Instruct --pretrained-model-name-or-path-siglip google/siglip-so400m-patch14-384`

In summary, we will do the following:
- Initialize the HuggingFace model with the pretrained weights. The model definition is [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py).
- Initialize a Nanotron model with empty weights. The model definition is [here](https://github.com/huggingface/nanotron/blob/main/src/nanotron/models/idefics.py).
- Copy the parameters layer by layer from one model to the other.
- Store the Nanotron model.

When comparing the HuggingFace implementation with the Nanotron implementation, the main difference lies in the Q, K & V matrices and in the MLP projections. In the HuggingFace implementation, these matrices are separated. It is crucial to pay attention to these details to convert the models correctly.

To perform the conversion, we will need at least **1 GPU**, although the operations will be carried out on the **CPU**. We will convert the models with a parallel configuration of DP = PP = TP = 1, but it should be noted that the checkpoints generated by Nanotron are topology agnostic.

## Simple evaluation
A simple sanity check for conversion being correct can be made using `loss_on_captions` scripts that run the model on ~100 samples from an image captioning dataset. 

- Check HF model performance:

`torchrun --nproc-per-node 1 tools/idefics3/loss_on_captions_hf.py --pretrained-model-name-or-path HuggingFaceM4/Idefics3-8B-Llama3`

- Check Nanotron model performance:

`torchrun --nproc-per-node 2 tools/idefics3/loss_on_captions_nanotron.py --tp 2 --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3`

Could also load a saved dataset from the drive:

`torchrun --nproc-per-node 2 tools/idefics3/loss_on_captions_nanotron.py --tp 2 --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3 --dataset-path "../datasets/ny_captions.hf"`