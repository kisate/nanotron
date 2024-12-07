from abc import ABC
from dataclasses import dataclass
import io
from typing import Any, Dict, List, TypedDict, Union

import torch
from nanotron.modular_dataloader.base import SampleEncoder, BatchEncoder
import numpy as np
from nanotron import distributed as dist
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from transformers import AutoProcessor
from PIL import Image

@dataclass
class FormattedTextSample:
    """
    A sample with formatted text for processing.
    :param text: Text should include a single <image> instance.
    """
    text: str
    image: bytes

@dataclass
class CaptionSampleEncoder(SampleEncoder[FormattedTextSample]):
    """
    Sample encoder for caption samples.
    """

    text_field: str = "caption"
    image_field: str = "jpg"
    image_token: str = "<image>"

    def encode(self, sample: Dict[str, Any]) -> FormattedTextSample:
        """
        Encode a caption sample.
        """
        return FormattedTextSample(text= f"{self.image_token}{sample[self.text_field]}", image=sample[self.image_field])

def byte_img_to_array(bimg):
    imageStream = io.BytesIO(bimg)
    imageFile = Image.open(imageStream)
    img_arr = np.array(imageFile)
    # imageFile.show()
    return img_arr

@dataclass
class SingleImageBatchEncoder(BatchEncoder[FormattedTextSample]):
    """
    Expects an Idefics3 compatible processor. Pads texts and splits images.
    Only works for a single image per caption. 
    - input_pp_rank: Discards last input id token. Returns input_ids, input_mask, pixel_values
    - output_pp_rank: Discards first label id token. Returns label_ids, label_mask
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    processor: AutoProcessor
    sequence_length: int

    def encode(self, batch: List[FormattedTextSample]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """
        Encode a batch of caption samples.
        """
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert len(batch) == 0
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "pixel_values": TensorPointer(group_rank=self.input_pp_rank),
            }

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["pixel_values"] = TensorPointer(group_rank=self.input_pp_rank)

        texts = [sample.text for sample in batch]
        images = [[byte_img_to_array(sample.image)] for sample in batch]

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding="longest", max_length=self.sequence_length + 1, truncation=True) 


        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = inputs["input_ids"][:, :-1]
            result["input_mask"] = inputs["attention_mask"]
            result["pixel_values"] = inputs["pixel_values"]

        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = inputs["input_ids"][:, 1:]
            result["label_mask"] = inputs["input_ids"][:, 1:] < self.processor.tokenizer.vocab_size

        return result


        