from typing import Any, Dict, List, Optional
from datasets import IterableDataset
from nanotron.parallel.context import ParallelContext
from nanotron import distributed as dist
from nanotron.modular_dataloader.base import SampleEncoder, BatchEncoder, T_encoded_sample
from multiprocessing.pool import ThreadPool
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets.distributed import split_dataset_by_node

from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer  

def from_columns(batch: Dict[str, List]):
    return [{k: batch[k][i] for k in batch} for i in range(len(batch[list(batch.keys())[0]]))]

class EmptyIterableDataset(IterableDataset):
    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

import itertools

class EmptyDataset(IterableDataset):
    def __init__(self, input_pp_rank: int, output_pp_rank: int, num_shards: int):
        super().__init__()
        self.input_pp_rank = input_pp_rank
        self.output_pp_rank = output_pp_rank
        self._num_shards = num_shards

    @property
    def num_shards(self):
        return self._num_shards

    def __iter__(self):
        return itertools.cycle([
            {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "pixel_values": TensorPointer(group_rank=self.input_pp_rank),
            }
        ])

def get_train_dataloader(
    train_dataset: IterableDataset,
    sample_encoder: SampleEncoder,
    batch_encoder: BatchEncoder,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    sample_encoding_batch: int,
    batch_encoding_batch: int,
    seed_worker: int,
    sample_encoding_workers: int,
    batch_encoding_workers: int,
    drop_last: bool = True,
    dataloader_state: Optional[Dict] = None,
): 
    if not isinstance(train_dataset, IterableDataset):
        raise ValueError("Dataset should be a datasets.IterableDataset")
    
    if dist.get_rank(parallel_context.pp_pg) not in [input_pp_rank, output_pp_rank]:

        def generator():
            while True:
                yield {
                    "input_ids": TensorPointer(group_rank=input_pp_rank),
                    "input_mask": TensorPointer(group_rank=input_pp_rank),
                    "label_ids": TensorPointer(group_rank=output_pp_rank),
                    "label_mask": TensorPointer(group_rank=output_pp_rank),
                    "pixel_values": TensorPointer(group_rank=input_pp_rank),
                }

        empty_dataset = IterableDataset.from_generator(generator)

        return StatefulDataLoader(
            empty_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda x: x[0],
        )
    
    train_dataset = split_dataset_by_node(train_dataset, rank=parallel_context.dp_pg.rank(), world_size=parallel_context.dp_pg.size())
    train_dataset = train_dataset.shuffle(seed=seed_worker)

    def encode_samples_batched(batch: Dict[str, List]):
        batch = from_columns(batch)

        with ThreadPool(sample_encoding_workers) as sample_worker_pool:
            encoded_batch = sample_worker_pool.map(sample_encoder.encode, batch)

        return {"sample_encoded": encoded_batch}
    
    train_dataset = train_dataset.map(encode_samples_batched, batched=True, remove_columns=train_dataset.column_names, batch_size=sample_encoding_batch)
    
    
    def collate_fn(batch: List[Dict[str, Any]]):
        batch = [x["sample_encoded"] for x in batch]
        return batch_encoder.encode(batch)

    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        num_workers=1,
        collate_fn=collate_fn,
    )

    if dataloader_state is not None:
        dataloader.load_state_dict(dataloader_state)


    return dataloader




    
    
