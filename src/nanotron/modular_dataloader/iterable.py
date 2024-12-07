from typing import Any, Dict, List, Optional
from datasets import IterableDataset
from nanotron.parallel.context import ParallelContext
from nanotron import distributed as dist
from nanotron.modular_dataloader.base import SampleEncoder, BatchEncoder, T_encoded_sample
from multiprocessing.pool import ThreadPool
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets.distributed import split_dataset_by_node  

def from_columns(batch: Dict[str, List]):
    return [{k: batch[k][i] for k in batch} for i in range(len(batch[list(batch.keys())[0]]))]

class DataLoaderWithPools(StatefulDataLoader):
    def __init__(self, pools: List[ThreadPool], *args, **kwargs):
        self.pools = pools
        super().__init__(*args, **kwargs)


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
        return None
    
    train_dataset = split_dataset_by_node(train_dataset, rank=parallel_context.dp_pg.rank(), world_size=parallel_context.dp_pg.size())
    train_dataset = train_dataset.shuffle(seed=seed_worker)

    import time

    def encode_samples_batched(batch: Dict[str, List]):
        start = time.time()
        batch = from_columns(batch)

        with ThreadPool(sample_encoding_workers) as sample_worker_pool:
            encoded_batch = sample_worker_pool.map(sample_encoder.encode, batch)

        end = time.time()

        print(f"Time taken to encode {len(encoded_batch)} samples: {end - start}")

        return {"sample_encoded": encoded_batch}
    
    train_dataset = train_dataset.map(encode_samples_batched, batched=True, remove_columns=train_dataset.column_names, batch_size=sample_encoding_batch)
    train_dataset = train_dataset.batch(micro_batch_size, drop_last_batch=drop_last)

    def encode_batch_batched(batch: Dict[str, List[List[T_encoded_sample]]]):
        start = time.time()

        def process_worker_batch(worker_batch: List[T_encoded_sample]):
            encoded_batch = batch_encoder.encode(worker_batch)
            return encoded_batch

        batch = batch["sample_encoded"]

        with ThreadPool(batch_encoding_workers) as processing_pool: 
            encoded_batch = processing_pool.map(process_worker_batch, batch)

        end = time.time()
        print(f"Time taken to encode {len(encoded_batch)} batches: {end - start}")

        return {"batch_encoded": encoded_batch}
    

    train_dataset = train_dataset.map(encode_batch_batched, batched=True, remove_columns=["sample_encoded"], batch_size=batch_encoding_batch)

    dataloader = DataLoaderWithPools(
        [],
        train_dataset,
        batch_size=1,
        num_workers=0,
    )

    if dataloader_state is not None:
        dataloader.load_state_dict(dataloader_state)


    return dataloader




    
    
