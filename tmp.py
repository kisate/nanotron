a = {'checkpoints': None, 'data_stages': None, 'general': {'benchmark_csv_path': None, 'consumed_train_samples': None, 'entity': None, 'ignore_sanity_checks': True, 'project': 'Nanotron', 'run': 'Idefics3', 'seed': 42, 'step': None}, 'lighteval': None, 'logging': None, 'model': {'ddp_bucket_cap_mb': 25, 'dtype': 'bfloat16', 'init_method': {'path': 'nanotron-ckpt'}, 'make_vocab_size_divisible_by': 1, 'model_config': {'image_token_id': 128257, 'llama_config':  {'bos_token_id': 128000, 'eos_token_id': [128001, 128008, 128009], 'hidden_act': 'silu', 'hidden_size': 4096, 'initializer_range': 0.02, 'intermediate_size': 14336, 'is_llama_config': True, 'max_position_embeddings': 131072, 'num_attention_heads': 32, 'num_hidden_layers': 32, 'num_key_value_heads': 8, 'pad_token_id': None, 'pretraining_tp': 1, 'rms_norm_eps': 1e-05, 'rope_interleaved': False, 'rope_scaling': {'factor': 8.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}, 'rope_theta': 500000.0, 'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 128260}, 'pad_token_id': None, 'scale_factor': 2, 'vision_config': {'attention_dropout': 0.0, 'hidden_act': 'gelu_pytorch_tanh', 'hidden_size': 1152, 'image_size': 364, 'intermediate_size': 4304, 'is_using_mup': False, 'layer_norm_eps': 1e-06, 'num_attention_heads': 16, 'num_channels':3, 'num_hidden_layers': 27, 'num_key_value_heads': 16, 'patch_size': 14}}}, 'optimizer': None, 'parallelism': {'dp': 1, 'expert_parallel_size': 1, 'pp': 1, 'pp_engine': 'afab', 'recompute_layer': False,'tp': 1, 'tp_linear_async_communication': False, 'tp_mode': 'ALL_REDUCE', 'tp_recompute_allgather': True}, 'profiler': None, 'tokenizer': {'tokenizer_max_length': None, 'tokenizer_name_or_path': 'nanotron-ckpt', 'tokenizer_revision': None}, 'tokens':None}


g = {'checkpoints': None, 'data_stages': None, 'general': {'benchmark_csv_path': None, 'consumed_train_samples': None, 'entity': None, 'ignore_sanity_checks': True, 'project': 'Nanotron', 'run': 'Idefics3', 'seed': 42, 'step': None},'lighteval': None, 'logging': None, 'model': {'ddp_bucket_cap_mb': 25, 'dtype': 'bfloat16', 'init_method': {'path': 'nanotron-ckpt'}, 'make_vocab_size_divisible_by': 1, 'model_config': {'image_token_id': 128257, 'llama_config': {'bos_token_id': 128000, 'eos_token_id': 128009, 'hidden_act': 'silu', 'hidden_size': 4096, 'initializer_range': 0.02, 'intermediate_size': 14336, 'is_llama_config': True, 'max_position_embeddings': 8192, 'num_attention_heads': 32, 'num_hidden_layers': 32, 'num_key_value_heads': 8, 'pad_token_id': None, 'pretraining_tp': 1, 'rms_norm_eps': 1e-05, 'rope_interleaved': False, 'rope_scaling': None, 'rope_theta': 500000.0, 'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 128260}, 'pad_token_id': 128002, 'scale_factor': 2, 'vision_config': {'attention_dropout': 0.0, 'hidden_act': 'gelu_pytorch_tanh', 'hidden_size': 1152, 'image_size': 384, 'intermediate_size': 4304, 'is_using_mup': False, 'layer_norm_eps': 1e-06, 'num_attention_heads': 16, 'num_channels': 3, 'num_hidden_layers': 27, 'num_key_value_heads': 16, 'patch_size': 14}}}, 'optimizer': None, 'parallelism': {'dp': 1, 'expert_parallel_size': 1, 'pp': 1, 'pp_engine': 'afab', 'recompute_layer': False, 'tp': 1, 'tp_linear_async_communication': False, 'tp_mode': 'ALL_REDUCE', 'tp_recompute_allgather': True}, 'profiler': None, 'tokenizer': {'tokenizer_max_length':None, 'tokenizer_name_or_path': 'nanotron-ckpt', 'tokenizer_revision': None}, 'tokens': None}  

# print(b)

def compare_dicts(a, b):
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        if key in a and key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                compare_dicts(a[key], b[key])
            elif a[key] != b[key]:
                print(f"Key: {key}, Value in a: {a[key]}, Value in b: {b[key]}")
        if key in a and key not in b:
            print(f"Key: {key} not in b")
        if key in b and key not in a:
            print(f"Key: {key} not in a")

compare_dicts(a, g)
