from .caption import CaptionSampleEncoder, ProcessSampleEncoder, SingleImageBatchEncoder, PreprocessedCollator

SAMPLE_ENCODERS = {
    "caption_simple": CaptionSampleEncoder,
    "caption_process": ProcessSampleEncoder
}

BATCH_ENCODERS = {
    "single_image": SingleImageBatchEncoder,
    "caption_preprocessed": PreprocessedCollator
}