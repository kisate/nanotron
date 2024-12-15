from .caption import CaptionSampleEncoder, ProcessSampleEncoder, SingleImageBatchEncoder, PreprocessedCollator, PreprocessedSampleEncoder

SAMPLE_ENCODERS = {
    "caption_simple": CaptionSampleEncoder,
    "caption_process": ProcessSampleEncoder,
    "caption_preprocessed": PreprocessedSampleEncoder
}

BATCH_ENCODERS = {
    "single_image": SingleImageBatchEncoder,
    "caption_preprocessed": PreprocessedCollator
}