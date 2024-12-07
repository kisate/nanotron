from .caption import CaptionSampleEncoder, SingleImageBatchEncoder

SAMPLE_ENCODERS = {
    "caption_simple": CaptionSampleEncoder,
}

BATCH_ENCODERS = {
    "single_image": SingleImageBatchEncoder
}