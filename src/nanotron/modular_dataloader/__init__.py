from .caption import CaptionSampleEncoder, SingleImageBatchEncoder

SAMPLE_ENCODERS = {
    "caption": CaptionSampleEncoder,
}

BATCH_ENCODERS = {
    "single_image": SingleImageBatchEncoder
}