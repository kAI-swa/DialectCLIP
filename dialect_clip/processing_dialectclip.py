import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from typing import Any, Optional, Union, List
from .configuration_dialectclip import DialectCLIPConfig


__all__ = [
    "DialectCLIPProcessor",
    "data_pipe"
]

class DialectCLIPProcessor(ProcessorMixin):
    r'''
    Constructs a DialectCLIP processor which wraps a feature extractor and a tokenizer into a single processor.

    Args:
        feature_extractor (`FeatureExtractor`):
            An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Tokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
    '''
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = ("WhisperFeatureExtractor")
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        speech_token = "<AUDIO>"
        self.tokenizer.add_tokens(speech_token)

    def __len__(self):
        return len(self.tokenizer)

    def __call__(
        self,
        prompt: Union[str, List[str], List[List[str]]] = None,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]] = None,
        sampling_rate: int = 16000,
        padding: Union[bool, str, PaddingStrategy] = "max_length",
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length=128,
    ) -> BatchFeature:
        '''
        Prepare inputs for DialectCLIP model
        ---------------------
        Args:
            prompt (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
            raw_audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate: int [sample rate of raw audio signal]
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.

        Returns:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `prompt` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `prompt` is not
              `None`).
            - **audio** -- mel audio features to be fed into model when raw_audio is not None                     
        '''
        if raw_audio is not None:
            inputs = self.feature_extractor(raw_audio, sampling_rate=sampling_rate, return_tensors="pt")
            audio = inputs.input_features
        else:
            audio = None
        text_inputs = self.tokenizer(
            prompt, max_length=max_length, truncation=truncation, padding=padding, return_tensors="pt"
        )

        return BatchFeature(
            data={
                **text_inputs,
                "audio": audio
            }
        )

    def batch_decode(self, output_ids: Union[List[torch.Tensor]], skip_special_tokens: Optional[bool] = None):
        if output_ids.ndim == 1:
            raise ValueError("id be decoded should be in batch to use batch_decode method")

        outputs = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens = skip_special_tokens,
        )

        return outputs

    def decode(self, output_ids: torch.Tensor, skip_special_tokens: Optional[bool] = None):
        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens = skip_special_tokens
        )
