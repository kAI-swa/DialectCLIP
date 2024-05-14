import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from dialect_clip.modeling_dialectclip import DialectCLIPForConditionalGeneration
from dialect_clip.processing_dialectclip import DialectCLIPProcessor
from dialect_clip.utils_dialectclip import WER, line_plot


class DialectCLIPEvaluator(nn.Module):
    def __init__(self, model: DialectCLIPForConditionalGeneration, device):
        super().__init__()
        self.model = model
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model.config.SPEECH_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(model.config.TEXT_MODEL_NAME)
        self.processor = DialectCLIPProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

        self.init_module()
    
    def init_module(self):
        self.tokenizer.add_tokens(self.model.config.speech_token)
        new_vocab_size = len(self.tokenizer)
        self.model.config.vocab_size = new_vocab_size
        self.model.config.speech_token_index = self.model.config.vocab_size - 1
        self.model.language_model.resize_token_embeddings(new_vocab_size)
        checkpoint = torch.load("./checkpoint/dialect_clip.pth")
        self.model.load_state_dict(checkpoint)
        
        self.model= self.model.to(device=self.device)

        self.model.eval()

    @torch.no_grad()
    def forward(
            self,
            dataset: Dataset,
            prompt_template: str = "<AUDIO>识别汉语; <Assistant>:",
            do_sample: Optional[bool] = None,
            temperature: Optional[bool] = None,
            num_beams: Optional[int] = 1,
            max_length: Optional[int] = 128,
            batch_size: Optional[int] = 16,
            num_workers: Optional[int] = 0,
        ):
        '''
        --------------
        Inputs:
            dataset: torch.utils.data.Dataset, dataset for evaluation
        '''
        def _collate_fn(batch):
            speech, dialect, transcript = list(zip(*batch))

            # speech preprocess
            inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
            input_speech_features = inputs.input_features

            # speech preprocess
            inputs = self.feature_extractor(dialect, sampling_rate=16000, return_tensors="pt")
            input_dialect_features = inputs.input_features

            # transcipt preprocess
            prompts = [self.model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = self.tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            return input_ids.to(device=self.device), attention_mask.to(device=self.device), \
            input_speech_features.to(device=self.device), input_dialect_features.to(device=self.device)
            

        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length
        )        
        wer = WER()
        self.model.eval()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn
        )
        prompts_template = [prompt_template for _ in range(batch_size)]
        prompt_len = len(prompt_template)
        prompts = self.processor.tokenizer(
            prompts_template, truncation=False, padding=False, return_tensors="pt"
        )
        prompts = prompts.to(device=self.device)
        error_rate_list = []
        with tqdm(dataloader, total=len(dataloader), leave=True) as t:
            t.set_description("DialectCLIP Evaluation")
            for input_ids, _, _, input_dialect_features in t:
                inputs = BatchFeature(
                    {
                        **prompts,
                        "input_features": input_dialect_features
                    }
                )
                generate_ids = self.model.generate(**inputs, generation_config=generation_config)
                output_texts = self.processor.batch_decode(generate_ids.detach().cpu(), skip_special_tokens=True)
                targets = self.processor.batch_decode(input_ids.detach().cpu(), skip_special_tokens=True)
                error_rate = wer.compute(
                    reference=[target[prompt_len:] for target in targets],
                    hypothesis=[output_text[prompt_len:] for output_text in output_texts]
                )
                error_rate_list.append(error_rate)
                t.set_postfix({"word error rate": error_rate})
        line_plot(error_rate_list, title="WER")
        print(f"Average WER: {sum(error_rate_list)/len(error_rate_list)*100}%")
