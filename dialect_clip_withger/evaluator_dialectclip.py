import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from transformers.generation import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from .modeling_dialectclip import DialectCLIP
from .utils_dialectclip import WER, line_plot


class DialectCLIPEvaluator(nn.Module):
    def __init__(self, model: DialectCLIP, device):
        super().__init__()
        self.model = model
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model.config.SPEECH_MODEL_NAME)

        self.init_module()
    
    def init_module(self):
        checkpoint = torch.load("./checkpoint/dialect_clip.pth")
        self.model.load_state_dict(checkpoint)
        self.model= self.model.to(device=self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(
            self,
            dataset: Dataset,
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
            inputs = self.processor.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
            input_speech_features = inputs.input_features

            # speech preprocess
            inputs = self.processor.feature_extractor(dialect, sampling_rate=16000, return_tensors="pt")
            input_dialect_features = inputs.input_features

            # transcipt preprocess
            prompts = [self.model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = self.processor.tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            input_ids = inputs.input_ids

            return input_ids.to(device=self.device), input_speech_features.to(device=self.device), input_dialect_features.to(device=self.device)

        generation_config = GenerationConfig(
            do_sample=self.model.config.do_sample,
            num_beams=self.model.config.num_beams,
            max_length=self.model.config.max_length
        )        
        wer = WER()
        self.model.eval()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            drop_last=True
        )
        error_rate_list = []
        with tqdm(dataloader, total=len(dataloader), leave=True) as t:
            t.set_description("DialectCLIP Evaluation")
            for input_ids, _, _, input_dialect_features in t:
                inputs = BatchFeature(
                    {
                        "input_features": input_dialect_features
                    }
                )
                generate_ids = self.model.generate(**inputs, generation_config=generation_config)
                output_texts = self.processor.batch_decode(generate_ids.detach().cpu(), skip_special_tokens=True)
                targets = self.processor.batch_decode(input_ids.detach().cpu(), skip_special_tokens=True)
                error_rate = wer.compute(
                    reference=[target for target in targets],
                    hypothesis=[output_text for output_text in output_texts]
                )
                error_rate_list.append(error_rate)
                t.set_postfix({"word error rate": error_rate})
        line_plot(error_rate_list, title="WER")
        print(f"Average WER: {sum(error_rate_list)/len(error_rate_list)*100}%")
