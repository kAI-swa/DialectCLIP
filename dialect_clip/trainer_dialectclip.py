import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers.generation import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoFeatureExtractor, AutoTokenizer
from .configuration_dialectclip import DialectCLIPTrainerConfig
from .modeling_dialectclip import DialectCLIPForConditionalGeneration
from .processing_dialectclip import DialectCLIPProcessor, collate_fn
from .utils_dialectclip import WER, line_plot


class DialectCLIPTrainer(nn.Module):
    def __init__(self, config: Optional[DialectCLIPTrainerConfig] = None):
        super().__init__()
        if config is None:
            config = DialectCLIPTrainerConfig()
        self.config = config
        self.device = self.config.device

    def _train_loop(
            self,
            model: DialectCLIPForConditionalGeneration,
            dataset: Dataset,
            feature_extractor: AutoFeatureExtractor,
            tokenizer: AutoTokenizer, 
    ):
        '''
        ------------
        Inputs:
            model: Initialized DialectCLIP class
            dataset: mindspore.dataset.Dataset
        '''
        def collate_fn(batch):
            speech, dialect, transcript = list(zip(*batch))

            # speech preprocess
            inputs = feature_extractor(speech, sampling_rate=16000)
            input_features = inputs.input_features
            input_speech_features = np.concatenate(input_features, axis=0)

            # audio preprocess
            inputs = feature_extractor(dialect, sampling_rate=16000)
            input_features = inputs.input_features
            input_dialect_features = np.concatenate(input_features, axis=0)

            # transcipt preprocess
            prompts = [model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            return input_ids.to(device=self.device), attention_mask.to(device=self.device), \
                torch.tensor(input_speech_features, device=self.device), torch.tensor(input_dialect_features, device=self.device)

        optimizer = nn.AdamWeightDecay(
            model.parameters(),
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.999,
            weight_decay=self.config.weight_decay_rate
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )
        model = model.to(device=self.config.device)
        model.train()
        loss_list = []

        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as t:
            for batch_idx, (input_ids, attention_mask, input_speech_features, input_dialect_features) in t:
                t.set_description("DialectCLIP Training")
                loss = model.train_forward(
                    input_ids=input_ids,
                    input_speech_features=input_speech_features,
                    input_dialect_features=input_dialect_features,
                    attention_mask=attention_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_description({"loss": loss.item()})
                loss_list.append(loss.item())
            
                if batch_idx % self.config.save_checkpoint_frequency == 0:
                    state = model.state_dict()
                    torch.save(state, self.config.model_save_path)
        line_plot(loss_list, title="Loss")
        return loss_list

    def _test_loop(
            self,
            model: DialectCLIPForConditionalGeneration,
            dataset: Dataset,
            processor: DialectCLIPProcessor
    ):
        generation_config = GenerationConfig(
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            num_beams=self.config.num_beams,
            max_length=self.config.max_length,
        )        
        wer = WER()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )
        prompts_template = [self.config.prompt_template for _ in range(self.config.batch_size)]
        prompt_len = len(self.config.prompt_template)
        prompts = processor.tokenizer(
            prompts_template, truncation=False, padding=False, return_tensors="pt"
        )
        error_rate_list = []
        with tqdm(dataloader, total=len(dataloader), leave=True) as t:
            t.set_description("DialectCLIP Evaluation")
            for _, input_dialect_features, input_ids, _ in t:
                inputs = BatchFeature(
                    {
                        **prompts,
                        "input_features": input_dialect_features
                    }
                )
                generate_ids = model.generate(**inputs, generation_config=generation_config)
                output_texts = processor.batch_decode(generate_ids, skip_special_tokens=True)
                targets = processor.batch_decode(input_ids, skip_special_tokens=True)
                error_rate = wer.compute(
                    reference=[target[prompt_len:] for target in targets],
                    hypothesis=[output_text[prompt_len:] for output_text in output_texts]
                )
                error_rate_list.append(error_rate)
                t.set_postfix({"word error rate": error_rate})
        line_plot(error_rate_list, title="WER")
        return error_rate_list

    def forward(
            self,
            model: DialectCLIPForConditionalGeneration,
            train_dataset: Dataset,
            test_dataset: Optional[Dataset] = None,
    ):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model.config.SPEECH_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(model.config.TEXT_MODEL_NAME)
        processor = DialectCLIPProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        tokenizer.add_tokens(model.config.audio_token)
        new_vocab_size = len(tokenizer)
        model.config.vocab_size = new_vocab_size
        model.config.audio_token_index = model.config.vocab_size - 1
        loss_lists = []
        error_rate_lists = []
        for epoch in range(self.config.epochs):
            print(f"[Epoch {epoch}/{self.config.epochs}]======>")
            loss_list = self._train_loop(model, train_dataset)
            loss_lists.append(loss_list)
            if test_dataset is not None:
                error_rate_list = self._test_loop(model, test_dataset, processor)
                error_rate_lists.append(error_rate_list)
        
        loss = [loss_item for loss_list in loss_lists for loss_item in loss_list]
        wer = [error_item for error_rate_list in error_rate_lists for error_item in error_rate_list]
        line_plot(loss, title="Loss")
        line_plot(wer, title="WER")
        