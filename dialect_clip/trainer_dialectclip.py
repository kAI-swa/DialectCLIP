import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer
from .configuration_dialectclip import DialectCLIPTrainerConfig
from .modeling_dialectclip import DialectCLIPForConditionalGeneration
from .processing_dialectclip import DialectCLIPProcessor
from .utils_dialectclip import line_plot


class DialectCLIPTrainer(nn.Module):
    def __init__(self, model: DialectCLIPForConditionalGeneration, config: Optional[DialectCLIPTrainerConfig] = None):
        super().__init__()
        if config is None:
            config = DialectCLIPTrainerConfig()
        self.config = config
        self.model = model
        self.device = self.config.device
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
        self.model.train()

    def _pretrain_loop(
            self,
            dataset: Dataset
    ):
        '''
        ------------
        Inputs:
            model: Initialized DialectCLIP class
            dataset: mindspore.dataset.Dataset
        Used for pre-train DialectCLIP on common language speech corpus
        '''
        def _collate_fn(batch):
            speech, transcript = list(zip(*batch))

            # speech preprocess
            inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
            input_speech_features = inputs.input_features

            # transcipt preprocess
            prompts = [self.model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = self.tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            return input_ids.to(device=self.device), attention_mask.to(device=self.device), input_speech_features.to(device=self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay_rate
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=_collate_fn
        )
        loss_list = []

        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as t:
            for batch_idx, (input_ids, attention_mask, input_features) in t:
                t.set_description("DialectCLIP Training")
                loss_clip, loss_casuallm, _ = self.model.forward_fn(
                    input_ids=input_ids,
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = self.model.config.alpha * loss_clip + self.model.config.beta * loss_casuallm
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix({"loss": loss.item()})
                loss_list.append(loss.item())
            
                if batch_idx % self.config.save_checkpoint_frequency == 0:
                    state = self.model.state_dict()
                    torch.save(state, self.config.model_save_path)
        line_plot(loss_list, title="Pre-train Loss")

    def _train_loop(
            self,
            dataset: Dataset
    ):
        '''
        ------------
        Inputs:
            model: Initialized DialectCLIP class
            dataset: mindspore.dataset.Dataset
        '''
        def _collate_fn(batch):
            speech, dialect, transcript = list(zip(*batch))

            # speech preprocess
            inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
            input_speech_features = inputs.input_features

            # dialect preprocess
            inputs = self.feature_extractor(dialect, sampling_rate=16000, return_tensors="pt")
            input_dialect_features = inputs.input_features

            # transcipt preprocess
            prompts = [self.model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = self.tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            return input_ids.to(device=self.device), attention_mask.to(device=self.device), \
            input_speech_features.to(device=self.device), input_dialect_features.to(device=self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.1,
            patience=5,
            verbose=True
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=_collate_fn
        )
        loss_list = []

        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as t:
            for batch_idx, (input_ids, attention_mask, input_speech_features, input_dialect_features) in t:
                t.set_description("DialectCLIP Training")
                loss = self.model.train_forward(
                    input_ids=input_ids,
                    input_speech_features=input_speech_features,
                    input_dialect_features=input_dialect_features,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update lr_scheduler
                scheduler.step(loss)

                t.set_postfix({"loss": loss.item()})
                loss_list.append(loss.item())
            
                if batch_idx % self.config.save_checkpoint_frequency == 0:
                    state = self.model.state_dict()
                    torch.save(state, self.config.model_save_path)
        line_plot(loss_list, title="Loss")

    def forward(
            self,
            dataset: Dataset,
            pretrain: Optional[bool] = None
    ):
        if pretrain:
            self._pretrain_loop(dataset)
        else:
            self._train_loop(dataset)
