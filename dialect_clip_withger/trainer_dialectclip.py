import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from .configuration_dialectclip import DialectCLIPTrainerConfig
from .modeling_dialectclip import DialectCLIP
from .utils_dialectclip import line_plot


class DialectCLIPTrainer(nn.Module):
    def __init__(self, model: DialectCLIP, config: Optional[DialectCLIPTrainerConfig] = None):
        super().__init__()
        if config is None:
            config = DialectCLIPTrainerConfig()
        self.config = config
        self.model = model
        self.device = self.config.device
        self.processor = AutoProcessor.from_pretrained(model.config.SPEECH_MODEL_NAME)

        self.init_module()
    
    def init_module(self):
        if self.config.load:
            checkpoint = torch.load("./checkpoint/dialect_clip.pth")
            self.model.load_state_dict(checkpoint)
        self.model= self.model.to(device=self.device)
        self.model.train()

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
            inputs = self.processor.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
            input_speech_features = inputs.input_features

            # speech preprocess
            inputs = self.processor.feature_extractor(dialect, sampling_rate=16000, return_tensors="pt")
            input_dialect_features = inputs.input_features

            # transcipt preprocess
            prompts = [self.model.config.default_prompt + transcript_item for transcript_item in transcript]
            inputs = self.processor.tokenizer(prompts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            decoder_input_ids = inputs.input_ids

            return decoder_input_ids.to(device=self.device), input_speech_features.to(device=self.device), input_dialect_features.to(device=self.device)

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
            for batch_idx, (decoder_input_ids, input_speech_features, input_dialect_features) in t:
                t.set_description("DialectCLIP Training")
                loss = self.model.train_forward(
                    input_speech_features=input_speech_features,
                    input_dialect_features=input_dialect_features,
                    decoder_input_ids=decoder_input_ids,
                    labels=decoder_input_ids
                )
                loss = loss / self.config.accum_iter
                loss.backward()

                if ((batch_idx + 1) % self.config.accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                    optimizer.zero_grad()
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
            train_dataset: Dataset,
    ):
        self._train_loop(train_dataset)
