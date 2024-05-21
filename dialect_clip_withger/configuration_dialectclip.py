import torch
from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DialectCLIPConfig",
    "DialectCLIPTrainerConfig"
]

class DialectCLIPConfig(PretrainedConfig):
    def __init__(
            self,
            SPEECH_MODEL_NAME = "openai/whisper-medium",
            TEXT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct",
            sampling_rate = 16000,
            padding = True,
            truncation = True,
            temperature = 0.07,
            logit_scale_init_value = 2.6592,
            initializer_range = 0.02,
            lora_rank = 32,
            lora_dropout = 0.01,
            post_norm = True,
            pad_token_id = None,
            lora_alpha = 8,
            alpha = 1.0,
            beta = 1.0,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = False,
            target_modules = ["q_proj", "k_proj"],
            is_encoder_decoder = False,
            do_sample = False,
            num_beams=3,
            max_length=128,    
    ):
        self.SPEECH_MODEL_NAME = SPEECH_MODEL_NAME
        self.TEXT_MODEL_NAME = TEXT_MODEL_NAME
        self.sampling_rate = sampling_rate
        self.truncation = truncation
        self.temperature = temperature
        self.logit_scale_init_value = logit_scale_init_value
        self.padding = padding
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout
        self.post_norm = post_norm
        self.lora_alpha = lora_alpha
        self.alpha = alpha
        self.beta = beta
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.target_modules = target_modules
        self.is_encoder_decoder = is_encoder_decoder

        # generation_config
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.max_length = max_length


class DialectCLIPTrainerConfig:
    def __init__(
            self,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            load = False,
            epochs = 3,
            batch_size = 2,
            accum_iter = 16,
            shuffle = True,
            num_workers = 8,
            learning_rate = 1e-5,
            weight_decay_rate = 0.001,
            save_checkpoint_frequency = 20,
            model_save_path = "./checkpoint/dialect_clip.pth",
    ) -> None:
        # train configuration
        self.device = device
        self.load = load
        self.epochs = epochs
        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.save_checkpoint_frequency = save_checkpoint_frequency
        self.model_save_path = model_save_path