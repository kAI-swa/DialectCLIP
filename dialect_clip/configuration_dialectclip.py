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
            TEXT_MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat",
            speech_token = "<AUDIO>",
            default_prompt = "<AUDIO>识别汉语; <Assistant>:",
            sampling_rate = 16000,
            padding = True,
            truncation = True,
            temperature = 0.07,
            logit_scale_init_value = 2.6592,
            initializer_range = 0.02,
            speech_dim = 1024,
            text_dim = 1024,
            num_group_tokens = 64,
            attn_dropout_rate = 0.2,
            mlp_dropout_rate = 0.2,
            post_norm = True,
            vocab_size = 32000, 
            speech_token_index = 32000,
            pad_token_id = None,
            ignore_index = -100,
            tau = 1.0,
            alpha = 1.0,
            beta = 1.0,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = False,
            is_encoder_decoder = False            
    ):
        self.SPEECH_MODEL_NAME = SPEECH_MODEL_NAME
        self.TEXT_MODEL_NAME = TEXT_MODEL_NAME
        self.speech_token = speech_token
        self.default_prompt = default_prompt
        self.sampling_rate = sampling_rate
        self.truncation = truncation
        self.temperature = temperature
        self.logit_scale_init_value = logit_scale_init_value
        self.padding = padding
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.num_group_tokens = num_group_tokens
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.post_norm = post_norm
        self.vocab_size = vocab_size
        self.speech_token_index = speech_token_index
        self.ignore_index = ignore_index
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.is_encoder_decoder = is_encoder_decoder


class DialectCLIPTrainerConfig:
    def __init__(
            self,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            epochs = 3,
            batch_size = 32,
            shuffle = True,
            num_workers = 8,
            learning_rate = 1e-5,
            weight_decay_rate = 0.001,
            save_checkpoint_frequency = 20,
            model_save_path = "./checkpoint/dialect_clip.pth",
            do_sample=False,
            temperature=None,
            num_beams=1,
            max_length=128,
            prompt_template = "<speech>识别汉语; <Assistant>:"
    ) -> None:
        # train configuration
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.save_checkpoint_frequency = save_checkpoint_frequency
        self.model_save_path = model_save_path

        # test generation configuration
        self.do_sample = do_sample
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_length = max_length
        self.prompt_template = prompt_template