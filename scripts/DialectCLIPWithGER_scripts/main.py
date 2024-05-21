import argparse
import torch
from typing import Optional
from torch.utils.data import random_split
from dataset_construct import *
from dialect_clip_withger import DialectCLIP
from dialect_clip_withger import DialectCLIPConfig, DialectCLIPTrainerConfig
from dialect_clip_withger import DialectCLIPTrainer, DialectCLIPEvaluator


def arg_parser():
    parser = argparse.ArgumentParser(
        prog="DialectCLIP",
        description="Automatic Speech Recognition for Low Resource Languages(Dialect)",
        epilog="Thank you for using %(prog)s"
    )

    parser.add_argument(
        "--dataset", type=str, choices=["Uyghur", "temp"],
        help="Dataset for running DialectCLIP"
    )

    model_parser = parser.add_argument_group("Model configuration", description="config DialectCLIP hyperparameters")
    model_parser.add_argument(
        "--speech_model", type=str,
        help="Choose Audio Backbone model to encode Speech features, default is openai/whisper-medium"
    )
    model_parser.add_argument(
        "--language_model", type=str,
        help="Choose Language Backbone model to encode Texture features, default is Qwen/Qwen1.5-0.5B-Chat"
    )
    model_parser.add_argument(
        "--sampling_rate", "-sr", type=int, default=16000,
        help="The frequency or times to sample the audio signal per second, default is 16000Hz"
    )
    model_parser.add_argument(
        "--logit_scale_init_value", type=float, default=2.6592,
        help="Scale factor when computing the CLIP similarity matrix"
    )
    model_parser.add_argument(
        "--initializer_range", type=float, default=0.02, 
        help="Variation range when initializing network Linear layers"
    )
    model_parser.add_argument(
        "--lora_rank", default=32, type=int, help="lora rank"
    )
    model_parser.add_argument(
        "--lora_dropout", default=0.01, type=float, help="dropout rate in Lora adapter"
    )
    model_parser.add_argument(
        "--no_post_norm", action="store_true",
        help="Whether to perform layer normalization at the end of the grouping layer, if --no_post_norm is given, then no post normalization will be performed"
    )
    model_parser.add_argument(
        "--pad_token_id", type=Optional[int], default=None,
        help="The special pad token index in the tokenizer"
    )
    model_parser.add_argument(
        "--tau", type=float, default=1.0,
        help="Temperature when computing the gumbel softmax"
    )
    model_parser.add_argument(
        "--lora_alpha", default=8, type=int, help="temperature hyperparamter in Lora"
    )
    model_parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="The weight of CLIP when computing DialectCLIPForGeneration loss"
    )
    model_parser.add_argument(
        "--beta", type=float, default=1.0,
        help="The weight of causal language model loss when computing DialectCLIPForGeneration loss"
    )
    model_parser.add_argument(
        "--output_attentions", action="store_true",
        help="Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail."
    )
    model_parser.add_argument(
        "--return_dict", action="store_true",
        help="Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple"
    )
    model_parser.add_argument(
        "--is_encoder_decoder", action="store_true",
        help="The model architecture is encoder_decoder or decoder only, if --is_encoder_decoder is given, the model architecture is a transformer style (encoder-decoder) by default"
    )

    trainer_parser = parser.add_argument_group("Trainer configuration", description="config Trainer hyperparameters")
    trainer_parser.add_argument("--device", default="cuda", help="Target Device to train on")
    trainer_parser.add_argument("--epochs", type=int, default=3, help="Training Epoch")
    trainer_parser.add_argument("--batch_size", type=int, default=2, help="The number of samples per batch")
    trainer_parser.add_argument("--accum_iter", default=16, type=int, help="accumulation step")
    trainer_parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the dataset or not")
    trainer_parser.add_argument("--num_workers", type=int, default=8, help="Number of process for data loading")
    trainer_parser.add_argument("--learning_rate", type=float, default=1e-4)
    trainer_parser.add_argument("--weight_decay_rate", type=float, default=0.001)
    trainer_parser.add_argument("--model_save_path", type=str, default="./checkpoint/dialect_clip.pth")
    trainer_parser.add_argument("--save_checkpoint_frequency", type=int, default=20, help="How long to save checkpoint")

    generation_parser = parser.add_argument_group("Generate Configuration", description="config generation parameters")
    generation_parser.add_argument("--do_sample", action="store_true", help="Whether to sample tokens when generating")
    generation_parser.add_argument("--temperature")
    generation_parser.add_argument("--num_beams", default=1, type=int, help="Beam search number")
    generation_parser.add_argument("--max_length", type=int, default=128, help="maximum length for generation")

    args = parser.parse_args()
    return args
    

def main():
    args = arg_parser()
    model_config = DialectCLIPConfig()
    model_config.SPEECH_MODEL_NAME = args.speech_model if args.speech_model is not None \
        else model_config.AUDIO_MODEL_NAME
    model_config.TEXT_MODEL_NAME = args.language_model if args.language_model is not None \
        else model_config.TEXT_MODEL_NAME
    model_config.sampling_rate = args.sampling_rate if args.sampling_rate is not None \
        else model_config.sampling_rate
    model_config.logit_scale_init_value = args.logit_scale_init_value if args.logit_scale_init_value is not None \
        else model_config.logit_scale_init_value
    model_config.initializer_range = args.initializer_range if args.initializer_range is not None \
        else model_config.initializer_range
    model_config.lora_rank = args.lora_rank if args.lora_rank is not None else model_config.lora_rank
    model_config.lora_dropout = args.lora_dropout if args.lora_dropout is not None else model_config.lora_dropout
    model_config.tau = args.tau if args.tau is not None else model_config.tau
    model_config.lora_alpha = args.lora_alpha if args.lora_alpha is not None else model_config.lora_alpha
    model_config.alpha = args.alpha if args.alpha is not None else model_config.alpha
    model_config.beta = args.beta if args.beta is not None else model_config.beta
    model_config.output_attentions = args.output_attentions if args.output_attentions is not None \
        else model_config.output_attentions
    model_config.return_dict = args.return_dict if args.return_dict is not None \
        else model_config.return_dict
    model_config.is_encoder_decoder = args.is_encoder_decoder if args.is_encoder_decoder is not None \
        else model_config.is_encoder_decoder
    model_config.do_sample = args.do_sample if args.do_sample is not None else model_config.do_sample
    model_config.num_beams = args.num_beams if args.num_beams is not None else model_config.num_beams
    model_config.max_length = args.max_length if args.max_length is not None else model_config.max_length
    model = DialectCLIP(config=model_config)

    train_config = DialectCLIPTrainerConfig()
    train_config.device = args.device if args.device is not None else train_config.device
    train_config.epochs = args.epochs if args.epochs is not None else train_config.epochs
    train_config.batch_size = args.batch_size if args.batch_size is not None else train_config.batch_size
    train_config.shuffle = args.shuffle if args.shuffle is not None else train_config.shuffle
    train_config.num_workers = args.num_workers if args.num_workers is not None else train_config.num_workers
    train_config.learning_rate = args.learning_rate if args.learning_rate is not None else train_config.learning_rate
    train_config.weight_decay_rate = args.weight_decay_rate if args.weight_decay_rate is not None else train_config.weight_decay_rate
    train_config.model_save_path = args.model_save_path if args.model_save_path is not None else train_config.model_save_path
    train_config.save_checkpoint_frequency = args.save_checkpoint_frequency if args.save_checkpoint_frequency is not None else train_config.save_checkpoint_frequency

    dialectclip_trainer = DialectCLIPTrainer(model=model, config=train_config)
    dialectclip_evaluator = DialectCLIPEvaluator(model=model, device=train_config.device)
    

    if args.dataset == "Uyghur":
        dataset = Uyghur_Dataset(file_path="./Data/Uyghur_Chinese")
    elif args.dataset == "temp":
        dataset = temp_dataset(file_path="./Data/Uyghur_Chinese")
    else:
        raise FileNotFoundError(f"Dataset {args.dataset} not support")
    
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    dialectclip_trainer(
        train_dataset=train_dataset,
    )
    dialectclip_evaluator(
        dataset=test_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers
    )


if __name__ == "__main__":
    main()
