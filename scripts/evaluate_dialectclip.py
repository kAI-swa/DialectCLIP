import argparse
from torch.utils.data import random_split
from dataset_construct import *
from dialect_clip.modeling_dialectclip import DialectCLIPForConditionalGeneration
from dialect_clip.configuration_dialectclip import DialectCLIPConfig
from dialect_clip.evaluator_dialectclip import DialectCLIPEvaluator


def arg_parser():
    parser = argparse.ArgumentParser(
        prog="DialectCLIP",
        description="Evalute DialectCLIP",
        epilog="Thank you for using %(prog)s"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["Uyghur", "temp"], help="Dataset for running DialectCLIP"
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Target device to run on"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="whether to sample when generate new tokens"
    )
    parser.add_argument(
         "--temperature", type=float, default=None, help="temperature for sample generation"
    )
    parser.add_argument(
         "--num_beams", type=int, default=1, help="Number of beams when doing beam search"
    )
    parser.add_argument(
         "--max_length", type=int, default=128, help="Maximum length when generating"
    )
    parser.add_argument(
         "--batch_size",  type=int, default=16
    )
    parser.add_argument(
         "--num_workers", type=int, default=0, help="Number of process for preprocessing data"
    )
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    model = DialectCLIPForConditionalGeneration(config=DialectCLIPConfig())
    device = args.device

    if args.dataset == "Uyghur":
        dataset = Uyghur_Dataset(file_path="./Data/Uyghur_Chinese")
    elif args.dataset == "temp":
        dataset = temp_dataset(file_path="./Data/Uyghur_Chinese")
    else:
        raise FileNotFoundError(f"Dataset {args.dataset} not support")
    
    _, test_dataset = random_split(dataset, [0.99, 0.01])

    evaluator = DialectCLIPEvaluator(model=model, device=device)
    evaluator(
        dataset=test_dataset,
        do_sample=args.do_sample,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

        
if __name__ == "__main__":
    main()
