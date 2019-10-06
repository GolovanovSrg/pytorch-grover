import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append('../')
from metrics import meteor, rouge, bertscore, extractiveness  # noqa
from metrics.utils import print_metrics  # noqa


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--references_path', type=Path, default='references.txt',
                        help='Path of file with referenced sentences')
    parser.add_argument('--candidates_path', type=Path, default='candidates.txt',
                        help='Path of file with generated sentences')
    parser.add_argument('--no_avg', action='store_true',
                        help='Return metrics for each sentence')
    parser.add_argument('--output_path', type=Optional[Path], default=None,
                        help='Write result as json file instead printing')
    parser.add_argument('--source_data_path', type=Path, default='for_inference_100k.jsonl',
                        help='Path to the source file in json format')

    return parser


def main(args: argparse.Namespace) -> None:
    avg = not args.no_avg
    meteor_score = meteor(args.references_path, args.candidates_path, avg=avg)
    rouge_score = rouge(args.references_path, args.candidates_path, avg=avg)
    extractiveness_score = extractiveness(args.soure_data_path, args.references_path, avg=args.avg)
    bert_score = bertscore(args.references_path, args.candidates_path, avg=avg)

    scores = {'meteor': meteor_score,
              'rouge': rouge_score,
              'bertscore': bert_score,
              'extractiveness': extractiveness_score}

    if args.output_path is None:
        print_metrics(scores)
    else:
        with open(args.output_path, 'w') as file:
            json.dump(scores, file)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_known_args()[0]
    main(args)
