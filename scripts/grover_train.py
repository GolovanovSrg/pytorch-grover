import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.append('../grover-pytorch')
from lm.modeling import GroverConfig, GroverModel, get_grover_model  # noqa
from lm.utils import load_tf_checkpoint  # noqa
from lm.datasets import JsonlTitleDataset  # noqa
from lm.trainers import Trainer  # noqa
from lm.utils import set_seed  # noqa

sys.path.append('../grover')
from grover.sample.encoder import Encoder, get_encoder  # noqa


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Pytorch training: '
                                                 'python -m torch.distributed.launch'
                                                 '--nproc_per_node=NUM_GPUS_YOU_HAVE SCRIPT.py'
                                                 '(--arg1 --arg2 --arg3 and all otherarguments of your training script')

    parser.add_argument('--data_path', type=Path, help='Path to jsonl with train and validation data')
    parser.add_argument('--tf_checkpoint_path', type=str, help='Path to tf checkpoint of model')
    parser.add_argument('--model_config_path', type=Path, help='Path to config of model')
    parser.add_argument('--adapter_mode', action='store_true', help='Use adapter mode for model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random state')
    parser.add_argument('--chunk_size', type=int, default=2, help='Max number of items in model forward')
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of threads in data loader')
    parser.add_argument('--label_smoothing', type=float, default=0, help='Label smoothing coefficient for loss')
    parser.add_argument('--lr', type=float, default=6.25e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--opt_level', type=str, default='O2', help='Level of apex.amp optimizations')
    parser.add_argument('--loss_scale', type=float, default=None, help='Loss scale for apex.amp')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of items in batch')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--last_checkpoint_path', type=Path, default=None, help='Path to last checkpoint of model')
    parser.add_argument('--best_checkpoint_path', type=Path, default='checkpoints/best_checkpoint.pt',
                        help='Path to best checkpoint of model')
    parser.add_argument("--local_rank", type=int, default=0, help='Rank for distributed training')

    return parser


def get_model(params: argparse.Namespace) -> GroverModel:
    model_config = GroverConfig.from_json_file(str(params.model_config_path))
    model = get_grover_model(model_config, params.adapter_mode)
    model_state = load_tf_checkpoint(params.tf_checkpoint_path, model_config.num_hidden_layers)
    model.load_state_dict(model_state, strict=(not model.adapter_mode))

    return model


def get_trainer(model: GroverModel, params: argparse.Namespace) -> Trainer:
    trainer = Trainer(model=model,
                      chunk_size=params.chunk_size,
                      optimizer_params={'lr': params.lr, 'weight_decay': params.weight_decay},
                      loss_params={'smoothing': params.label_smoothing},
                      amp_params={'opt_level': params.opt_level, 'loss_scale': params.loss_scale},
                      n_jobs=params.n_jobs,
                      rank=params.local_rank)

    return trainer


def get_datasets(vocab: Encoder, params: argparse.Namespace) -> Tuple[JsonlTitleDataset, JsonlTitleDataset]:
    train_dataset = JsonlTitleDataset(jsonl_path=params.data_path,
                                      vocab=vocab,
                                      split='train')
    val_dataset = JsonlTitleDataset(jsonl_path=params.data_path,
                                    vocab=vocab,
                                    split='val')

    return train_dataset, val_dataset


def main(args: argparse.Namespace) -> None:
    torch.multiprocessing.set_start_method('spawn')
    torch.distributed.init_process_group(backend="nccl")

    set_seed(args.seed + torch.distributed.get_rank())

    vocab = get_encoder()
    model = get_model(args)
    train_dataset, val_dataset = get_datasets(vocab, args)
    trainer = get_trainer(model, args)

    trainer.train(train_data=train_dataset,
                  test_data=val_dataset,
                  batch_size=args.batch_size,
                  n_epochs=args.n_epochs,
                  last_checkpoint_path=args.last_checkpoint_path,
                  best_checkpoint_path=args.best_checkpoint_path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
