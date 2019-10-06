import argparse
import copy
import json
import sys
import random

import numpy as np
import tensorflow as tf
import torch

from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Union, Callable


TF_BACKEND = True
if TF_BACKEND:
    sys.path.append('../grover')
    from lm.modeling import GroverConfig, sample  # noqa
else:
    sys.path.append('../grover-pytorch')
    from lm.modeling import GroverConfig, get_grover_model, sample_sequences, beamsearch_sequences  # noqa
    from lm.utils import load_tf_checkpoint  # noqa

sys.path.append('../grover')
from grover.sample.encoder import Encoder, get_encoder, _tokenize_article_pieces, extract_generated_target  # noqa


ArticleType = Dict[str, Union[str, int, float, List[str]]]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Contextual generation')
    parser.add_argument(
        '-metadata_fn',
        dest='metadata_fn',
        type=Path,
        help='Path to a JSONL containing metadata',
    )
    parser.add_argument(
        '-out_fn',
        dest='out_fn',
        type=str,
        help='Out jsonl, which will contain the completed jsons',
    )
    parser.add_argument(
        '-model_config_fn',
        dest='model_config_fn',
        default='../lm/configs/base.json',
        type=str,
        help='Configuration JSON for the model',
    )
    parser.add_argument(
        '-model_ckpt',
        dest='model_ckpt',
        default='../models/base/model.ckpt',
        type=str,
        help='checkpoint file for the model',
    )
    parser.add_argument(
        '-n_samples',
        dest='n_samples',
        default=1,
        type=int,
        help='How many things to generate per context. will split into chunks if need be',
    )
    parser.add_argument(
        '-num_folds',
        dest='num_folds',
        default=1,
        type=int,
        help='Number of folds. useful if we want to split up a big file into multiple jobs.',
    )
    parser.add_argument(
        '-fold',
        dest='fold',
        default=0,
        type=int,
        help='Which fold we are on. useful if we want to split up a big file into multiple jobs.'
    )
    parser.add_argument(
        '-batch_size',
        dest='batch_size',
        default=None,
        type=int,
        help='Batch size. You can leave this out and we will infer one based on the number of hidden layers',
    )
    parser.add_argument(
        '-top_p',
        dest='top_p',
        default=0.95,
        type=float,
        help='p to use for top p sampling. if this isn\'t none, use this for everthing'
    )
    parser.add_argument(
        '-candidates_path',
        dest='candidates_path',
        default='candidates.txt',
        type=Path,
        help='Path of candidates file for metrics',
    )
    parser.add_argument(
        '-references_path',
        dest='references_path',
        default='references.txt',
        type=Path,
        help='Path of references file for metrics',
    )
    parser.add_argument(
        '-subsample_size',
        dest='subsample_size',
        default=-1,
        type=int,
        help='Get only part of JSONL containing metadata',
    )
    parser.add_argument(
        '-max_sample_iter',
        dest='max_sample_iter',
        default=100,
        type=int,
        help='Max iterations of sampling per article',
    )
    parser.add_argument(
        '-max_target_len',
        dest='max_target_len',
        default=256,
        type=int,
        help='Max number of symbols in target',
    )
    parser.add_argument(
        '-beam_size',
        dest='beam_size',
        default=None,
        type=int,
        help='Size of beam in beam-search',
    )
    parser.add_argument(
        '-length_penalty',
        dest='length_penalty',
        default=0.8,
        type=int,
        help='Length penalty in beam-search',
    )

    return parser


def load_articles(path: str, subsample_size: int, num_folds: int, fold: int) -> List[ArticleType]:
    with open(path, 'r') as file:
        articles = [json.loads(line) for line in file]

    if subsample_size > 0:
        if len(articles) < subsample_size:
            subsample_size = len(articles)
        articles = random.sample(articles, k=subsample_size)

    articles = [a for i, a in enumerate(articles) if i % num_folds == fold]

    return articles


def get_formatted_article(encoder: Encoder, article: ArticleType, target: str,
                          max_target_len: int, max_input_len: int = 1024) -> List[int]:
    article_pieces = _tokenize_article_pieces(encoder, article)

    context_formatted: List[int] = []
    for key in ['domain', 'date', 'authors', 'title', 'article']:
        if key != target:
            context_formatted.extend(article_pieces.get(key, []))

    max_context_len = max_input_len - max_target_len // 2
    if len(context_formatted) >= max_context_len:
        context_formatted = context_formatted[:max_context_len] + [context_formatted[-1]]

    context_formatted.append(encoder.__dict__['begin_{}'.format(target)])

    return context_formatted


def extract_target(encoder: Encoder, tokens: np.ndarray, target: str) -> str:
    extraction = extract_generated_target(output_tokens=tokens, encoder=encoder, target=target)
    target_text = extraction['extraction']

    return target_text


def build_tf_sampler(session: tf.Session, encoder: Encoder, news_config: GroverConfig,
                     args: argparse.Namespace, batch_size: int) -> Callable[[List[ArticleType]], List[List[str]]]:
    assert args.beam_size is None, 'Beam-search is implemented only for pytorch'

    top_p = np.ones(batch_size, dtype=np.float32) * args.top_p
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_{}'.format(args.target)]] = 0

    initial_context = tf.placeholder(tf.int32, [batch_size, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                           do_topk=False)

    def run_sample(context_formatted: List[int], n: int) -> np.ndarray:
        assert n == batch_size
        tokens_out, _ = session.run([tokens, probs],
                                    feed_dict={initial_context: [context_formatted] * n,
                                               eos_token: encoder.__dict__['end_{}'.format(args.target)],
                                               ignore_ids: ignore_ids_np,
                                               p_for_topp: top_p})

    def sampler(articles_batch: List[ArticleType]) -> List[List[str]]:
        contexts = [get_formatted_article(encoder, a, args.target, args.max_target_len) for a in articles_batch]

        gens_batch: List[List[str]] = [[] for _ in range(len(contexts))]
        for i, context in enumerate(contexts):
            counter = 0
            while len(gens_batch[i]) < args.n_samples:
                counter += 1
                if counter > args.n_samples + args.max_sample_iter:
                    break

                tokens_batch = run_sample(context, batch_size)

                for tokens in tokens_batch:
                    text = extract_target(encoder, tokens, args.target)
                    if len(text) <= args.max_target_len and len(gens_batch[i]) < args.n_samples:
                        gens_batch[i].append(text)

        return gens_batch

    return sampler


def build_pytorch_sampler(encoder: Encoder, news_config: GroverConfig, args: argparse.Namespace) -> \
        Callable[[List[ArticleType]], List[List[str]]]:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    news_config = GroverConfig.from_json_file(args.model_config_fn)
    model = get_grover_model(news_config)
    model_state = load_tf_checkpoint(args.model_ckpt, news_config.num_hidden_layers)
    model.load_state_dict(model_state)
    model = model.to(device)

    bos_id = encoder.__dict__['begin_{}'.format(args.target)]
    eos_id = encoder.__dict__['end_{}'.format(args.target)]
    ignore_ids = [i for i, b in enumerate(encoder.special_tokens_onehot) if (b and i != eos_id)]
    ignore_ids_tensor = torch.tensor(ignore_ids, dtype=torch.long, device=device)

    def pad_and_sample(contexts: List[List[int]]) -> List[np.ndarray]:
        contexts_tensor = pad_sequence_left(contexts, encoder.encoder['<|padding|>'], device)
        if args.beam_size is None:
            sequences = sample_sequences(model, contexts_tensor, bos_id, eos_id, news_config.max_position_embeddings,
                                         ignore_ids_tensor, args.top_p)
        else:
            sequences = beamsearch_sequences(model, contexts_tensor, bos_id, eos_id, beam_size=args.beam_size,
                                             vocab_size=news_config.vocab_size, length_penalty_coef=args.length_penalty,
                                             diversity_coef=0, diversity_groups=1, return_beams=False,
                                             max_seq_len=news_config.max_position_embeddings,
                                             ignore_ids=ignore_ids_tensor)

        return [s.cpu().numpy() for s in sequences]

    def sampler(articles_batch: List[ArticleType]) -> List[List[str]]:
        contexts = [get_formatted_article(encoder, a, args.target, args.max_target_len) for a in articles_batch]

        gens_batch: List[List[str]] = [[] for _ in range(len(contexts))]
        indexes = list(range(len(contexts)))
        counter = 0
        while len(indexes):
            counter += 1
            if counter > args.n_samples + args.max_sample_iter:
                break

            current_contexts = [contexts[idx] for idx in indexes]
            tokens_batch = pad_and_sample(current_contexts)

            for idx, tokens in zip(indexes, tokens_batch):
                text = extract_target(encoder, tokens, args.target)
                if args.beam_size is not None or len(text) <= args.max_target_len:
                    gens_batch[idx].append(text)

            indexes = [idx for idx in indexes if len(gens_batch[idx]) < args.n_samples]

        return gens_batch

    return sampler


def chunks(sequence: List[Any], chunk_size: int) -> List[List[Any]]:
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]


def pad_sequence_left(sequences: List[List[int]], padding_value: int, device: torch.device,
                      dtype: torch.dtype = torch.long) -> torch.Tensor:
    max_len = max([len(s) for s in sequences])
    new_sequences = [[padding_value] * (max_len - len(s)) + s for s in sequences]
    tensor = torch.tensor(new_sequences, dtype=dtype, device=device)
    return tensor


def process_pages(sampler: Callable[[List[ArticleType]], List[List[str]]],
                  article: ArticleType, batch_size: int) -> ArticleType:
    if isinstance(article['text'], list):
        subarticles: List[ArticleType] = []
        for page_text in article['text']:
            subarticle = copy.deepcopy(article)
            subarticle['text'] = page_text
            subarticles.append(subarticle)

        gens: List[str] = list(sum([sum(sampler(chunk), []) for chunk in chunks(subarticles, batch_size)], []))
        article['text'] = '\n'.join(gens)

    return article


def process_articles(args: argparse.Namespace, sampler: Callable[[List[ArticleType]], List[List[str]]],
                     articles: List[ArticleType], batch_size: int) -> None:
    with open(args.out_fn, 'w') as f_out, \
        open(args.candidates_path, 'w') as candidates_out, \
            open(args.references_path, 'w') as references_out:

        for articles_batch in tqdm(chunks(articles, batch_size)):
            articles_batch = [process_pages(sampler, a, batch_size) for a in articles_batch]
            gens_batch = sampler(articles_batch)

            for gens, article in zip(gens_batch, articles_batch):
                for g in gens:
                    candidates_out.write(g.replace('\n', '   ').replace('\r', '   ') + '\n')
                    references_out.write(article[args.target].replace('\n', '   ').replace('\r', '   ') + '\n')

                article[f'gen_{args.target}'] = gens
                f_out.write(json.dumps(article) + '\n')


def generate(args: argparse.Namespace) -> None:
    encoder = get_encoder()
    news_config = GroverConfig.from_json_file(args.model_config_fn)

    default_mbs = {12: 32, 24: 16, 48: 3}
    batch_size = args.batch_size if args.batch_size is not None else default_mbs[news_config.num_hidden_layers]
    print(f'\n~~\nnumber of samples = {args.n_samples}, batch size = {batch_size}', flush=True)

    articles = load_articles(args.metadata_fn, args.subsample_size, args.num_folds, args.fold)

    if TF_BACKEND:
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
            sampler = build_tf_sampler(sess, encoder, news_config, args, batch_size)
            process_articles(args, sampler, articles, batch_size)
    else:
        sampler = build_pytorch_sampler(encoder, news_config, args)
        process_articles(args, sampler, articles, batch_size)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def main(args: argparse.Namespace) -> None:
    set_seed()
    generate(args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.target = 'title'
    main(args)
