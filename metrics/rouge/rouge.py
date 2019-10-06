import sys
import rouge as rouge_lib

from pathlib import Path
from typing import Dict, List, Union

from ..utils import SpacyTokenizer

sys.setrecursionlimit(20000)

RougeScoreSubType = Dict[str, Dict[str, float]]
RougeScoreType = Union[RougeScoreSubType, List[RougeScoreSubType]]


def rouge(references_path: Union[str, Path], candidates_path: Union[str, Path], avg: bool = True) -> RougeScoreType:
    tokenizer = SpacyTokenizer()

    with open(candidates_path, encoding="utf-8", mode="r") as hyp_file:
        hyps = [line[:-1] for line in hyp_file]
        hyps = [' '.join(tokenizer(h)) for h in hyps]

    with open(references_path, encoding="utf-8", mode="r") as ref_file:
        refs = [line[:-1] for line in ref_file]
        refs = [' '.join(tokenizer(r)) for r in refs]

    filtered_hyps, filtered_refs = zip(*[[h, r] for h, r in zip(hyps, refs) if len(h) and len(r)])
    assert len(filtered_hyps) == len(filtered_refs)

    r = rouge_lib.Rouge()
    scores = r.get_scores(filtered_hyps, filtered_refs, avg=avg)

    return scores
