from pathlib import Path
from typing import Dict, Union, List
from bert_score import score

BertScoreSubType = Dict[str, Dict[str, float]]
BertScoreType = Union[BertScoreSubType, List[BertScoreSubType]]


def bertscore(references_path: Union[str, Path], candidates_path: Union[str, Path],
              avg: bool = True) -> BertScoreType:
    with open(candidates_path) as f:
        cands = [line.strip() for line in f]

    with open(references_path) as f:
        refs = [line.strip() for line in f]

    p, r, f1 = score(cands, refs, bert='bert-base-uncased')
    p, r, f1 = p.numpy(), r.numpy(), f1.numpy()

    if avg:
        res = {'p_mean': p.mean(), 'p_std': p.std(),
               'r_mean': r.mean(), 'r_std': r.std(),
               'f_mean': f1.mean(), 'f_std': f1.std()}
    else:
        res = {'p': p, 'r': r, 'f': f1}

    return res
