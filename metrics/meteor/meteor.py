import os
import subprocess
import tempfile

from pathlib import Path
from typing import Union, List

from ..utils import SpacyTokenizer


MeteorScoreType = Union[float, List[float]]


def meteor(references_path: Union[str, Path], candidates_path: Union[str, Path], avg: bool = True) -> MeteorScoreType:
    tokenizer = SpacyTokenizer()

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as cand_temp, \
            tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as ref_temp:
        with open(candidates_path, encoding='utf-8', mode='r') as hyp_file:
            hyps = [line[:-1] for line in hyp_file]
            hyps = [' '.join(tokenizer(h)) for h in hyps]
            cand_temp.write('\n'.join(hyps))
            cand_temp.flush()

        with open(references_path, encoding='utf-8', mode='r') as ref_file:
            refs = [line[:-1] for line in ref_file]
            refs = [' '.join(tokenizer(r)) for r in refs]
            ref_temp.write('\n'.join(refs))
            ref_temp.flush()

        cmd = ['java', '-Xmx2G', '-jar', os.path.join(os.path.dirname(__file__), 'meteor-1.5.jar'),
               cand_temp.name, ref_temp.name, '-l', 'en', '-norm']
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        if avg:
            for line in res.stdout.decode().split('\n'):
                if line.startswith('Final score:'):
                    return float(line.split()[-1])
        else:
            scores = []
            for line in res.stdout.decode().split('\n'):
                if line.startswith('Segment'):
                    segment_score = float(line.split()[-1])
                    scores.append(segment_score)

            return scores

        raise RuntimeError(f'meteor-1.5.jar returns unexpected message. cmd = {" ".join(cmd)}, '
                           f'stdout = {res.stdout.decode()}, stderr = {res.stderr.decode()}')
