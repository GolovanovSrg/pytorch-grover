from typing import Dict, Any
from typing import List

import spacy


class SpacyTokenizer:
    def __init__(self) -> None:
        self.tokenizer = spacy.load('en',
                                    disable=['parser', 'tagger', 'ner', 'textcat'])

    def __call__(self, string: str, normalize: bool = True) -> List[str]:
        string = string.lower()
        words = [t.text.strip() for t in self.tokenizer(string) if t.text.strip()]

        return words


def print_metrics(scores: Dict[str, Any]) -> None:
    bertscores = scores['bertscore']
    scheme = 'bertscore (f/p/r): {:05.3f} ± {:05.3f} / ' \
             '{:05.3f} ± {:05.3f} / {:05.3f} ± {:05.3f}'

    message = scheme.format(
        bertscores['f_mean'], bertscores['f_std'],
        bertscores['p_mean'], bertscores['p_std'],
        bertscores['r_mean'], bertscores['r_std']
    )
    print(message)

    def _print_rouge(name: str) -> None:
        message = '{} (f/p/r): {:05.3f} {:05.3f} {:05.3f}'.format(
            name,
            scores['rouge'][name]['f'],
            scores['rouge'][name]['p'],
            scores['rouge'][name]['r'],
        )
        print(message)

    for name in ('rouge-1', 'rouge-2', 'rouge-l'):
        _print_rouge(name)

    print('meteor: {:05.3f}'.format(scores['meteor']))
