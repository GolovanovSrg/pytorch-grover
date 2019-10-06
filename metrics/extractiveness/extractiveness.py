from nltk import ngrams
import spacy
import json
from typing import List, Any, Union
from pathlib import Path


class ExtractivenessScorer:
    def __init__(self) -> None:
        self.spacy_model = spacy.load('en', disable=['parser', 'ner'])

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        parsed_text = self.spacy_model(text.lower().strip())
        return [token.lemma_ for token in parsed_text]

    def score(self, source_text: str, target_text: str, n: int = 4) -> float:
        common_ngrams_count = 0
        tokenized_source_text: List[str] = self.tokenize_and_lemmatize(source_text)
        tokenized_target_text: List[str] = self.tokenize_and_lemmatize(target_text)
        source_length: int = len(tokenized_source_text)
        target_length: int = len(tokenized_target_text)
        max_ngram_size: int = min(source_length, target_length) + 1
        processed_source_text: str = " ".join(tokenized_source_text)
        ngrams_count = 0
        for ngram_size in range(n, max_ngram_size):
            for ngram in ngrams(tokenized_target_text, ngram_size):
                if " ".join(ngram) in processed_source_text:
                    common_ngrams_count += 1
                ngrams_count += 1
        return common_ngrams_count / ngrams_count if ngrams_count else 0.0


def extractiveness(source_data_path: Union[str, Path], target_text_path: Union[str, Path], avg: bool) -> Any:
    scores = []
    scorer = ExtractivenessScorer()
    with open(source_data_path, encoding='utf-8') as source_data_input_stream, \
            open(target_text_path, encoding='utf-8') as target_text_input_stream:
        for source_data, target_text in zip(source_data_input_stream, target_text_input_stream):
            source_text = json.loads(source_data)['text']
            scores.append(scorer.score(source_text, target_text))

    if avg:
        return sum(scores) / len(scores)
    return scores
