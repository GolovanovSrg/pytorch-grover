import json

from pathlib import Path
from typing import Any, Optional, List, Dict

import torch
from torch.utils.data import Dataset


class JsonlTitleDataset(Dataset):
    @staticmethod
    def _read_jsonl(path: Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
        with open(path, 'r') as file:
            data = [json.loads(line) for line in file]

        if split is not None:
            return [item for item in data if item['split'] == split]

        return data

    @staticmethod
    def _tokenize_item_pieces(encoder: Any, item: Dict[str, Any]) -> Dict[str, List[int]]:
        article_pieces = {
            'article': [encoder.begin_article] + encoder.encode(item['text']) + [encoder.end_article],
            'domain': [encoder.begin_domain] + encoder.encode(item['domain']) + [encoder.end_domain],
            'title': [encoder.begin_title] + encoder.encode(item['title']) + [encoder.end_title],
        }

        # date
        date_split = item['publish_date'].split('-')
        assert len(date_split) == 3
        assert date_split[0].isdigit()

        date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                    'August', 'September', 'October', 'November', 'December'][
                       int(date_split[0]) - 1] + ' {}, {}'.format(date_split[1], date_split[2])
        article_pieces['date'] = [encoder.begin_date] + encoder.encode(date_txt) + [encoder.end_date]

        # authors
        authors = ', '.join(item['authors'])
        if len(authors) > 5:
            article_pieces['authors'] = [encoder.begin_authors] + encoder.encode(authors) + [encoder.end_authors]
        return article_pieces

    @staticmethod
    def _get_formatted_item(item_pieces: Dict[str, List[int]], max_len: int) -> List[int]:

        context_formatted = []
        for key in ['domain', 'date', 'authors', 'article']:
            context_formatted.extend(item_pieces.get(key, []))

        max_context_len = max_len - len(item_pieces['title'])
        if len(context_formatted) >= max_context_len:
            context_formatted = context_formatted[:max_context_len] + [context_formatted[-1]]
        context_formatted.extend(item_pieces['title'])

        return context_formatted

    def __init__(self, jsonl_path: Path, vocab: Any, max_len: int = 1024, split: Optional[str] = None) -> None:
        super().__init__()

        if split is not None:
            assert split in ['train', 'val']

        self.vocab = vocab
        self.max_len = max_len
        self.data = JsonlTitleDataset._read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.data[idx]
        item_pieces = JsonlTitleDataset._tokenize_item_pieces(self.vocab, item)
        tokens = JsonlTitleDataset._get_formatted_item(item_pieces, self.max_len)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return tokens_tensor
