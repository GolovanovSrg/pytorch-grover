from pathlib import Path
from typing import Any, Iterator, Iterable, Dict, Optional, List

import torch
from apex import amp, parallel
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .optim import Adafactor
from .losses import LabelSmoothingLoss
from .utils import pad_sequence, chunks


class AvgMeter:
    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        self._sum += value
        self._count += 1

    def __call__(self) -> float:
        if self._count:
            return self._sum / self._count
        return 0


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self) -> Iterator[List[int]]:
        buckets: Dict[int, int] = {}
        for idx in self.sampler:
            length = len(self.sampler.dataset[idx])
            bucket_id = length // 64

            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)

            if len(buckets[bucket_id]) == self.batch_size:
                yield buckets[bucket_id]
                buckets[bucket_id] = []

        leftover = [idx for bucket in buckets.values() for idx in bucket]
        batch: List[int] = []
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class Trainer:
    def __init__(self, model: torch.nn.Module, chunk_size: int, optimizer_params: Dict[str, float] = {},
                 loss_params: Dict[str, float] = {}, amp_params: Dict[str, Any] = {},
                 n_jobs: int = 0, rank: int = 0) -> None:
        assert torch.cuda.is_available() and torch.distributed.is_initialized(), \
            'Only distributed gpu training is supported'

        torch.cuda.set_device(rank)
        device = torch.device('cuda:' + str(rank))
        is_master = torch.distributed.get_rank() == 0

        smoothing = loss_params.get('smoothing', 0)
        lr = optimizer_params.get('lr', 1e-3)
        weight_decay = optimizer_params.get('weight_decay', 0)
        opt_level = amp_params.get('opt_level', 'O0')
        loss_scale = amp_params.get('loss_scale', None)

        self.padding_idx = model.embedding.tok_padding_idx
        self.model = model.to(device)
        self.lm_criterion = LabelSmoothingLoss(n_labels=self.model.embedding.n_embeddings,
                                               ignore_index=self.padding_idx,
                                               smoothing=smoothing).to(device)

        if self.model.adapter_mode:
            param_optimizer = self.model.adapter_named_parameters()
        else:
            param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]

        self.optimizer = Adafactor(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        # TODO: fix loss scaling
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level=opt_level,
                                                    master_weights=False,  # Adafactor supports fp16
                                                    cast_model_outputs=torch.float32,
                                                    loss_scale=loss_scale)
        self.model = parallel.DistributedDataParallel(self.model, delay_allreduce=True)

        self.chunk_size = chunk_size
        self.last_epoch = 0
        self.device = device
        self.is_master = is_master
        self.n_jobs = n_jobs

    def _save_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def _collate_func(self, data: Iterable[torch.Tensor]) -> torch.Tensor:
        chunks_list = [pad_sequence(c, batch_first=True, padding_value=self.padding_idx, left=True)
                       for c in chunks(data, self.chunk_size)]
        return chunks_list

    def _train_epoch(self, train_dataloader: DataLoader) -> None:
        if self.is_master:
            pbar = tqdm(desc=f'Train, epoch #{self.last_epoch}', total=len(train_dataloader))

        self.model.train()

        lm_loss = AvgMeter()
        for chunks_list in train_dataloader:
            self.optimizer.zero_grad()

            for chunk in chunks_list:
                chunk = chunk.to(self.device)

                logits, _, _ = self.model(chunk[:, :-1], return_past=False)
                chunk_lm_loss = self.lm_criterion(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1))
                full_loss = chunk_lm_loss / len(chunks_list)
                with amp.scale_loss(full_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                lm_loss.update(full_loss.item())

            self.optimizer.step()

            info_tensor = torch.tensor([lm_loss()], device=self.device)
            torch.distributed.reduce(info_tensor, dst=0)

            if self.is_master:
                info_tensor = info_tensor / torch.distributed.get_world_size()
                pbar.update(1)
                pbar.set_postfix({'lm_loss': info_tensor[0].item()})

    def _test_epoch(self, test_dataloader: DataLoader) -> float:
        with torch.no_grad():
            if self.is_master:
                pbar = tqdm(desc=f'Test, epoch #{self.last_epoch}', total=len(test_dataloader))

            self.model.eval()

            lm_loss = AvgMeter()
            for chunks_list in test_dataloader:
                for chunk in chunks_list:
                    chunk = chunk.to(self.device)

                    logits, _, _ = self.model(chunk[:, :-1], return_past=False)
                    chunk_lm_loss = self.lm_criterion(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1))
                    full_loss = chunk_lm_loss / len(chunks_list)

                    lm_loss.update(full_loss.item())

                info_tensor = torch.tensor([lm_loss()], device=self.device)
                torch.distributed.reduce(info_tensor, dst=0)

                if self.is_master:
                    info_tensor = info_tensor / torch.distributed.get_world_size()
                    pbar.update(1)
                    pbar.set_postfix({'lm_loss': info_tensor[0].item()})
                    quality_metric = info_tensor[0].item()

            return -quality_metric

    def train(self, train_data: Dataset, n_epochs: int, batch_size: int, test_data: Optional[Dataset] = None,
              last_checkpoint_path: Path = None, best_checkpoint_path: Path = None) -> None:

        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        batch_size = batch_size // num_replicas

        train_sampler = DistributedSampler(train_data, shuffle=True, num_replicas=num_replicas, rank=rank)
        train_batch_sampler = LenMatchBatchSampler(train_sampler, batch_size=batch_size, drop_last=False)
        train_dataloader = DataLoader(train_data, batch_sampler=train_batch_sampler, collate_fn=self._collate_func,
                                      num_workers=self.n_jobs)

        if test_data is not None:
            test_sampler = DistributedSampler(test_data, shuffle=False, num_replicas=num_replicas, rank=rank)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                                         collate_fn=self._collate_func, num_workers=self.n_jobs)

        best_metric = float("-inf")
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None:
                torch.cuda.empty_cache()
                metric = self._test_epoch(test_dataloader)

                if best_checkpoint_path is not None:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1
