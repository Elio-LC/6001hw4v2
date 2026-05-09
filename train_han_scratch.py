import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    ScratchBiLSTM,
    build_vocab,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    set_seed,
    split_indices,
    tokenize,
)


SENT_END = {".", "!", "?", ";", "</s>"}


def split_sentences(tokens: Sequence[str], max_sents: int, max_words: int) -> List[List[str]]:
    sentences: List[List[str]] = []
    cur: List[str] = []
    for tok in tokens:
        cur.append(tok)
        if tok in SENT_END:
            sentences.append(cur[:max_words])
            cur = []
            if len(sentences) >= max_sents:
                break
    if cur and len(sentences) < max_sents:
        sentences.append(cur[:max_words])
    if not sentences:
        sentences = [tokens[:max_words] if tokens else ["<unk>"]]
    return sentences[:max_sents]


@dataclass
class HanDataset:
    x: torch.Tensor          # [N, S, W]
    word_lens: torch.Tensor  # [N, S]
    sent_lens: torch.Tensor  # [N]
    y: Optional[torch.Tensor] = None

    def __len__(self):
        return self.x.size(0)


class HanBatcher:
    def __init__(self, dataset: HanDataset, batch_size: int, shuffle: bool, seed: int, bucket_size: int = 1024):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            rng.shuffle(idxs)
            buckets = [idxs[i : i + self.bucket_size] for i in range(0, len(idxs), self.bucket_size)]
            for b in buckets:
                b.sort(key=lambda i: int(self.dataset.sent_lens[i]), reverse=True)
            rng.shuffle(buckets)
            idxs = [i for b in buckets for i in b]
        else:
            idxs.sort(key=lambda i: int(self.dataset.sent_lens[i]), reverse=True)

        for start in range(0, len(idxs), self.batch_size):
            batch = idxs[start : start + self.batch_size]
            idx = torch.tensor(batch, dtype=torch.long)
            fields = [
                self.dataset.x.index_select(0, idx),
                self.dataset.word_lens.index_select(0, idx),
                self.dataset.sent_lens.index_select(0, idx),
            ]
            if self.dataset.y is not None:
                fields.append(self.dataset.y.index_select(0, idx))
            yield tuple(fields)


def make_han_dataset(
    texts: Sequence[str],
    labels: Optional[Sequence[int]],
    word2idx: dict,
    max_sents: int,
    max_words: int,
) -> HanDataset:
    x_rows: List[List[List[int]]] = []
    wl_rows: List[List[int]] = []
    sl_rows: List[int] = []

    for text in texts:
        tokens = tokenize(text)
        sents = split_sentences(tokens, max_sents=max_sents, max_words=max_words)
        sent_ids: List[List[int]] = []
        word_lens: List[int] = []
        for sent in sents:
            ids = [word2idx.get(t, 1) for t in sent[:max_words]]
            wl = len(ids)
            if wl < max_words:
                ids.extend([0] * (max_words - wl))
            sent_ids.append(ids)
            word_lens.append(wl)
        sl = len(sent_ids)
        if sl < max_sents:
            pad_sent = [0] * max_words
            sent_ids.extend([pad_sent] * (max_sents - sl))
            word_lens.extend([0] * (max_sents - sl))
        x_rows.append(sent_ids)
        wl_rows.append(word_lens)
        sl_rows.append(sl)

    y_tensor = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
    return HanDataset(
        x=torch.tensor(x_rows, dtype=torch.long),
        word_lens=torch.tensor(wl_rows, dtype=torch.long),
        sent_lens=torch.tensor(sl_rows, dtype=torch.long),
        y=y_tensor,
    )


def subset_han(dataset: HanDataset, idxs: Sequence[int]) -> HanDataset:
    idx = torch.tensor(idxs, dtype=torch.long)
    return HanDataset(
        x=dataset.x.index_select(0, idx),
        word_lens=dataset.word_lens.index_select(0, idx),
        sent_lens=dataset.sent_lens.index_select(0, idx),
        y=dataset.y.index_select(0, idx) if dataset.y is not None else None,
    )


class HanClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        word_hidden: int,
        sent_hidden: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.word_encoder = ScratchBiLSTM(emb_dim, word_hidden, dropout)
        self.word_attn = nn.Sequential(
            nn.Linear(word_hidden * 2, word_hidden),
            nn.Tanh(),
            nn.Linear(word_hidden, 1),
        )
        self.sent_encoder = ScratchBiLSTM(word_hidden * 2, sent_hidden, dropout)
        self.sent_attn = nn.Sequential(
            nn.Linear(sent_hidden * 2, sent_hidden),
            nn.Tanh(),
            nn.Linear(sent_hidden, 1),
        )
        feat_dim = sent_hidden * 2 * 3
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, sent_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sent_hidden * 2, sent_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(sent_hidden, 1),
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

    @staticmethod
    def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int):
        scores = scores.masked_fill(~mask, -1e4)
        probs = F.softmax(scores, dim=dim)
        probs = probs * mask.to(probs.dtype)
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, word_lens: torch.Tensor, sent_lens: torch.Tensor):
        # x: [B, S, W]
        bsz, max_s, max_w = x.shape
        flat_x = x.view(bsz * max_s, max_w)
        flat_wlens = word_lens.view(-1)
        flat_wlens_clamped = flat_wlens.clamp_min(1)

        emb = self.emb_dropout(self.embedding(flat_x))
        word_out, _ = self.word_encoder(emb, flat_wlens_clamped)
        word_mask = (
            torch.arange(max_w, device=x.device).unsqueeze(0) < flat_wlens.unsqueeze(1)
        )
        word_scores = self.word_attn(word_out).squeeze(2)
        word_alpha = self.masked_softmax(word_scores, word_mask, dim=1).unsqueeze(2)
        sent_vec = torch.sum(word_out * word_alpha, dim=1)
        sent_vec = sent_vec * (flat_wlens > 0).to(sent_vec.dtype).unsqueeze(1)
        sent_vec = sent_vec.view(bsz, max_s, -1)

        sent_out, sent_last = self.sent_encoder(sent_vec, sent_lens.clamp_min(1))
        sent_mask = (
            torch.arange(max_s, device=x.device).unsqueeze(0) < sent_lens.unsqueeze(1)
        )
        sent_scores = self.sent_attn(sent_out).squeeze(2)
        sent_alpha = self.masked_softmax(sent_scores, sent_mask, dim=1).unsqueeze(2)
        attn_pool = torch.sum(sent_out * sent_alpha, dim=1)
        max_pool = sent_out.masked_fill(~sent_mask.unsqueeze(2), -1e4).max(dim=1).values
        mean_pool = (
            sent_out.masked_fill(~sent_mask.unsqueeze(2), 0.0).sum(dim=1)
            / sent_lens.clamp_min(1).to(sent_out.dtype).unsqueeze(1)
        )
        feat = torch.cat([sent_last, attn_pool, mean_pool], dim=1)
        return self.head(feat).squeeze(1)


def evaluate(model, batcher, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in batcher:
            x, wlens, slens, labels = [t.to(device, non_blocking=True) for t in batch]
            logits = model(x, wlens, slens)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * labels.numel()
            total_correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / total, total_correct / total


def collect_probs(model, dataset: HanDataset, batch_size: int, device):
    model.eval()
    probs: List[Optional[float]] = [None] * len(dataset)
    indices = list(range(len(dataset)))
    indices.sort(key=lambda i: int(dataset.sent_lens[i]), reverse=True)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            idx = torch.tensor(batch_idx, dtype=torch.long)
            x = dataset.x.index_select(0, idx).to(device, non_blocking=True)
            wlens = dataset.word_lens.index_select(0, idx).to(device, non_blocking=True)
            slens = dataset.sent_lens.index_select(0, idx).to(device, non_blocking=True)
            logits = model(x, wlens, slens)
            batch_probs = torch.sigmoid(logits).cpu().tolist()
            for original_idx, prob in zip(batch_idx, batch_probs):
                probs[original_idx] = prob
    return [float(p) for p in probs if p is not None]


def calibrate(model, valid_data, batch_size, device):
    probs = collect_probs(model, valid_data, batch_size, device)
    labels = valid_data.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--unlabel", default="train_unlabel.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--out-dir", default="runs_han_scratch")
    p.add_argument("--submission", default="submission_han.csv")
    p.add_argument("--max-sents", type=int, default=24)
    p.add_argument("--max-words", type=int, default=48)
    p.add_argument("--max-vocab", type=int, default=90000)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--word-hidden", type=int, default=128)
    p.add_argument("--sent-hidden", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0012)
    p.add_argument("--weight-decay", type=float, default=0.0003)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--warmup-ratio", type=float, default=0.08)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2041)
    p.add_argument("--split-seed", type=int, default=2029)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--skip-predict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel)
    test_ids, test_texts = read_test(args.test)

    tokenized = [tokenize(t) for t in (train_texts + unlabel_texts + test_texts)]
    word2idx, idx2word = build_vocab(tokenized, args.min_count, args.max_vocab)
    print(f"device={device} vocab={len(idx2word)} train={len(train_texts)}")

    full = make_han_dataset(train_texts, train_labels, word2idx, args.max_sents, args.max_words)
    train_idx, valid_idx = split_indices(len(full), args.valid_ratio, args.split_seed)
    train_data = subset_han(full, train_idx)
    valid_data = subset_han(full, valid_idx)

    model = HanClassifier(
        vocab_size=len(idx2word),
        emb_dim=args.emb_dim,
        word_hidden=args.word_hidden,
        sent_hidden=args.sent_hidden,
        dropout=args.dropout,
    ).to(device)

    train_batcher = HanBatcher(train_data, args.batch_size, True, args.seed)
    valid_batcher = HanBatcher(valid_data, args.batch_size, False, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_batcher) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "han_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, wlens, slens, labels = [t.to(device, non_blocking=True) for t in batch]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x, wlens, slens)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total += labels.numel()
                total_loss += loss.item() * labels.numel()
        val_loss, val_acc = evaluate(model, valid_batcher, device)
        cal_acc, cal_th = calibrate(model, valid_data, args.batch_size, device)
        print(
            f"han epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_acc={val_acc:.4f} "
            f"cal_acc={cal_acc:.4f} cal_th={cal_th:.3f}"
        )
        if cal_acc > best_acc:
            best_acc = cal_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "word2idx": word2idx,
                    "idx2word": idx2word,
                    "calibrated_val_acc": best_acc,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"saved {best_path} cal_acc={best_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if not args.skip_predict:
        test_data = make_han_dataset(test_texts, None, word2idx, args.max_sents, args.max_words)
        probs = collect_probs(model, test_data, args.batch_size, device)
        # Default threshold 0.5
        with open(args.submission, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for sample_id, p in zip(test_ids, probs):
                w.writerow([sample_id, int(p >= 0.5)])
        # Calibrated threshold from validation
        cal_acc, cal_th = calibrate(model, valid_data, args.batch_size, device)
        with open(args.submission.replace(".csv", "_calibrated.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for sample_id, p in zip(test_ids, probs):
                w.writerow([sample_id, int(p >= cal_th)])
        # Balanced
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        labels = [0] * len(probs)
        for i in order[: len(probs) // 2]:
            labels[i] = 1
        with open(args.submission.replace(".csv", "_balanced.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for sample_id, label in zip(test_ids, labels):
                w.writerow([sample_id, label])
        print(f"wrote {args.submission} (+ calibrated/balanced)")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "calibrated_val_acc": best_acc,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
