import argparse
import csv
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    EncodedDataset,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    set_seed,
    split_indices,
)
from train_ulmfit_scratch import ULMFiTClassifier


def bernoulli_kl_with_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)
    logp = torch.log(p)
    log1p = torch.log1p(-p)
    logq = F.logsigmoid(q_logits)
    log1q = F.logsigmoid(-q_logits)
    return p * (logp - logq) + (1 - p) * (log1p - log1q)


def reverse_padded_tokens(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # Keep padding positions in place and reverse only the valid prefix.
    bsz, seq_len = x.shape
    pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
    rev = (lengths.unsqueeze(1) - 1 - pos).clamp_min(0)
    gather_idx = torch.where(pos < lengths.unsqueeze(1), rev, pos)
    return x.gather(1, gather_idx)


def reverse_padded_repr(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, dim = h.shape
    pos = torch.arange(seq_len, device=h.device).unsqueeze(0).expand(bsz, -1)
    rev = (lengths.unsqueeze(1) - 1 - pos).clamp_min(0)
    gather_idx = torch.where(pos < lengths.unsqueeze(1), rev, pos)
    gather_idx = gather_idx.unsqueeze(2).expand(-1, -1, dim)
    return h.gather(1, gather_idx)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        one_minus = 1.0 - self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=one_minus)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        for name, p in model.named_parameters():
            if name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}


@contextmanager
def use_ema(ema: Optional[EMA], model: nn.Module):
    if ema is None:
        yield
        return
    ema.apply_shadow(model)
    try:
        yield
    finally:
        ema.restore(model)


class BiContextULMFiT(nn.Module):
    def __init__(
        self,
        base: ULMFiTClassifier,
        hidden_dim: int,
        dropout: float,
        msd_samples: int,
        pad_idx: int = 0,
        unk_idx: int = 1,
        word_dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout

        # Reuse pretrained modules.
        self.embedding = base.embedding
        self.emb_dropout = base.emb_dropout
        self.encoder = base.encoder

        self.attn_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.attn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        side_dim = hidden_dim * 4
        fuse_in = side_dim * 2
        self.gate = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, side_dim),
            nn.Sigmoid(),
        )

        feature_dim = side_dim * 4
        mid_dim = hidden_dim * 4
        self.pre_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, mid_dim),
            nn.GELU(),
        )
        self.sample_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(max(1, msd_samples))]
        )
        self.out = nn.Linear(mid_dim, 1)

    def _word_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.word_dropout <= 0.0:
            return x
        mask = (torch.rand_like(x, dtype=torch.float32) < self.word_dropout) & (x != self.pad_idx)
        return x.masked_fill(mask, self.unk_idx)

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        out, _, _ = self.encoder(emb, lengths, None)
        return out

    def _pool(self, out: torch.Tensor, lengths: torch.Tensor, attn_layer: nn.Module) -> torch.Tensor:
        seq_len = out.size(1)
        mask = torch.arange(seq_len, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        last_idx = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, last_idx).squeeze(1)
        max_pool = out.masked_fill(~mask.unsqueeze(2), -1e4).max(dim=1).values
        mean_pool = (
            out.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1)
            / lengths.clamp_min(1).to(out.dtype).unsqueeze(1)
        )
        attn_score = attn_layer(out).squeeze(2).masked_fill(~mask, -1e4)
        attn = F.softmax(attn_score, dim=1).unsqueeze(2)
        attn_pool = (out * attn).sum(dim=1)
        return torch.cat([last, max_pool, mean_pool, attn_pool], dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self._word_dropout(x)
        out_f = self._encode(x, lengths)

        x_rev = reverse_padded_tokens(x, lengths)
        out_b_rev = self._encode(x_rev, lengths)
        out_b = reverse_padded_repr(out_b_rev, lengths)

        feat_f = self._pool(out_f, lengths, self.attn_f)
        feat_b = self._pool(out_b, lengths, self.attn_b)
        gate = self.gate(torch.cat([feat_f, feat_b], dim=1))
        mix = gate * feat_f + (1.0 - gate) * feat_b
        diff = torch.abs(feat_f - feat_b)
        feat = torch.cat([feat_f, feat_b, mix, diff], dim=1)
        hidden = self.pre_head(feat)
        logits = [self.out(d(hidden)).squeeze(1) for d in self.sample_dropouts]
        return torch.stack(logits, dim=0).mean(dim=0)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is None:
        return loss.mean()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def calibrate(model: nn.Module, valid_data: EncodedDataset, batch_size: int, device) -> Tuple[float, float]:
    probs = predict_probs(model, valid_data, batch_size, device)
    labels = valid_data.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)
    return best


def evaluate(model: nn.Module, valid_data: EncodedDataset, batch_size: int, device) -> Tuple[float, float]:
    model.eval()
    batcher = BucketBatcher(valid_data, batch_size, False, 1234)
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in batcher:
            x, lengths, labels = [t.to(device, non_blocking=True) for t in batch]
            logits = model(x, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * labels.numel()
            total_correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / max(1, total), total_correct / max(1, total)


def parse_batch(batch):
    if len(batch) == 4:
        x, lengths, labels, weights = batch
    else:
        x, lengths, labels = batch
        weights = None
    return x, lengths, labels, weights


def fgm_attack(model: BiContextULMFiT, epsilon: float) -> Optional[torch.Tensor]:
    emb = model.embedding.weight
    if emb.grad is None:
        return None
    grad = emb.grad
    norm = torch.norm(grad)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        return None
    r_adv = epsilon * grad / norm
    emb.data.add_(r_adv)
    return r_adv


def train_stage(
    model: BiContextULMFiT,
    train_data: EncodedDataset,
    valid_data: EncodedDataset,
    args,
    device,
    out_dir: str,
    stage_name: str,
    epochs: int,
    lr: float,
    seed_offset: int = 0,
) -> str:
    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed + seed_offset)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.embedding.parameters(), "lr": lr * 0.2},
            {"params": model.encoder.parameters(), "lr": lr * 0.6},
            {"params": model.attn_f.parameters(), "lr": lr},
            {"params": model.attn_b.parameters(), "lr": lr},
            {"params": model.gate.parameters(), "lr": lr},
            {"params": model.pre_head.parameters(), "lr": lr},
            {"params": model.out.parameters(), "lr": lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_batcher) * epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    best = -1.0
    best_path = os.path.join(out_dir, f"{stage_name}_best.pt")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, lengths, labels, weights = parse_batch(batch)
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True) if weights is not None else None

            targets = labels * (1.0 - 2.0 * args.label_smoothing) + args.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            logits1 = model(x, lengths)
            logits2 = model(x, lengths)
            sup = 0.5 * (
                weighted_bce_with_logits(logits1, targets, weights)
                + weighted_bce_with_logits(logits2, targets, weights)
            )
            kl_vec = 0.5 * (
                bernoulli_kl_with_logits(logits1, logits2)
                + bernoulli_kl_with_logits(logits2, logits1)
            )
            if weights is not None:
                kl = (kl_vec * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                kl = kl_vec.mean()
            loss = sup + args.rdrop_alpha * kl
            loss.backward()

            if args.adv_eps > 0 and args.adv_weight > 0:
                r_adv = fgm_attack(model, args.adv_eps)
                if r_adv is not None:
                    adv_logits = model(x, lengths)
                    adv_loss = weighted_bce_with_logits(adv_logits, targets, weights)
                    (args.adv_weight * adv_loss).backward()
                    model.embedding.weight.data.sub_(r_adv)

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                pred = (torch.sigmoid(logits1) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total += labels.numel()
                total_loss += loss.item() * labels.numel()

            if global_step % 80 == 0:
                print(
                    f"{stage_name} epoch {epoch:02d} step {global_step}/{total_steps} "
                    f"loss={total_loss/max(1,total):.5f}"
                )

        with use_ema(ema, model):
            val_loss, val_acc = evaluate(model, valid_data, args.batch_size, device)
            cal_acc, cal_th = calibrate(model, valid_data, args.batch_size, device)
        print(
            f"{stage_name} epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_acc={val_acc:.4f} "
            f"cal_acc={cal_acc:.4f} cal_th={cal_th:.3f}"
        )
        if cal_acc > best:
            best = cal_acc
            with use_ema(ema, model):
                torch.save(
                    {
                        "model": model.state_dict(),
                        "calibrated_val_acc": cal_acc,
                        "val_acc": val_acc,
                        "cal_th": cal_th,
                        "stage": stage_name,
                    },
                    best_path,
                )
            print(f"saved {best_path} cal_acc={cal_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return best_path


def select_pseudo(
    model: nn.Module,
    unlabel_texts: Sequence[str],
    word2idx: dict,
    seq_len: int,
    batch_size: int,
    device,
    threshold: float,
    max_per_class: int,
    pseudo_weight: float,
    pseudo_power: float,
) -> Tuple[List[str], List[float], List[float]]:
    unlabel_data = make_dataset(unlabel_texts, None, word2idx, seq_len)
    probs = predict_probs(model, unlabel_data, batch_size, device)
    pos = [(i, p) for i, p in enumerate(probs) if p >= threshold]
    neg = [(i, p) for i, p in enumerate(probs) if p <= (1.0 - threshold)]
    pos.sort(key=lambda x: x[1], reverse=True)
    neg.sort(key=lambda x: x[1])
    if max_per_class > 0:
        pos = pos[:max_per_class]
        neg = neg[:max_per_class]
    picked = pos + neg
    texts = [unlabel_texts[i] for i, _ in picked]
    labels = [1.0 if p >= 0.5 else 0.0 for _, p in picked]
    weights = []
    for _, p in picked:
        conf = abs(p - 0.5) * 2.0
        w = pseudo_weight * (max(1e-4, conf) ** pseudo_power)
        weights.append(float(w))
    return texts, labels, weights


def write_submissions(
    model: nn.Module,
    test_ids: Sequence[str],
    test_texts: Sequence[str],
    word2idx: dict,
    seq_len: int,
    batch_size: int,
    device,
    valid_data: EncodedDataset,
    submission_path: str,
) -> None:
    test_data = make_dataset(test_texts, None, word2idx, seq_len)
    probs = predict_probs(model, test_data, batch_size, device)
    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= 0.5)])

    cal_acc, cal_th = calibrate(model, valid_data, batch_size, device)
    cal_path = submission_path.replace(".csv", "_calibrated.csv")
    with open(cal_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= cal_th)])

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    labels = [0] * len(probs)
    for i in order[: len(probs) // 2]:
        labels[i] = 1
    bal_path = submission_path.replace(".csv", "_balanced.csv")
    with open(bal_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, label in zip(test_ids, labels):
            w.writerow([sample_id, label])

    print(f"calibrated_val_acc={cal_acc:.4f} threshold={cal_th:.3f}")
    print(f"wrote {submission_path}")
    print(f"wrote {cal_path}")
    print(f"wrote {bal_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    p.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    p.add_argument("--out-dir", default="runs_ulmfit_bicontext_adv")
    p.add_argument("--submission", default="submission_ulmfit_bicontext_adv.csv")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--unlabel", default="train_unlabel.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--pseudo-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.8e-4)
    p.add_argument("--pseudo-lr", type=float, default=1.8e-4)
    p.add_argument("--weight-decay", type=float, default=2.5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.08)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--rdrop-alpha", type=float, default=3.0)
    p.add_argument("--adv-eps", type=float, default=0.35)
    p.add_argument("--adv-weight", type=float, default=0.35)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=0.25)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=2029)
    p.add_argument("--seed", type=int, default=2049)
    p.add_argument("--msd-samples", type=int, default=4)
    p.add_argument("--pseudo-rounds", type=int, default=1)
    p.add_argument("--pseudo-threshold", type=float, default=0.985)
    p.add_argument("--pseudo-max-per-class", type=int, default=12000)
    p.add_argument("--pseudo-weight", type=float, default=0.25)
    p.add_argument("--pseudo-power", type=float, default=1.5)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-unlabel", type=int, default=None)
    p.add_argument("--skip-predict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_final, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["args"])
    word2idx = base["word2idx"]
    idx2word = base["idx2word"]

    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel, args.max_unlabel)
    test_ids, test_texts = read_test(args.test)

    train_idx, valid_idx = split_indices(len(train_texts), args.valid_ratio, args.split_seed)
    train_split_texts = [train_texts[i] for i in train_idx]
    train_split_labels = [float(train_labels[i]) for i in train_idx]
    valid_texts = [train_texts[i] for i in valid_idx]
    valid_labels = [float(train_labels[i]) for i in valid_idx]

    train_data = make_dataset(train_split_texts, train_split_labels, word2idx, cfg["seq_len"])
    valid_data = make_dataset(valid_texts, valid_labels, word2idx, cfg["seq_len"])

    base_model = ULMFiTClassifier(
        len(idx2word),
        cfg["emb_dim"],
        cfg["hidden_dim"],
        cfg["layers"],
        cfg["dropout"],
        word_dropout=cfg.get("word_dropout", 0.04),
    )
    base_model.load_state_dict(ckpt["model"])
    model = BiContextULMFiT(
        base_model,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        msd_samples=args.msd_samples,
        word_dropout=cfg.get("word_dropout", 0.04),
    ).to(device)

    best_path = train_stage(
        model,
        train_data,
        valid_data,
        args,
        device,
        args.out_dir,
        "stage1",
        epochs=args.epochs,
        lr=args.lr,
        seed_offset=0,
    )

    for round_id in range(args.pseudo_rounds):
        print(f"pseudo round {round_id + 1}/{args.pseudo_rounds} selecting ...")
        pseudo_texts, pseudo_labels, pseudo_weights = select_pseudo(
            model,
            unlabel_texts,
            word2idx,
            cfg["seq_len"],
            args.batch_size,
            device,
            args.pseudo_threshold,
            args.pseudo_max_per_class,
            args.pseudo_weight,
            args.pseudo_power,
        )
        print(f"pseudo selected={len(pseudo_texts)} threshold={args.pseudo_threshold}")
        if not pseudo_texts:
            break

        combined_texts = train_split_texts + pseudo_texts
        combined_labels = train_split_labels + pseudo_labels
        combined_weights = [1.0] * len(train_split_texts) + pseudo_weights
        pseudo_data = make_dataset(
            combined_texts,
            combined_labels,
            word2idx,
            cfg["seq_len"],
            weights=combined_weights,
        )
        best_path = train_stage(
            model,
            pseudo_data,
            valid_data,
            args,
            device,
            args.out_dir,
            f"pseudo{round_id + 1}",
            epochs=args.pseudo_epochs,
            lr=args.pseudo_lr,
            seed_offset=17 + round_id * 11,
        )

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])

    if not args.skip_predict:
        write_submissions(
            model,
            test_ids,
            test_texts,
            word2idx,
            cfg["seq_len"],
            args.batch_size,
            device,
            valid_data,
            args.submission,
        )

    torch.save(
        {
            "model": model.state_dict(),
            "args": cfg,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "best_stage_checkpoint": best_path,
            "best_val_acc": best.get("val_acc"),
            "best_calibrated_val_acc": best.get("calibrated_val_acc"),
            "model_class": "BiContextULMFiT",
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
