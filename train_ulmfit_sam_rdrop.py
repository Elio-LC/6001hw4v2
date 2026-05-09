import argparse
import csv
import math
import os
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    split_indices,
    subset_dataset,
)
from train_ulmfit_scratch import ULMFiTClassifier, evaluate_classifier


def bernoulli_kl_with_logits(p_logits: torch.Tensor, q_logits: torch.Tensor):
    p = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)
    logp = torch.log(p)
    log1p = torch.log1p(-p)
    logq = F.logsigmoid(q_logits)
    log1q = F.logsigmoid(-q_logits)
    return p * (logp - logq) + (1 - p) * (log1p - log1q)


class SAM:
    def __init__(self, params: Iterable[torch.nn.Parameter], base_optimizer, rho=0.05, adaptive=False, **kwargs):
        self.params = list(params)
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(self.params, **kwargs)
        self.state = {}

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
            p.add_(e_w)
            self.state[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p in self.state:
                p.sub_(self.state[p])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        norms = []
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.adaptive:
                g = g * p.abs()
            norms.append(torch.norm(g, p=2))
        if not norms:
            return torch.tensor(0.0, device=self.params[0].device)
        return torch.norm(torch.stack(norms), p=2)


def calibrate(model, valid_data, batch_size, device):
    probs = predict_probs(model, valid_data, batch_size, device)
    labels = valid_data.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)
    return best


def write_outputs(model, test_ids, test_texts, word2idx, cfg, args, device):
    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, test_data, cfg["batch_size"], device)
    # default
    with open(args.submission, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= 0.5)])
    # calibrated
    cal_acc, cal_th = calibrate(model, args.valid_data_for_cal, cfg["batch_size"], device)
    with open(args.submission.replace(".csv", "_calibrated.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= cal_th)])
    # balanced
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    labels = [0] * len(probs)
    for i in order[: len(probs) // 2]:
        labels[i] = 1
    with open(args.submission.replace(".csv", "_balanced.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, label in zip(test_ids, labels):
            w.writerow([sample_id, label])
    print(f"calibrated_val_acc={cal_acc:.4f} threshold={cal_th:.3f}")
    print(f"wrote {args.submission} (+ calibrated/balanced)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    p.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    p.add_argument("--out-dir", default="runs_ulmfit_sam_rdrop")
    p.add_argument("--submission", default="submission_ulmfit_sam_rdrop.csv")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--rho", type=float, default=0.06)
    p.add_argument("--rdrop-alpha", type=float, default=4.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=0.25)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=2029)
    p.add_argument("--seed", type=int, default=2043)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--skip-predict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_final, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    word2idx = base["word2idx"]
    idx2word = base["idx2word"]

    train_texts, train_labels = read_train(args.train, args.max_train)
    train_idx, valid_idx = split_indices(len(train_texts), args.valid_ratio, args.split_seed)
    train_texts_split = [train_texts[i] for i in train_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    valid_texts = [train_texts[i] for i in valid_idx]
    valid_labels = [train_labels[i] for i in valid_idx]

    train_data = make_dataset(train_texts_split, train_labels_split, word2idx, cfg["seq_len"])
    valid_data = make_dataset(valid_texts, valid_labels, word2idx, cfg["seq_len"])
    args.valid_data_for_cal = valid_data
    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed)
    valid_batcher = BucketBatcher(valid_data, args.batch_size, False, args.seed)

    model = ULMFiTClassifier(
        len(idx2word),
        cfg["emb_dim"],
        cfg["hidden_dim"],
        cfg["layers"],
        cfg["dropout"],
        word_dropout=cfg.get("word_dropout", 0.04),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    sam = SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, rho=args.rho)
    total_steps = len(train_batcher) * args.epochs
    warmup = max(1, int(total_steps * 0.08))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(sam.base_optimizer, lr_lambda)
    # SAM already performs two backward passes per step; keep it in full precision
    # for training stability and to avoid AMP scaler state conflicts.
    use_amp = False
    best_cal = -1.0
    best_path = os.path.join(args.out_dir, "sam_rdrop_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, lengths, labels = [t.to(device, non_blocking=True) for t in batch]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            # first forward-backward
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits1 = model(x, lengths)
                logits2 = model(x, lengths)
                sup = 0.5 * (
                    F.binary_cross_entropy_with_logits(logits1, targets)
                    + F.binary_cross_entropy_with_logits(logits2, targets)
                )
                kl1 = bernoulli_kl_with_logits(logits1, logits2)
                kl2 = bernoulli_kl_with_logits(logits2, logits1)
                rdrop = 0.5 * (kl1 + kl2).mean()
                loss = sup + args.rdrop_alpha * rdrop
            sam.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            sam.first_step(zero_grad=True)

            # second forward-backward on perturbed weights
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits1b = model(x, lengths)
                logits2b = model(x, lengths)
                sup_b = 0.5 * (
                    F.binary_cross_entropy_with_logits(logits1b, targets)
                    + F.binary_cross_entropy_with_logits(logits2b, targets)
                )
                kl1b = bernoulli_kl_with_logits(logits1b, logits2b)
                kl2b = bernoulli_kl_with_logits(logits2b, logits1b)
                rdrop_b = 0.5 * (kl1b + kl2b).mean()
                loss_b = sup_b + args.rdrop_alpha * rdrop_b
            loss_b.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            sam.second_step(zero_grad=True)
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits1) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total += labels.numel()
                total_loss += loss.item() * labels.numel()

        val_loss, val_acc = evaluate_classifier(model, valid_batcher, device)
        cal_acc, cal_th = calibrate(model, valid_data, cfg["batch_size"], device)
        print(
            f"sam-rdrop epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_acc={val_acc:.4f} "
            f"cal_acc={cal_acc:.4f} cal_th={cal_th:.3f}"
        )
        if cal_acc > best_cal:
            best_cal = cal_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": cfg,
                    "word2idx": word2idx,
                    "idx2word": idx2word,
                    "calibrated_val_acc": cal_acc,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"saved {best_path} cal_acc={cal_acc:.4f}")

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])

    if not args.skip_predict:
        test_ids, test_texts = read_test(args.test)
        write_outputs(model, test_ids, test_texts, word2idx, cfg, args, device)
    torch.save(best, os.path.join(args.out_dir, "final_model.pt"))
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
