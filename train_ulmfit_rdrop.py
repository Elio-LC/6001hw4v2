import argparse
import csv
import math
import os

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
)
from train_ulmfit_scratch import ULMFiTClassifier, evaluate_classifier


def bernoulli_kl_with_logits(p_logits: torch.Tensor, q_logits: torch.Tensor):
    p = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)
    logp = torch.log(p)
    log1p = torch.log1p(-p)
    logq = F.logsigmoid(q_logits)
    log1q = F.logsigmoid(-q_logits)
    return p * (logp - logq) + (1 - p) * (log1p - log1q)


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    p.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    p.add_argument("--out-dir", default="runs_ulmfit_rdrop")
    p.add_argument("--submission", default="submission_ulmfit_rdrop.csv")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--rdrop-alpha", type=float, default=4.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=0.25)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=2029)
    p.add_argument("--seed", type=int, default=2045)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-train", type=int, default=None)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_batcher) * args.epochs
    warmup = max(1, int(total_steps * 0.08))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_cal = -1.0
    best_path = os.path.join(args.out_dir, "rdrop_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for step, batch in enumerate(train_batcher, 1):
            x, lengths, labels = [t.to(device, non_blocking=True) for t in batch]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits1 = model(x, lengths)
                logits2 = model(x, lengths)
                sup = 0.5 * (
                    F.binary_cross_entropy_with_logits(logits1, targets)
                    + F.binary_cross_entropy_with_logits(logits2, targets)
                )
                kl = 0.5 * (
                    bernoulli_kl_with_logits(logits1, logits2)
                    + bernoulli_kl_with_logits(logits2, logits1)
                ).mean()
                loss = sup + args.rdrop_alpha * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits1) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total += labels.numel()
                total_loss += loss.item() * labels.numel()
            if step % 50 == 0:
                print(
                    f"rdrop epoch {epoch:02d} step {step}/{len(train_batcher)} "
                    f"loss={total_loss/max(1,total):.5f}"
                )

        val_loss, val_acc = evaluate_classifier(model, valid_batcher, device)
        cal_acc, cal_th = calibrate(model, valid_data, cfg["batch_size"], device)
        print(
            f"rdrop epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
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
    test_ids, test_texts = read_test(args.test)
    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, test_data, cfg["batch_size"], device)

    with open(args.submission, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= 0.5)])

    cal_acc, cal_th = calibrate(model, valid_data, cfg["batch_size"], device)
    with open(args.submission.replace(".csv", "_calibrated.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, p in zip(test_ids, probs):
            w.writerow([sample_id, int(p >= cal_th)])

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    labels = [0] * len(probs)
    for i in order[: len(probs) // 2]:
        labels[i] = 1
    with open(args.submission.replace(".csv", "_balanced.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for sample_id, label in zip(test_ids, labels):
            w.writerow([sample_id, label])

    torch.save(best, os.path.join(args.out_dir, "final_model.pt"))
    print(f"calibrated_val_acc={cal_acc:.4f} threshold={cal_th:.3f}")
    print(f"wrote {args.submission} (+ calibrated/balanced)")


if __name__ == "__main__":
    main()
