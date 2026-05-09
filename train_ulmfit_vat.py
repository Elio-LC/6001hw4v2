import argparse
import csv
import itertools
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
    read_unlabel,
    split_indices,
    subset_dataset,
)
from train_ulmfit_scratch import ULMFiTClassifier, evaluate_classifier


def forward_from_emb(model: ULMFiTClassifier, emb: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor):
    out, _, _ = model.encoder(emb, lengths, None)
    mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    last_idx = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.size(2))
    last = out.gather(1, last_idx).squeeze(1)
    max_pool = out.masked_fill(~mask.unsqueeze(2), -1e4).max(dim=1).values
    mean_pool = (
        out.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1)
        / lengths.clamp_min(1).to(out.dtype).unsqueeze(1)
    )
    attn_score = model.attn(out).squeeze(2).masked_fill(~mask, -1e4)
    attn = F.softmax(attn_score, dim=1).unsqueeze(2)
    attn_pool = (out * attn).sum(dim=1)
    return model.head(torch.cat([last, max_pool, mean_pool, attn_pool], dim=1)).squeeze(1)


def normalize_perturbation(d: torch.Tensor, x: torch.Tensor):
    mask = (x != 0).to(d.dtype).unsqueeze(2)
    d = d * mask
    flat = d.view(d.size(0), -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-8)
    return (flat / norm).view_as(d) * mask


def bernoulli_kl_with_logits(base_logits, perturbed_logits):
    p = torch.sigmoid(base_logits.detach()).clamp(1e-6, 1 - 1e-6)
    logp = torch.log(p)
    log1p = torch.log1p(-p)
    logq = F.logsigmoid(perturbed_logits)
    log1q = F.logsigmoid(-perturbed_logits)
    return (p * (logp - logq) + (1 - p) * (log1p - log1q)).mean()


def vat_loss(model, x, lengths, xi, eps):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        emb = model.embedding(x)
        base_logits = forward_from_emb(model, emb, x, lengths)

    d = torch.randn_like(emb)
    d = normalize_perturbation(d, x)
    d.requires_grad_()
    perturbed_logits = forward_from_emb(model, emb + xi * d, x, lengths)
    adv_distance = bernoulli_kl_with_logits(base_logits, perturbed_logits)
    grad = torch.autograd.grad(adv_distance, d, only_inputs=True)[0]
    r_adv = eps * normalize_perturbation(grad.detach(), x)
    perturbed_logits = forward_from_emb(model, emb + r_adv, x, lengths)
    loss = bernoulli_kl_with_logits(base_logits, perturbed_logits)
    if was_training:
        model.train()
    return loss


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


def write_variants(model, test_ids, test_texts, word2idx, cfg, args, device):
    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, test_data, args.batch_size, device)
    with open(args.submission, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, prob in zip(test_ids, probs):
            writer.writerow([sample_id, int(prob >= 0.5)])

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    labels = [0] * len(probs)
    for i in order[: len(probs) // 2]:
        labels[i] = 1
    balanced_path = args.submission.replace(".csv", "_balanced.csv")
    with open(balanced_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, label in zip(test_ids, labels):
            writer.writerow([sample_id, label])
    print(f"wrote {args.submission} and {balanced_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    parser.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    parser.add_argument("--out-dir", default="runs_ulmfit_vat")
    parser.add_argument("--submission", default="submission_ulmfit_vat.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--unlabel", default="train_unlabel.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unlabel-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vat-weight", type=float, default=0.8)
    parser.add_argument("--vat-xi", type=float, default=1e-3)
    parser.add_argument("--vat-eps", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-unlabel", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_final, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["args"])
    word2idx = base["word2idx"]
    idx2word = base["idx2word"]

    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel, args.max_unlabel)
    test_ids, test_texts = read_test(args.test)
    full_train = make_dataset(train_texts, train_labels, word2idx, cfg["seq_len"])
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    train_idx, valid_idx = split_indices(len(full_train), args.valid_ratio, split_seed)
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)
    unlabel_data = make_dataset(unlabel_texts, None, word2idx, cfg["seq_len"])

    model = ULMFiTClassifier(
        len(idx2word),
        cfg["emb_dim"],
        cfg["hidden_dim"],
        cfg["layers"],
        cfg["dropout"],
        word_dropout=cfg.get("word_dropout", 0.04),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed)
    unlabel_batcher = BucketBatcher(unlabel_data, args.unlabel_batch_size, True, args.seed + 77)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.embedding.parameters(), "lr": args.lr * 0.25},
            {"params": model.encoder.parameters(), "lr": args.lr * 0.5},
            {"params": model.attn.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_batcher) * args.epochs
    warmup = max(1, int(total_steps * 0.08))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "ulmfit_vat_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        unlabeled_iter = itertools.cycle(iter(unlabel_batcher))
        for step, batch in enumerate(train_batcher, 1):
            x, lengths, labels = [item.to(device, non_blocking=True) for item in batch]
            ux, ulengths = [item.to(device, non_blocking=True) for item in next(unlabeled_iter)]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x, lengths)
                sup_loss = F.binary_cross_entropy_with_logits(logits, targets)
            adv_loss = vat_loss(model, ux, ulengths, args.vat_xi, args.vat_eps)
            loss = sup_loss + args.vat_weight * adv_loss
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
            if step % 50 == 0:
                print(
                    f"vat epoch {epoch:02d} step {step}/{len(train_batcher)} "
                    f"loss={total_loss/max(1,total):.5f}"
                )
        valid_batcher = BucketBatcher(valid_data, args.batch_size, False, args.seed)
        val_loss, val_acc = evaluate_classifier(model, valid_batcher, device)
        cal_acc, cal_th = calibrate(model, valid_data, args.batch_size, device)
        print(
            f"vat epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_acc={val_acc:.4f} "
            f"cal_acc={cal_acc:.4f} cal_th={cal_th:.3f}"
        )
        if cal_acc > best_acc:
            best_acc = cal_acc
            torch.save(
                {"model": model.state_dict(), "args": cfg, "calibrated_val_acc": best_acc, "val_acc": val_acc},
                best_path,
            )
            print(f"saved {best_path} cal_acc={best_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    write_variants(model, test_ids, test_texts, word2idx, cfg, args, device)
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": cfg,
            "calibrated_val_acc": best_acc,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
