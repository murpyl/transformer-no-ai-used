import torch
import torch.nn as nn
import math

from model import Transformer
from data import (PAD_ID, load_raw_pairs, train_tokenizer, load_tokenizer,
                  TranslationDataset, get_dataloader)

D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 3
DROPOUT = 0.1
VOCAB_SIZE = 8000
BATCH_SIZE = 32
WARMUP_STEPS = 1000
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
N_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def noam_lambda(step, d_model=D_MODEL, warmup_steps=WARMUP_STEPS):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

def run_epoch(model, dataloader, optimizer, scheduler, loss_fn, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            src, tgt_input, tgt_output = batch["src"].to(DEVICE), batch["tgt_input"].to(DEVICE), batch["tgt_output"].to(DEVICE)
            logits = model(src, tgt_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
            tokens = (tgt_output != PAD_ID).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens


def main():
    train_pairs = load_raw_pairs("train")
    val_pairs = load_raw_pairs("validation")
    test_pairs = load_raw_pairs("test")
    sp = train_tokenizer(train_pairs)
    #sp = load_tokenizer("spm.model")

    train_ds = TranslationDataset(train_pairs, sp)
    val_ds = TranslationDataset(val_pairs, sp)
    test_ds = TranslationDataset(test_pairs, sp)

    train_dl = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, dropout=DROPOUT, pad_id=PAD_ID,
    )
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = noam_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=PAD_ID)

    best_val_loss = math.inf

    for epoch in range(N_EPOCHS):
        train_loss = run_epoch(model, train_dl, optimizer, scheduler, loss_fn, train=True)
        val_loss = run_epoch(model, val_dl, optimizer, scheduler, loss_fn, train=False)
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()
