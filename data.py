import torch
from datasets import load_dataset
import os
import tempfile
import sentencepiece as spm

PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
SPLIT = {"train": "train", "validation": "validation", "test": "test"}

def load_raw_pairs(split: str):
    ds = load_dataset("bentrevett/multi30k", split=SPLIT[split])
    return [(ex["de"], ex["en"]) for ex in ds]


def train_tokenizer(train_pairs, vocab_size = 8000, model_prefix = "spm"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for de, en in train_pairs:
            f.write(f"{de.strip()}\n{en.strip()}\n")
        corpus_path = f.name

    spm = spm.SentencePieceTrainer.train(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type = "bpe",
        character_coverage=1.0,
        bos_id=SOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
    )
    os.remove(corpus_path)
    return load_tokenizer(f"{model_prefix}.model")

def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def encode_pair(sp, src, tgt, max_len: int=100):
    src = sp.encode(src) + [EOS_ID]
    tgt = [SOS_ID] + sp.encode(tgt) + [EOS_ID]
    if len(src) > max_len or len(tgt) > max_len:
        return None
    return src, tgt

