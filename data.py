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

    spm.SentencePieceTrainer.train(
        input=corpus_path,
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

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, sp, max_len: int=100):
        self.examples = []
        for src, tgt in pairs:
            encoded = encode_pair(sp, src, tgt, max_len)
            if encoded is not None:
                self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)

    max_src_len = max([len(seq) for seq in src_seqs])
    max_tgt_len = max([len(seq) for seq in tgt_seqs])

    src = torch.full((len(batch), max_src_len), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt_len), PAD_ID, dtype=torch.long)

    for i, (src_seq, tgt_seq) in enumerate(zip(src_seqs, tgt_seqs)):
        src[i, :len(src_seq)] = torch.tensor(src_seq, dtype=torch.long)
        tgt[i, :len(tgt_seq)] = torch.tensor(tgt_seq, dtype=torch.long)

    return {
        "src": src,
        "tgt_input": tgt[:, :-1],
        "tgt_output": tgt[:, 1:],
    }

def get_dataloader(dataset, batch_size: int=32, shuffle: bool = True):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=shuffle,
    )