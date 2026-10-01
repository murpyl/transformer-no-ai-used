import argparse
import sacrebleu
import torch

from data import EOS_ID, SOS_ID, load_raw_pairs, load_tokenizer
from model import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str):
    sd = torch.load(path, map_location=DEVICE)
    vocab_size, d_model = sd["decoder.embed.weight"].shape
    model = Transformer(
        vocab_size,
        d_model=d_model,
        n_heads=4,  
        d_ff=sd["encoder.layers.0.feed_forward.net.0.weight"].shape[0],
        n_layers=sum(k.endswith("self_attn.q_proj.weight") for k in sd if k.startswith("encoder.layers.")),
        max_len=sd["encoder.pos_encoder.pe"].shape[1],
        pad_id=0, #i f*d up with the model by not saving it in a normal json format looool
    ).to(DEVICE)
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def translate(model, sp, src_text: str, max_len: int = 100) -> str:
    src = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long, device=DEVICE)
    src_mask = model.make_src_mask(src)
    memory = model.encoder(src, src_mask)  

    generated = [SOS_ID]
    for _ in range(max_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        dec_out = model.decode(tgt, memory, src_mask)
        next_id = model.out_proj(dec_out[:, -1]).argmax(dim=-1).item()  
        if next_id == EOS_ID:
            break
        generated.append(next_id)

    return sp.decode(generated[1:]) 


def evaluate(model, sp, split: str, n: int | None = None, show: int = 5):
    pairs = load_raw_pairs(split)[:n]
    hyps = [translate(model, sp, de) for de, _ in pairs]
    refs = [en for _, en in pairs]

    for (de, en), hyp in list(zip(pairs, hyps))[:show]:
        print(f"SRC: {de}\nREF: {en}\nHYP: {hyp}\n")

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"{split} BLEU ({len(pairs)} sentences): {bleu.score:.2f}")
    return bleu.score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--spm", default="spm.model")
    ap.add_argument("--sentence")
    ap.add_argument("--eval", choices=["train", "validation", "test"])
    ap.add_argument("-n", type=int, default=None)
    args = ap.parse_args()

    sp = load_tokenizer(args.spm)
    model = load_model(args.checkpoint)

    if args.sentence:
        print(translate(model, sp, args.sentence))
    if args.eval:
        evaluate(model, sp, args.eval, args.n)


if __name__ == "__main__":
    main()