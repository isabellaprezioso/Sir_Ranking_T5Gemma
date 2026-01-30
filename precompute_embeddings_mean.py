# -*- coding: utf-8 -*-
"""Genera offline gli embeddings T5/GEMMA per utenti e item, allineati a user2id/item2id.
Applica mean pooling su tutta la sequenza, output (num_utenti, hidden_size)
"""
import os, json, argparse, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Es. ./dataset/Baby_5_data")
    p.add_argument("--model_path", type=str, default="~/.cache/huggingface/hub/models--google--t5gemma-2-270m-270m", help="Cartella/ID modello GEMMA")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=500)
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    p.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp32")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _join_texts(texts):
    if not texts:
        return "<unk>"
    merged = " ".join(texts).strip()
    return merged if merged else "<unk>"

def _build_ordered_corpus(mapping, texts_dict):
    max_id = max(int(v) for v in mapping.values())
    ordered = ["<unk>"] * (max_id + 1)
    for original_id, idx in mapping.items():
        idx = int(idx)
        ordered[idx] = _join_texts(texts_dict.get(original_id, ["<unk>"]))
    return ordered

@torch.no_grad()
def _encode_texts_mean(texts, tokenizer, encoder, device, max_len, batch_size):
    """Codifica i testi in embeddings con mean pooling lungo la sequenza."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attn_mask = enc["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=torch.bfloat16):
            out = encoder(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            pooled = summed / mask.sum(dim=1).clamp(min=1e-9)

        embs.append(pooled.float().cpu().numpy())
    return np.vstack(embs)

def _get_hidden_size(model, tokenizer, device):
    # Prova a ottenere hidden_size da config del modello GEMMA
    for attr in ["d_model", "hidden_size"]:
        hs = getattr(model.config, attr, None)
        if isinstance(hs, int) and hs > 0:
            return hs
    # fallback: prova con encoder output
    toks = tokenizer(["hello world"], return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = model.get_encoder()(**toks, return_dict=True)
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return int(last.size(-1))

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = args.data_path

    # Controllo file richiesti
    for r in ["user2id.json","item2id.json","user_texts.json","item_texts.json"]:
        if not os.path.exists(os.path.join(dp, r)):
            raise FileNotFoundError(f"Manca {r}")

    user2id = _load_json(os.path.join(dp, "user2id.json"))
    item2id = _load_json(os.path.join(dp, "item2id.json"))
    user_texts = _load_json(os.path.join(dp, "user_texts.json"))
    item_texts = _load_json(os.path.join(dp, "item_texts.json"))

    print(f" data_path = {dp}")
    print(f" model_path = {args.model_path}")
    print(f" device = {device}")
    print(f" batch={args.batch_size} max_len={args.max_len} dtype={args.save_dtype}")

    print(f"Carico T5Gemma da cache (FAST tokenizer)...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.model_path), use_fast=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.expanduser(args.model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    encoder = model.get_encoder()

    emb_dim = _get_hidden_size(model, tokenizer, device)
    print(f" Modello caricato | hidden_size={emb_dim}")

    print(" Preparo corpora ordinati...")
    ordered_user = _build_ordered_corpus(user2id, user_texts)
    ordered_item = _build_ordered_corpus(item2id, item_texts)
    num_users, num_items = len(ordered_user), len(ordered_item)
    print(f" users={num_users} | items={num_items}")

    print(" Encoding USERS ...")
    user_embs = _encode_texts_mean(ordered_user, tokenizer, encoder, device, args.max_len, args.batch_size)
    print(" Encoding ITEMS ...")
    item_embs = _encode_texts_mean(ordered_item, tokenizer, encoder, device, args.max_len, args.batch_size)

    if args.normalize:
        print("L2 normalize")
        user_embs /= (np.linalg.norm(user_embs, axis=1, keepdims=True) + 1e-9)
        item_embs /= (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-9)

    if args.save_dtype == "fp16":
        user_embs = user_embs.astype(np.float16)
        item_embs = item_embs.astype(np.float16)
    else:
        user_embs = user_embs.astype(np.float32)
        item_embs = item_embs.astype(np.float32)

    ue = os.path.join(dp, "user_embeddings.npy")
    ie = os.path.join(dp, "item_embeddings.npy")
    me = os.path.join(dp, "embeddings_meta.json")
    np.save(ue, user_embs)
    np.save(ie, item_embs)
    with open(me, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "dtype": args.save_dtype,
            "normalize": args.normalize,
            "num_users": int(num_users),
            "num_items": int(num_items),
            "emb_dim": int(user_embs.shape[1]),
            "pooling": "sequence"
        }, f, indent=2)

    print("\n Salvati:")
    print(f"    {ue} shape={user_embs.shape}")
    print(f"    {ie} shape={item_embs.shape}")
    print(f"    {me}")
    print(" Done.")

if __name__ == "__main__":
    main()

