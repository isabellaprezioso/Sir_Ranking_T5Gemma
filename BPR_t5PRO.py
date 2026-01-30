# -*- encoding: utf-8 -*-
import os
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

now = datetime.now

from dataset.data_review_t5PRO import ReviewData
from framework import Model
from models.Losses import BPRLoss
import models
import config
from main import unpack_input          # classico
from main_t5PRO import unpack_input_t5, load_t5_model_and_tokenizer


# =====================================================
# COLLATE UNIFICATA (Embeddings / T5 / Classico)
# =====================================================
def collate_fn_bpr(batch):
    """
    Riconosce automaticamente la struttura del sample:
      - Embeddings: (user, pos, neg, {'user_vec','pos_vec','neg_vec'})
      - T5:        (user, pos, neg, {'user_input_ids',...})
      - Classico:  (user, pos, neg)
    Restituisce una tupla che il training loop interpreta a seconda della modalità.
    """
    first = batch[0]

    # Caso: sample con dict di encodings
    if isinstance(first, tuple) and len(first) == 4 and isinstance(first[3], dict):
        enc = first[3]

        # ===== PRECOMPUTED (DeepCoNN / NARRE) =====
        if 'user_vec' in enc and 'pos_vec' in enc and 'neg_vec' in enc:
            user_vecs = torch.stack([b[3]['user_vec'] for b in batch])
            pos_vecs  = torch.stack([b[3]['pos_vec']  for b in batch])
            neg_vecs  = torch.stack([b[3]['neg_vec']  for b in batch])

            users = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
            pos_it = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
            neg_it = torch.as_tensor([b[2] for b in batch], dtype=torch.long)

            return ('emb', users, pos_it, neg_it, user_vecs, pos_vecs, neg_vecs)

        # ===== T5 tokenizzato =====
        if ('user_input_ids' in enc and 'pos_input_ids' in enc
                and 'neg_input_ids' in enc):
            user_ids = torch.stack([b[3]['user_input_ids']      for b in batch])
            user_msk = torch.stack([b[3]['user_attention_mask'] for b in batch])
            pos_ids  = torch.stack([b[3]['pos_input_ids']       for b in batch])
            pos_msk  = torch.stack([b[3]['pos_attention_mask']  for b in batch])
            neg_ids  = torch.stack([b[3]['neg_input_ids']       for b in batch])
            neg_msk  = torch.stack([b[3]['neg_attention_mask']  for b in batch])

            users = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
            pos_it = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
            neg_it = torch.as_tensor([b[2] for b in batch], dtype=torch.long)

            encodings = {
                'user_input_ids': user_ids,
                'user_attention_mask': user_msk,
                'pos_input_ids': pos_ids,
                'pos_attention_mask': pos_msk,
                'neg_input_ids': neg_ids,
                'neg_attention_mask': neg_msk,
            }
            return ('t5', users, pos_it, neg_it, encodings)

    # ===== Classico =====
    users, pos_it, neg_it = zip(*batch)
    users = torch.as_tensor(users, dtype=torch.long)
    pos_it = torch.as_tensor(pos_it, dtype=torch.long)
    neg_it = torch.as_tensor(neg_it, dtype=torch.long)
    return ('classic', users, pos_it, neg_it)


# TRAINING
def train_bpr(**kwargs):
    # Config
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    opt.parse(kwargs)

    # se viene passato esplicitamente (narre/deepconn/auto)
    if 'precomputed_mode' in kwargs:
        opt.precomputed_mode = kwargs['precomputed_mode']

    if not opt.pth_path:
        os.makedirs("./checkpoints", exist_ok=True)
        opt.pth_path = "./checkpoints"

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)
    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    device = torch.device("cuda" if opt.use_gpu else "cpu")

    # Modalità
    opt.use_precomputed = bool(kwargs.get('use_precomputed', getattr(opt, 'use_precomputed', False)))

    if opt.use_precomputed:
        print("Modalità PRECOMPUTED (DeepCoNN/NARRE): disabilito T5.")
        opt.use_t5 = False

        # --- lettura meta per info dim e pooling ---
        narre_meta_path = os.path.join(opt.data_root, 'embeddings_narre_meta.json')
        dc_meta_path    = os.path.join(opt.data_root, 'embeddings_meta.json')
        meta = {}
        if os.path.exists(narre_meta_path):
            with open(narre_meta_path, 'r') as f:
                meta = json.load(f)
            opt.precomputed_pooling_type = meta.get('pooling', 'review_sequence')
            opt.precomputed_dim = int(meta.get('emb_dim', 768))
            opt.user_max_reviews = meta.get('user_max_reviews', 10)
            opt.item_max_reviews = meta.get('item_max_reviews', 28)
            opt.review_max_len   = meta.get('review_max_len', 58)
            print(f"Meta NARRE: pooling='{opt.precomputed_pooling_type}', "
                  f"dim={opt.precomputed_dim}, "
                  f"u_max_r={opt.user_max_reviews}, "
                  f"i_max_r={opt.item_max_reviews}, "
                  f"r_max_len={opt.review_max_len}")
        elif os.path.exists(dc_meta_path):
            with open(dc_meta_path, 'r') as f:
                meta = json.load(f)
            opt.precomputed_pooling_type = meta.get('pooling', 'sequence')
            opt.precomputed_dim = int(meta.get('emb_dim', 768))
            opt.max_seq_length  = meta.get('max_len', 256)
            print(f"Meta DeepCoNN: pooling='{opt.precomputed_pooling_type}', "
                  f"dim={opt.precomputed_dim}, max_len={opt.max_seq_length}")
        else:
            opt.precomputed_pooling_type = 'unknown'
            opt.precomputed_dim = getattr(opt, 'precomputed_dim', 768)
            print(f"Nessun meta embedding trovato. Uso dim predefinita: {opt.precomputed_dim}")
        tokenizer = None
    else:
        tokenizer = None
        if opt.use_t5:
            tokenizer, t5_encoder_fn, t5_model = load_t5_model_and_tokenizer(opt.t5_model_name, device)
            opt.t5_encoder = t5_encoder_fn

    # Modello
    model = Model(opt, getattr(models, opt.model))
    model.to(device)
    if opt.use_gpu and len(opt.gpu_ids) > 0:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    # Dataset & Loader
    train_data = ReviewData(
        opt.data_root,
        mode="Train",
        setup="BPR",
        tokenizer=tokenizer,
        max_length=256,
        return_tokenized=(opt.use_t5 and not opt.use_precomputed),
        return_embeddings=opt.use_precomputed,
        precomputed_mode=getattr(opt, "precomputed_mode", "auto")
    )
    val_data = ReviewData(
        opt.data_root,
        mode="Val",
        setup="BPR",
        tokenizer=tokenizer,
        max_length=256,
        return_tokenized=(opt.use_t5 and not opt.use_precomputed),
        return_embeddings=opt.use_precomputed,
        precomputed_mode=getattr(opt, "precomputed_mode", "auto")
    )

    # DataLoader più veloce
    num_workers = getattr(opt, "num_workers", 4)
    pin_memory = bool(opt.use_gpu)
    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=collate_fn_bpr,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    print(f"train data: {len(train_data)}; val data: {len(val_data)}")

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    bpr_loss_func = BPRLoss()

    print("Inizio training BPR...")
    min_loss = 1e+10

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}")

        for _, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            mode = batch[0]

            if mode == 'emb':
                # PRECOMPUTED (DeepCoNN / NARRE)
                _, users, pos_it, neg_it, user_vec, pos_vec, neg_vec = batch

                user_vec = user_vec.to(device, non_blocking=True)
                pos_vec  = pos_vec.to(device, non_blocking=True)
                neg_vec  = neg_vec.to(device, non_blocking=True)

                pos_scores = model((user_vec, pos_vec))
                neg_scores = model((user_vec, neg_vec))
                loss = bpr_loss_func(pos_scores, neg_scores)

            elif mode == 't5':
                _, users, pos_it, neg_it, enc = batch
                positives, negatives = unpack_input_t5(opt, (users, pos_it, neg_it, enc), setup="BPR")
                for k in positives:
                    positives[k] = positives[k].to(device, non_blocking=True)
                    negatives[k] = negatives[k].to(device, non_blocking=True)
                pos_scores = model(positives)
                neg_scores = model(negatives)
                loss = bpr_loss_func(pos_scores, neg_scores)

            else:  # classic
                # if opt.use_precomputed:
                    # Ignora batch che finirebbe nel ramo classic
                #   continue
                _, users, pos_it, neg_it = batch
                positives, negatives = unpack_input(opt, zip(users, pos_it, neg_it), setup="BPR")
                positives = [x.to(device, non_blocking=True) for x in positives]
                negatives = [x.to(device, non_blocking=True) for x in negatives]
                pos_scores = model(positives)
                neg_scores = model(negatives)
                loss = bpr_loss_func(pos_scores, neg_scores)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_data))

        # Validazione più leggera: meno utenti e meno item
        neg_recall_sum, neg_avg_recall = predict_bpr(model, val_data, opt, k=10, N=100, U=64)
        recall_sum = -neg_recall_sum
        avg_recall = -neg_avg_recall

        print(f"Epoch {epoch} | avg_loss={avg_loss:.6f} | "
              f"recall_sum@10={recall_sum:.6f} | avg_recall@10={avg_recall:.6f}")

        # Salvataggio best (sulla metrica 'neg_recall_sum' più bassa)
        if neg_recall_sum < min_loss:
            min_loss = neg_recall_sum
            tag = "Pre" if opt.use_precomputed else ("T5" if opt.use_t5 else "Classic")
            model_name = f"{opt.pth_path}/BPR_model_{opt.model}_{opt.dataset}_{opt.setup}_{tag}.pth"
            torch.save(model.state_dict(), model_name)
            print(f"💾 Salvato miglior modello: {model_name}")
        print("*" * 30)


# PREDICT / EVAL (Recall@K, senza AUC)
def predict_bpr(model, data, opt, k=10, N=100, U=64):
    """
    Versione più leggera:
    - per default valuta solo U=64 utenti
    - con al massimo N=100 negativi per utente
    """
    if N == -1:
        N = float("inf")
    if U == -1:
        U = float("inf")

    recall_k_sum = 0.0
    model.eval()
    device = next(model.parameters()).device

    users = list(data.positive_items.keys())
    if len(users) > U:
        users = random.sample(users, U)

    with torch.no_grad():
        for user in users:
            positives = list(data.positive_items[user])

            interacted = []
            if hasattr(data, "interacted_train") and user in getattr(data, "interacted_train", {}):
                interacted.extend(data.interacted_train[user])
            if hasattr(data, "interacted_val") and user in getattr(data, "interacted_val", {}):
                interacted.extend(data.interacted_val[user])

            candidates = [i for i in data.all_items
                          if i not in interacted and i not in positives]
            if len(candidates) > N:
                candidates = random.sample(candidates, N)

            all_items = candidates + positives
            if not all_items:
                continue

            # Shuffle per evitare che con score piatti i positivi restino sempre in fondo
            random.shuffle(all_items)

            if getattr(opt, "use_precomputed", False):
                # Precomputed DeepCoNN / NARRE
                uvec_np = data.get_user_vec(int(user))
                if uvec_np is None:
                    continue

                ivecs_np = [data.get_item_vec(int(i)) for i in all_items]
                mask = [v is not None for v in ivecs_np]
                all_items = [i for i, keep in zip(all_items, mask) if keep]
                if not all_items:
                    continue
                ivecs_np = [v for v in ivecs_np if v is not None]

                uvec = torch.as_tensor(
                    np.stack([uvec_np] * len(all_items)),
                    dtype=torch.float32,
                    device=device
                )
                ivec = torch.as_tensor(
                    np.stack(ivecs_np),
                    dtype=torch.float32,
                    device=device
                )
                scores = model((uvec, ivec)).detach().view(-1).cpu().numpy()

            elif opt.use_t5:
                tokenizer = data.tokenizer
                user_texts = [data.get_user_text(user) for _ in all_items]
                item_texts = [data.get_item_text(i) for i in all_items]
                uenc = tokenizer(
                    user_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                )
                ienc = tokenizer(
                    item_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                )
                x = {
                    "user_input_ids": uenc["input_ids"].to(device),
                    "user_attention_mask": uenc["attention_mask"].to(device),
                    "item_input_ids": ienc["input_ids"].to(device),
                    "item_attention_mask": ienc["attention_mask"].to(device),
                    "user_ids": torch.tensor([user] * len(all_items),
                                             dtype=torch.long,
                                             device=device),
                    "item_ids": torch.tensor(all_items,
                                             dtype=torch.long,
                                             device=device),
                }
                scores = model(x).detach().view(-1).cpu().numpy()

            else:
                tmp = [user for _ in range(len(all_items))]
                x = unpack_input(opt, zip(tmp, all_items))
                x = [t.to(device) for t in x]
                scores = model(x).detach().view(-1).cpu().numpy()

            ranked = sorted(zip(all_items, scores), key=lambda p: p[1], reverse=True)
            top_k = [i for i, _ in ranked[:k]]
            hits = sum(1 for i in top_k if i in positives)
            recall_k_sum += hits / max(1, len(positives))

    avg_recall = recall_k_sum / max(1, len(users))
    model.train()
    return -recall_k_sum, -avg_recall


# TEST + METRICHE (Recall, Precision, HitRate, AUC)
def predict_bpr_with_auc(model, data, opt, k=10, N=200, U=128):

    if N == -1:
        N = float("inf")
    if U == -1:
        U = float("inf")

    recall_k = 0.0
    precision_k_total = 0.0
    hit_rate_k_total = 0
    auc_total = 0.0

    model.eval()
    device = next(model.parameters()).device

    PREDICTION_BATCH_SIZE = 128

    with torch.no_grad():
        users = list(data.positive_items.keys())
        if len(users) > U:
            users = random.sample(users, U)

        num_users_to_eval = max(1, len(users))

        # --- TUTTI GLI ITEM EMBEDDING IN CPU ---
        print("Caricamento item embeddings in memoria (CPU)…")
        all_item_embeddings = torch.as_tensor(data.item_embs[:], dtype=torch.float32)
        print("…Fatto. Gli embeddings RESTANO su CPU.")

        for user in tqdm(users, desc="Valutazione BPR"):
            interacted = []
            positives = list(data.positive_items[user])

            if hasattr(data, "interacted_train") and user in data.interacted_train:
                interacted.extend(data.interacted_train[user])
            if hasattr(data, "interacted_val") and user in data.interacted_val:
                interacted.extend(data.interacted_val[user])

            # Item candidati non interagiti
            all_item = [item for item in data.all_items
                        if item not in interacted and item not in positives]

            if len(all_item) > N:
                all_item = random.sample(all_item, N)

            all_item += positives
            if not all_item:
                continue

            all_scores = []

            if getattr(opt, "use_precomputed", False):
                uvec_np = data.get_user_vec(int(user))
                if uvec_np is None:
                    continue

                uvec_tensor_full = torch.as_tensor(
                    uvec_np,
                    dtype=torch.float32,
                    device=device
                )

                for j in range(0, len(all_item), PREDICTION_BATCH_SIZE):
                    batch_item_ids = all_item[j:j + PREDICTION_BATCH_SIZE]

                    # slice da CPU
                    ivec_cpu = all_item_embeddings[batch_item_ids]
                    ivec = ivec_cpu.to(device)

                    # Adatta la forma di uvec al batch
                    if uvec_tensor_full.ndim == 2 and ivec.ndim == 3:
                        # DeepCoNN
                        uvec = uvec_tensor_full.unsqueeze(0).expand(
                            ivec.shape[0], -1, -1
                        )
                    elif uvec_tensor_full.ndim == 3 and ivec.ndim == 4:
                        # NARRE
                        uvec = uvec_tensor_full.unsqueeze(0).expand(
                            ivec.shape[0], -1, -1, -1
                        )
                    else:
                        raise ValueError(
                            f"Dimensioni inattese in predict_bpr_with_auc: "
                            f"uvec.ndim={uvec_tensor_full.ndim}, ivec.ndim={ivec.ndim}"
                        )

                    batch_scores = model((uvec, ivec)).detach().view(-1)
                    all_scores.append(batch_scores)

            elif opt.use_t5:
                raise NotImplementedError("La logica T5 on-the-fly non è implementata in predict_bpr_with_auc.")
            else:
                raise NotImplementedError("La logica Classica non è implementata in predict_bpr_with_auc.")

            if not all_scores:
                continue

            Y = torch.cat(all_scores).cpu().numpy()

            z = list(zip(all_item, Y))
            if not z:
                continue

            z = sorted(z, key=lambda x: x[1], reverse=True)
            top_k = z[:k]

            total_relevant = len(positives)
            count = sum(1 for item, _ in top_k if item in positives)

            if total_relevant > 0:
                recall_k += (count / total_relevant)

            precision_k_total += count / k
            if count > 0:
                hit_rate_k_total += 1

            labels = [1 if item in positives else 0 for item, _ in top_k]
            scores = [score for _, score in top_k]

            if len(set(labels)) > 1:
                try:
                    auc_total += roc_auc_score(labels, scores)
                except ValueError:
                    pass

    avg_recall = recall_k / num_users_to_eval
    avg_precision = precision_k_total / num_users_to_eval
    avg_hit_rate = hit_rate_k_total / num_users_to_eval
    avg_auc = auc_total / num_users_to_eval

    model.train()
    return -recall_k, -avg_recall, avg_precision, avg_hit_rate, avg_auc


# TEST
def test_bpr(**kwargs):
    # Config
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    opt.parse(kwargs)
    if 'precomputed_mode' in kwargs:
        opt.precomputed_mode = kwargs['precomputed_mode']

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    device = torch.device("cuda" if opt.use_gpu else "cpu")

    opt.use_precomputed = bool(kwargs.get('use_precomputed', getattr(opt, 'use_precomputed', False)))

    if opt.use_precomputed:
        print("Modalità precomputed: skip T5.")
        opt.use_t5 = False

        narre_meta_path = os.path.join(opt.data_root, 'embeddings_narre_meta.json')
        dc_meta_path    = os.path.join(opt.data_root, 'embeddings_meta.json')
        meta = {}
        if os.path.exists(narre_meta_path):
            with open(narre_meta_path, 'r') as f:
                meta = json.load(f)
            opt.precomputed_pooling_type = meta.get('pooling', 'review_sequence')
            opt.precomputed_dim = int(meta.get('emb_dim', 768))
            opt.user_max_reviews = meta.get('user_max_reviews', 10)
            opt.item_max_reviews = meta.get('item_max_reviews', 28)
            opt.review_max_len   = meta.get('review_max_len', 58)
            print(f"Meta NARRE: pooling='{opt.precomputed_pooling_type}', "
                  f"dim={opt.precomputed_dim}")
        elif os.path.exists(dc_meta_path):
            with open(dc_meta_path, 'r') as f:
                meta = json.load(f)
            opt.precomputed_pooling_type = meta.get('pooling', 'sequence')
            opt.precomputed_dim = int(meta.get('emb_dim', 768))
            opt.max_seq_length  = meta.get('max_len', 256)
            print(f"Meta DeepCoNN: pooling='{opt.precomputed_pooling_type}', "
                  f"dim={opt.precomputed_dim}")
        else:
            opt.precomputed_pooling_type = 'unknown'
            opt.precomputed_dim = getattr(opt, 'precomputed_dim', 768)
            print(f"Nessun meta embedding trovato. Uso dim predefinita: {opt.precomputed_dim}")
        tokenizer = None
    elif opt.use_t5:
        tokenizer, t5_encoder_fn, t5_model = load_t5_model_and_tokenizer(opt.t5_model_name, device)
        opt.t5_encoder = t5_encoder_fn
    else:
        tokenizer = None

    # Modello
    model = Model(opt, getattr(models, opt.model))
    model.to(device)
    tag = "Pre" if opt.use_precomputed else ("T5" if opt.use_t5 else "Classic")
    model_path = f"{opt.pth_path}/BPR_model_{opt.model}_{opt.dataset}_{opt.setup}_{tag}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modello caricato da: {model_path}")

    test_data = ReviewData(
        opt.data_root,
        mode="Test",
        setup="BPR",
        tokenizer=tokenizer,
        max_length=256,
        return_tokenized=(opt.use_t5 and not opt.use_precomputed),
        return_embeddings=opt.use_precomputed,
        precomputed_mode=getattr(opt, "precomputed_mode", "auto")
    )

    print(f"{now()}: test su dataset completo")

    recall, avg_recall, avg_precision, avg_hit_rate, avg_auc = predict_bpr_with_auc(
        model, test_data, opt, U=-1, N=-1
    )
    print(f"Recall medio @10: {abs(avg_recall):.4f}")
    print(f"Precision medio @10: {avg_precision:.4f}")
    print(f"Hit Rate medio @10: {avg_hit_rate:.4f}")
    print(f"AUC medio: {avg_auc:.4f}")
    return recall, avg_recall, avg_precision, avg_hit_rate, avg_auc
