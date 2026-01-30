# -*- coding: utf-8 -*-
import json
import time
import random
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import fire

# Dataset aggiornato (con supporto embeddings precalcolati)
from dataset.data_review_t5PRO import ReviewData
from framework import Model
from models.Losses import *
import models
import config
from main import collate_fn, unpack_input  # percorso classico

# T5 (solo quando use_t5=True)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# BPR module
import BPR_t5PRO as bpr_module


def core_unpack(opt, uids, iids):
    uids = list(uids)
    iids = list(iids)
    user_reviews = opt.users_review[list(uids)]
    user_item2id = opt.user2itemid[list(uids)]
    user_doc = opt.user_doc[uids]
    item_reviews = opt.items_review[list(iids)]
    item_user2id = opt.item2userid[list(iids)]
    item_doc = opt.item_doc[iids]
    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x), data))
    return data


def unpack_input_t5(opt, batch_data, setup="Default"):
    if setup == "Default":
        if len(batch_data) == 3:
            (user_ids, item_ids), scores, encodings = batch_data
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, dtype=torch.float)
            else:
                scores = scores.to(torch.float)
            return {
                'user_ids': torch.tensor(user_ids, dtype=torch.long),
                'item_ids': torch.tensor(item_ids, dtype=torch.long),
                'scores': scores,
                'user_input_ids': encodings['user_input_ids'],
                'user_attention_mask': encodings['user_attention_mask'],
                'item_input_ids': encodings['item_input_ids'],
                'item_attention_mask': encodings['item_attention_mask']
            }
        else:
            return unpack_input(opt, batch_data, setup)
    elif setup == "BPR":
        if len(batch_data) == 4:
            users, pos_items, neg_items, encodings = batch_data
            users = torch.tensor(users, dtype=torch.long)
            pos_items = torch.tensor(pos_items, dtype=torch.long)
            neg_items = torch.tensor(neg_items, dtype=torch.long)

            pos_data = {
                'users': users,
                'pos_items': pos_items,
                'user_input_ids': encodings['user_input_ids'],
                'user_attention_mask': encodings['user_attention_mask'],
                'item_input_ids': encodings['pos_input_ids'],
                'item_attention_mask': encodings['pos_attention_mask']
            }

            neg_data = {
                'users': users,
                'neg_items': neg_items,
                'user_input_ids': encodings['user_input_ids'],
                'user_attention_mask': encodings['user_attention_mask'],
                'item_input_ids': encodings['neg_input_ids'],
                'item_attention_mask': encodings['neg_attention_mask']
            }

            return pos_data, neg_data
        else:
            return unpack_input(opt, batch_data, setup)

def unpack_input_emb(batch_data, setup="Default"):
    """
    Converte i sample di ReviewData(return_embeddings=True) in dict di tensori.
    Default: ((uids,iids), scores, {'user_vec','item_vec'})
    NB: il modello (DeepCoNN / NARRE PRECOMPUTED) riceve poi una TUPLA (user_vec, item_vec).
    """
    (user_ids, item_ids), scores, enc = batch_data
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    item_ids = torch.tensor(item_ids, dtype=torch.long)
    # Sono già tensori in ReviewData, ma torch.as_tensor è safe
    user_vec = torch.as_tensor(enc["user_vec"], dtype=torch.float32)
    item_vec = torch.as_tensor(enc["item_vec"], dtype=torch.float32)
    return {
        'user_ids': user_ids,
        'item_ids': item_ids,
        'scores': scores,
        'user_vec': user_vec,
        'item_vec': item_vec,
    }


def load_t5_model_and_tokenizer(model_name, device=None):
    print(f"Caricamento T5-GEMMA: {model_name}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.bfloat16)
    t5_model.to(device)
    for param in t5_model.parameters():
        param.requires_grad = False
    t5_encoder_fn = lambda input_ids, attention_mask: t5_model.get_encoder()(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        return_dict=True
    ).last_hidden_state
    print(f"T5-GEMMA caricato: {sum(p.numel() for p in t5_model.parameters()) / 1e6:.1f}M parametri")
    return tokenizer, t5_encoder_fn, t5_model


# TRAINING
def train(**kwargs):
    # BPR delegato al modulo dedicato
    if 'setup' in kwargs and kwargs['setup'] == 'BPR':
        print("Richiamo modulo BPR per training...")
        return bpr_module.train_bpr(**kwargs)

    # Config
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    # Se passi precomputed_mode=auto/narre/deepconn da riga di comando
    if 'precomputed_mode' in kwargs:
        opt.precomputed_mode = kwargs['precomputed_mode']

    opt.parse(kwargs)  # se use_precomputed=True, qui chiama _maybe_load_precomputed_meta()

    # Flag embeddings precalcolati
    opt.use_precomputed = bool(kwargs.get('use_precomputed', getattr(opt, 'use_precomputed', False)))
    if opt.use_precomputed:
        opt.use_t5 = False
        # meta (emb_dim, ecc.) sono già caricati da config._maybe_load_precomputed_meta()
        print(f"Modalità PRECOMPUTED attiva (mode={getattr(opt, 'precomputed_mode', 'auto')})")

    if not opt.pth_path:
        os.makedirs("./checkpoints", exist_ok=True)
        opt.pth_path = "./checkpoints"

    mode_str = 'Precomputed' if opt.use_precomputed else ('T5-GEMMA' if opt.use_t5 else 'Classica')
    print(f"Avvio training - Modalità: {mode_str}")

    device = torch.device("cuda" if opt.use_gpu else "cpu")

    tokenizer, t5_model = None, None
    if opt.use_t5 and not opt.use_precomputed:
        tokenizer, t5_encoder_fn, t5_model = load_t5_model_and_tokenizer(opt.t5_model_name)
        opt.t5_encoder = t5_encoder_fn

    # Dataset
    if opt.use_precomputed:
        train_data = ReviewData(
            opt.data_root, 'Train',
            setup=opt.setup,
            return_embeddings=True,
            precomputed_mode=getattr(opt, 'precomputed_mode', 'auto')
        )
        val_data = ReviewData(
            opt.data_root, 'Val',
            setup=opt.setup,
            return_embeddings=True,
            precomputed_mode=getattr(opt, 'precomputed_mode', 'auto')
        )
    elif opt.use_t5:
        train_data = ReviewData(
            opt.data_root, 'Train',
            tokenizer=tokenizer, max_length=opt.max_seq_length,
            return_tokenized=True, setup=opt.setup
        )
        val_data = ReviewData(
            opt.data_root, 'Val',
            tokenizer=tokenizer, max_length=opt.max_seq_length,
            return_tokenized=True, setup=opt.setup
        )
    else:
        train_data = ReviewData(opt.data_root, 'Train', setup=opt.setup)
        val_data = ReviewData(opt.data_root, 'Val',   setup=opt.setup)

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data,   batch_size=opt.batch_size, shuffle=False)
    print(f"Dataset: Train={len(train_data)}, Val={len(val_data)}")

    # Modello
    model = Model(opt, getattr(models, opt.model))
    model.to(device)
    if opt.multi_gpu and opt.use_gpu:
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    best_val_loss = float('inf')
    for epoch in range(opt.num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, train_datas in enumerate(train_dataloader):
            optimizer.zero_grad()

            if opt.use_precomputed:
                batch_data = unpack_input_emb(train_datas, opt.setup)
                for k in ['scores', 'user_vec', 'item_vec']:
                    if isinstance(batch_data[k], torch.Tensor):
                        batch_data[k] = batch_data[k].to(device)
                # PASSA TUPLA AL MODELLO
                output = model((batch_data['user_vec'], batch_data['item_vec']))
                loss = torch.nn.MSELoss()(output, batch_data['scores'])

            elif opt.use_t5:
                batch_data = unpack_input_t5(opt, train_datas, opt.setup)
                for k in ['user_input_ids', 'user_attention_mask',
                          'item_input_ids', 'item_attention_mask', 'scores']:
                    batch_data[k] = batch_data[k].to(device)
                output = model(batch_data)
                loss = torch.nn.MSELoss()(output, batch_data['scores'])

            else:
                train_datas = unpack_input(opt, train_datas, opt.setup)
                train_datas[-1] = train_datas[-1].to(device)
                output = model(train_datas)
                loss = torch.nn.MSELoss()(output, train_datas[-1])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_datas in val_dataloader:
                if opt.use_precomputed:
                    batch_data = unpack_input_emb(val_datas, opt.setup)
                    for k in ['scores', 'user_vec', 'item_vec']:
                        if isinstance(batch_data[k], torch.Tensor):
                            batch_data[k] = batch_data[k].to(device)
                    output = model((batch_data['user_vec'], batch_data['item_vec']))
                    loss = torch.nn.MSELoss()(output, batch_data['scores'])

                elif opt.use_t5:
                    batch_data = unpack_input_t5(opt, val_datas, opt.setup)
                    for k in ['user_input_ids', 'user_attention_mask',
                              'item_input_ids', 'item_attention_mask', 'scores']:
                        batch_data[k] = batch_data[k].to(device)
                    output = model(batch_data)
                    loss = torch.nn.MSELoss()(output, batch_data['scores'])

                else:
                    val_datas = unpack_input(opt, val_datas, opt.setup)
                    val_datas[-1] = val_datas[-1].to(device)
                    output = model(val_datas)
                    loss = torch.nn.MSELoss()(output, val_datas[-1])
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(opt.pth_path, exist_ok=True)
            suffix = 'Pre' if opt.use_precomputed else ('T5' if opt.use_t5 else 'Classic')
            model_name = f"{opt.pth_path}/best_model_{opt.model}_{opt.dataset}_{opt.setup}_{suffix}.pth"
            torch.save(model.state_dict(), model_name)
            print(f"Salvato miglior modello: {model_name}")


def test(**kwargs):
    # BPR delegato al modulo dedicato
    if 'setup' in kwargs and kwargs['setup'] == 'BPR':
        print("Richiamo modulo BPR per test...")
        return bpr_module.test_bpr(**kwargs)

    # Config
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    if 'precomputed_mode' in kwargs:
        opt.precomputed_mode = kwargs['precomputed_mode']

    opt.parse(kwargs)  # se use_precomputed=True, carica meta giusti

    opt.use_precomputed = bool(kwargs.get('use_precomputed', getattr(opt, 'use_precomputed', False)))
    if opt.use_precomputed:
        opt.use_t5 = False
        print(f"Modalità PRECOMPUTED attiva in test (mode={getattr(opt, 'precomputed_mode', 'auto')})")

    print(f"Avvio test - Modalità: {'Precomputed' if opt.use_precomputed else ('T5-GEMMA' if opt.use_t5 else 'Classica')}")

    device = torch.device("cuda" if opt.use_gpu else "cpu")
    tokenizer, t5_model = None, None
    if opt.use_t5 and not opt.use_precomputed:
        tokenizer, t5_encoder_fn, t5_model = load_t5_model_and_tokenizer(opt.t5_model_name)
        opt.t5_encoder = t5_encoder_fn

    # Dataset
    if opt.use_precomputed:
        test_data = ReviewData(
            opt.data_root, 'Test',
            setup=opt.setup,
            return_embeddings=True,
            precomputed_mode=getattr(opt, 'precomputed_mode', 'auto')
        )
    elif opt.use_t5:
        test_data = ReviewData(
            opt.data_root, 'Test',
            tokenizer=tokenizer, max_length=opt.max_seq_length,
            return_tokenized=True, setup=opt.setup
        )
    else:
        test_data = ReviewData(opt.data_root, 'Test', setup=opt.setup)

    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    # Modello
    model = Model(opt, getattr(models, opt.model))
    model.to(device)
    suffix = 'Pre' if opt.use_precomputed else ('T5' if opt.use_t5 else 'Classic')
    model_name = f"{opt.pth_path}/best_model_{opt.model}_{opt.dataset}_{opt.setup}_{suffix}.pth"
    model.load_state_dict(torch.load(model_name, map_location=device))
    print(f"Modello caricato da: {model_name}")
    model.eval()

    predictions, targets = [], []
    with torch.no_grad():
        for test_datas in test_dataloader:
            if opt.use_precomputed:
                batch_data = unpack_input_emb(test_datas, opt.setup)
                for k in ['scores', 'user_vec', 'item_vec']:
                    if isinstance(batch_data[k], torch.Tensor):
                        batch_data[k] = batch_data[k].to(device)
                output = model((batch_data['user_vec'], batch_data['item_vec']))
                predictions.extend(output.detach().cpu().numpy())
                targets.extend(batch_data['scores'].detach().cpu().numpy())

            elif opt.use_t5:
                batch_data = unpack_input_t5(opt, test_datas, opt.setup)
                for k in ['user_input_ids', 'user_attention_mask',
                          'item_input_ids', 'item_attention_mask', 'scores']:
                    batch_data[k] = batch_data[k].to(device)
                output = model(batch_data)
                predictions.extend(output.cpu().numpy())
                targets.extend(batch_data['scores'].cpu().numpy())

            else:
                test_datas = unpack_input(opt, test_datas, opt.setup)
                test_datas[-1] = test_datas[-1].to(device)
                output = model(test_datas)
                predictions.extend(output.cpu().numpy())
                targets.extend(test_datas[-1].cpu().numpy())

    mse = np.mean((np.array(predictions) - np.array(targets)) ** 2)
    print(f"Test MSE: {mse:.4f}")
    return mse


if __name__ == '__main__':
    fire.Fire()
