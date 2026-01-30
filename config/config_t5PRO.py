# -*- coding: utf-8 -*-
"""
config_t5PRO.py — Config estesa: Classico, T5-GEMMA, Precomputed).
"""

import os, json
import numpy as np
import torch

#CONFIG BASE (DEFAULT)

class DefaultConfig:

    def __init__(self):
        self.data_root = "./dataset"


    #  T5-GEMMA
    model = 'DeepCoNN'
    dataset = 'Digital_Music_data'

    use_t5 = False
    t5_model_name = "./t5gemma_offline"
    t5_hidden_size = 2304
    max_seq_length = 256
    truncation = True
    padding = "max_length"
    freeze_t5_encoder = True
    t5_encoder = None  # popolato a runtime

    #  PRECOMPUTED

    use_precomputed = False
    precomputed_dim = 768
    precomputed_mode = "auto"

    user_emb_path = None
    item_emb_path = None
    emb_meta_path = None


    #  Base

    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 100
    num_workers = 0

    optimizer = 'AdamW'
    weight_decay = 1e-3
    lr = 1e-3
    loss_method = 'mse'
    drop_out = 0.5

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'cat'
    ui_merge = 'cat'
    output = 'lfm'

    fine_step = False
    pth_path = ""
    print_opt = 'default'
    setup = "Default"


    #  PATH SETUP

    def set_path(self, name):
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        # Percorsi classico
        self.user_list_path    = f'{prefix}/userReview2Index.npy'
        self.item_list_path    = f'{prefix}/itemReview2Index.npy'
        self.user2itemid_path  = f'{prefix}/user_item2id.npy'
        self.item2userid_path  = f'{prefix}/item_user2id.npy'
        self.user_doc_path     = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path     = f'{prefix}/itemDoc2Index.npy'
        self.w2v_path          = f'{prefix}/w2v_'

        # Precomputed
        self.user_emb_path = f'{self.data_root}/user_embeddings.npy'
        self.item_emb_path = f'{self.data_root}/item_embeddings.npy'
        self.emb_meta_path = f'{self.data_root}/embeddings_meta.json'

        # NARRE alternative
        self.narre_user_emb = f'{self.data_root}/user_review_embeddings.npy'
        self.narre_item_emb = f'{self.data_root}/item_review_embeddings.npy'
        self.narre_meta     = f'{self.data_root}/embeddings_narre_meta.json'

        # Fix compatibilità precomputed: liste Python
        self.users_review_list  = [None] * getattr(self, 'user_num', 0)
        self.items_review_list  = [None] * getattr(self, 'item_num', 0)
        self.user2itemid_list   = [None] * getattr(self, 'user_num', 0)
        self.item2userid_list   = [None] * getattr(self, 'item_num', 0)
        self.user_doc           = [None] * getattr(self, 'user_num', 0)
        self.item_doc           = [None] * getattr(self, 'item_num', 0)


    #  CARICAMENTO META PRECOMPUTED
    def _maybe_load_precomputed_meta(self):
        if self.precomputed_mode == "auto":
            if os.path.exists(self.narre_meta):
                self.user_emb_path = self.narre_user_emb
                self.item_emb_path = self.narre_item_emb
                self.emb_meta_path = self.narre_meta
                print(f" Auto-detect: trovati file NARRE → uso NARRE")
            elif os.path.exists(self.emb_meta_path):
                print(f" Auto-detect: trovati file DeepCoNN → uso DeepCoNN")
            else:
                print("️ Nessun file precomputed trovato (NARRE o DeepCoNN).")
        elif self.precomputed_mode == "narre":
            if os.path.exists(self.narre_meta):
                self.user_emb_path = self.narre_user_emb
                self.item_emb_path = self.narre_item_emb
                self.emb_meta_path = self.narre_meta
                print(f" Forzato: uso NARRE")
            else:
                print(" precomputed_mode='narre' ma file mancanti.")
        elif self.precomputed_mode == "deepconn":
            print(" Forzato: uso DeepCoNN")

        if self.emb_meta_path and os.path.exists(self.emb_meta_path):
            try:
                with open(self.emb_meta_path, 'r') as f:
                    meta = json.load(f)
                if 'emb_dim' in meta:
                    self.precomputed_dim = int(meta['emb_dim'])
                print(f" Precomputed meta → dim={self.precomputed_dim}, "
                      f"users={meta.get('num_users','?')}, items={meta.get('num_items','?')}")
            except:
                print(f"️ Impossibile leggere {self.emb_meta_path}")


    #  PARSING CONFIG (USATO DA main_t5PRO)

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"opt has No key: {k}")
            setattr(self, k, v)

        if not self.pth_path:
            self.pth_path = "./checkpoints"

        if self.use_precomputed:
            print(" Modalità PRECOMPUTED")
            self._maybe_load_precomputed_meta()
        elif not self.use_t5:
            print(" Loading classical npy files...")
            try:
                self.users_review_list  = np.load(self.user_list_path, allow_pickle=True)
                self.items_review_list  = np.load(self.item_list_path, allow_pickle=True)
                self.user2itemid_list   = np.load(self.user2itemid_path, allow_pickle=True)
                self.item2userid_list   = np.load(self.item2userid_path, allow_pickle=True)
                self.user_doc           = np.load(self.user_doc_path, allow_pickle=True)
                self.item_doc           = np.load(self.item_doc_path, allow_pickle=True)
                print(" Classical files loaded.")
            except:
                print(" Classical files missing.")
        else:
            print(" T5-GEMMA mode: skip npy.")

        if self.use_t5:
            if self.lr == 1e-3:
                self.lr = 5e-4
                print(f" Auto LR for T5: {self.lr}")
            if hasattr(self, 'batch_size') and self.batch_size == 64:
                self.batch_size = 32
                print(f" Auto batch_size for T5: {self.batch_size}")

        # Stampa config
        print("*************************************************")
        if self.use_precomputed:
            print("PRECOMPUTED CONFIG")
        elif self.use_t5:
            print(f"T5-GEMMA CONFIG ({self.t5_model_name})")
        else:
            print("CLASSICAL CONFIG")
        print("*************************************************")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(f"{k} => {getattr(self, k)}")
        print("*************************************************")

# CONFIG SPECIFICHE PER DATASET

class Digital_Music_data_Config(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.set_path("Digital_Music_data")

    vocab_size = 50001
    word_dim   = 200
    r_max_len  = 202
    u_max_r    = 13
    i_max_r    = 24

    train_data_size = 51764
    test_data_size  = 6471
    val_data_size   = 6471

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 64
    print_step = 100


class Toys_and_Games_5_data_Config(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.set_path("Toys_and_Games_5_data")

    vocab_size = 50002
    word_dim   = 100
    r_max_len  = 58
    u_max_r = 10
    i_max_r = 28

    train_data_size = 134087
    test_data_size  = 16755
    val_data_size   = 16755

    user_num = 19412 + 2
    item_num = 11924 + 2

    batch_size = 64
    print_step = 100


class Clothing_Shoes_and_Jewelry_5_data_Config(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.set_path("Clothing_Shoes_and_Jewelry_5_data")

    vocab_size = 50002
    word_dim   = 100
    r_max_len  = 59
    u_max_r = 10
    i_max_r = 28

    train_data_size = 222978
    test_data_size  = 27849
    val_data_size   = 27850

    user_num = 39387 + 2
    item_num = 23033 + 2

    batch_size = 64
    print_step = 100


class Baby_5_data_Config(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.set_path("Baby_5_data")

    vocab_size = 50002
    word_dim   = 100
    r_max_len  = 58
    u_max_r = 10
    i_max_r = 28

    train_data_size = 128644
    test_data_size  = 16074
    val_data_size   = 16074

    user_num = 19445 + 2
    item_num = 7050 + 2

    batch_size = 64
    print_step = 100
