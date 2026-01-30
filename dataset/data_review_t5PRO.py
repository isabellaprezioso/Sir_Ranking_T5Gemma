# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class ReviewData(Dataset):
    """
    Versione UNIFICATA:
    - DeepCoNN: embeddings 3D su disco  (num_entities, seq_len, emb_dim)
                → per singolo utente/item: (seq_len, emb_dim)  (ndim=2)
    - NARRE:    embeddings 4D su disco  (num_entities, max_reviews, review_len, emb_dim)
                → per singolo utente/item: (max_reviews, review_len, emb_dim) (ndim=3)

    Supporta:
      - setup="Default"
      - setup="BPR"

    Se return_embeddings=True:
      - NON c'è fallback al classico, se gli embedding ci sono.
    """

    def __init__(self,
                 root_path,
                 mode,
                 setup="Default",
                 user=None,
                 tokenizer=None,
                 max_length=256,
                 return_tokenized=False,
                 return_embeddings=False,
                 precomputed_mode="auto"
                 ):

        self.setup = setup
        self.mode = mode
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tokenized = return_tokenized and (tokenizer is not None)
        self.return_embeddings = return_embeddings
        self.precomputed_mode = precomputed_mode

        
        # LOAD DATA (Train / Val / Test / Inference)
        if mode == 'Train':
            p = os.path.join(root_path, 'train/')
            print("loading train data from", p)
            self.data = np.load(p + "Train.npy", allow_pickle=True)
            self.scores = np.load(p + "Train_Score.npy")

        elif mode == 'Val':
            p = os.path.join(root_path, 'val/')
            print("loading val data from", p)
            self.data = np.load(p + "Val.npy", allow_pickle=True)
            self.scores = np.load(p + "Val_Score.npy")

        elif mode == 'Inference':
            p = os.path.join(root_path, 'test/')
            print("loading test data from", p)
            base = np.load(p + "Test.npy", allow_pickle=True)
            user_id = user
            item_ids = np.unique(base[:, 1])
            self.data = [(user_id, iid) for iid in item_ids]
            self.scores = np.zeros(len(self.data), dtype=np.float32)

        else:  # Test
            p = os.path.join(root_path, 'test/')
            print("loading test data from", p)
            self.data = np.load(p + "Test.npy", allow_pickle=True)
            self.scores = np.load(p + "Test_Score.npy")

        # LOAD ID MAPS
        try:
            with open(os.path.join(root_path, "user2id.json"), "r", encoding="utf-8") as f:
                u2id = json.load(f)
            with open(os.path.join(root_path, "item2id.json"), "r", encoding="utf-8") as f:
                i2id = json.load(f)
            self.id2user = {int(v): k for k, v in u2id.items()}
            self.id2item = {int(v): k for k, v in i2id.items()}
        except Exception as e:
            print("[ReviewData] Warning: cannot load id maps:", e)
            self.id2user, self.id2item = {}, {}

        # LOAD user_texts / item_texts
        self.user_texts, self.item_texts = {}, {}

        candidates = [
            os.path.join(root_path, "user_texts.json"),
            os.path.join(root_path, "train/user_texts.json"),
        ]

        for c in candidates:
            if os.path.exists(c):
                base = os.path.dirname(c)
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        self.user_texts = json.load(f)
                    with open(os.path.join(base, "item_texts.json"), "r", encoding="utf-8") as f:
                        self.item_texts = json.load(f)
                    print("[ReviewData] Loaded user/item texts from", base)
                except Exception as e:
                    print("[ReviewData] Warning: cannot load texts JSONs:", e)
                break

        # LOAD PRECOMPUTED EMBEDDINGS
        self.user_embs = None
        self.item_embs = None
        self.embeddings_meta = None

        narre_ue = os.path.join(root_path, "user_review_embeddings.npy")
        narre_ie = os.path.join(root_path, "item_review_embeddings.npy")
        narre_me = os.path.join(root_path, "embeddings_narre_meta.json")

        dc_ue = os.path.join(root_path, "user_embeddings.npy")
        dc_ie = os.path.join(root_path, "item_embeddings.npy")
        dc_me = os.path.join(root_path, "embeddings_meta.json")

        def _try_load_narre():
            if not (os.path.exists(narre_ue) and os.path.exists(narre_ie) and os.path.exists(narre_me)):
                return False
            try:
                U = np.load(narre_ue, mmap_mode="r")
                I = np.load(narre_ie, mmap_mode="r")
                with open(narre_me, "r", encoding="utf-8") as f:
                    M = json.load(f)
                if U.ndim == 4 and I.ndim == 4 and M.get("pooling") == "review_sequence":
                    self.user_embs = U
                    self.item_embs = I
                    self.embeddings_meta = M
                    print(f"[ReviewData] Loaded NARRE embeddings {U.shape} / {I.shape}")
                    print(f"[ReviewData] Meta: {M}")
                    return True
                return False
            except Exception as e:
                print("[ReviewData] NARRE load error:", e)
                return False

        def _try_load_deepconn():
            if not (os.path.exists(dc_ue) and os.path.exists(dc_ie) and os.path.exists(dc_me)):
                return False
            try:
                U = np.load(dc_ue, mmap_mode="r")
                I = np.load(dc_ie, mmap_mode="r")
                with open(dc_me, "r", encoding="utf-8") as f:
                    M = json.load(f)
                if U.ndim == 3 and I.ndim == 3 and M.get("pooling") == "sequence":
                    self.user_embs = U
                    self.item_embs = I
                    self.embeddings_meta = M
                    print(f"[ReviewData] Loaded DeepCoNN embeddings {U.shape} / {I.shape}")
                    print(f"[ReviewData] Meta: {M}")
                    return True
                # Se sono 2D (precooked T5 embeddings), forzo shape 3D
                if U.ndim == 2 and I.ndim == 2 and M.get("pooling") == "sequence":
                    self.user_embs = U[:, np.newaxis, :]
                    self.item_embs = I[:, np.newaxis, :]
                    self.embeddings_meta = M
                    print(f"[ReviewData] Loaded DeepCoNN 2D embeddings reshaped to 3D {self.user_embs.shape} / {self.item_embs.shape}")
                    print(f"[ReviewData] Meta: {M}")
                    return True
                return False
            except Exception as e:
                print("[ReviewData] DeepCoNN load error:", e)
                return False

        # In base al precomputed_mode
        if self.return_embeddings:
            if precomputed_mode == "narre":
                if not _try_load_narre():
                    print("[ReviewData] precomputed_mode='narre' ma NARRE non caricabile.")
            elif precomputed_mode == "deepconn":
                if not _try_load_deepconn():
                    print("[ReviewData] precomputed_mode='deepconn' ma DeepCoNN non caricabile.")
            else:
                if not _try_load_narre():
                    _try_load_deepconn()

        # BPR logics
        if setup == "BPR":
            self.len = len(self.data)
            self.positive_items = self._positive_items_dict()

            if self.mode == "Train":
                self.all_items = np.unique(self.data[:, 1])
            else:
                base = np.load(os.path.join(root_path, "train/Train.npy"), allow_pickle=True)
                self.all_items = np.unique(base[:, 1])

            if mode == "Test":
                base_val = np.load(os.path.join(root_path, "val/Val.npy"), allow_pickle=True)
                scores_val = np.load(os.path.join(root_path, "val/Val_Score.npy"))
                self.interacted_val = self._positive_items_dict_from(base_val, scores_val)

            if mode in ["Val", "Test"]:
                base_tr = np.load(os.path.join(root_path, "train/Train.npy"), allow_pickle=True)
                scores_tr = np.load(os.path.join(root_path, "train/Train_Score.npy"))
                self.interacted_train = self._positive_items_dict_from(base_tr, scores_tr)

            self.x = self._generate_bpr_triples()
        else:
            self.x = list(zip(self.data, self.scores))

    # Helpers for texts
    def _id_to_key(self, idx, is_user=True):
        if is_user:
            return self.id2user.get(int(idx), str(int(idx)))
        else:
            return self.id2item.get(int(idx), str(int(idx)))

    def get_user_text(self, user_id, join_sentences=True, sep=" <SEP> "):
        key = self._id_to_key(user_id, is_user=True)
        reviews = self.user_texts.get(key, ["<unk>"])
        return sep.join(reviews) if join_sentences else reviews

    def get_item_text(self, item_id, join_sentences=True, sep=" <SEP> "):
        key = self._id_to_key(item_id, is_user=False)
        reviews = self.item_texts.get(key, ["<unk>"])
        return sep.join(reviews) if join_sentences else reviews

    # EMBEDDINGS HELPERS
    def get_user_vec(self, uid):
        if self.user_embs is None:
            return None
        if uid < 0 or uid >= self.user_embs.shape[0]:
            return None
        return self.user_embs[int(uid)]

    def get_item_vec(self, iid):
        if self.item_embs is None:
            return None
        if iid < 0 or iid >= self.item_embs.shape[0]:
            return None
        return self.item_embs[int(iid)]

    # BPR utilities
    def _positive_items_dict(self):
        d = {}
        for (u, i), s in zip(self.data, self.scores):
            if s < 4:
                continue
            d.setdefault(int(u), set()).add(int(i))
        return d

    def _positive_items_dict_from(self, data, scores):
        d = {}
        for (u, i), s in zip(data, scores):
            if s < 4:
                continue
            d.setdefault(int(u), set()).add(int(i))
        return d

    def _generate_bpr_triples(self):
        items = np.unique(self.data[:, 1])
        triples = []
        for (u, pos), s in zip(self.data, self.scores):
            u = int(u)
            pos = int(pos)
            if u not in self.positive_items:
                continue
            neg = self._sample_negative(u, items)
            if neg is not None:
                triples.append((u, pos, int(neg)))
        return triples

    def _sample_negative(self, user, items):
        cand = list(set(items) - self.positive_items[user])
        return random.choice(cand) if cand else None

    # __getitem__
    def __getitem__(self, idx):
        if self.setup == "BPR" and self.mode != "Train":
            idx = idx % len(self.x)

        sample = self.x[idx]

        pooling = self.embeddings_meta.get("pooling", None) if self.embeddings_meta else None

        # PRECOMPUTED EMBEDDINGS
        if self.return_embeddings and self.user_embs is not None:

            # Default setup 
            if self.setup == "Default":
                (uid, iid), score = sample
                uid = int(uid)
                iid = int(iid)
                U = self.get_user_vec(uid)
                I = self.get_item_vec(iid)

                if U is None or I is None:
                    print(f"[ReviewData] ERROR: missing embedding for u={uid} i={iid}")
                    return sample

                # DeepCoNN 2D
                if pooling == "sequence" and U.ndim == 2 and I.ndim == 2:
                    return ((uid, iid), score, {
                        "user_vec": torch.from_numpy(U.copy()).float(),
                        "item_vec": torch.from_numpy(I.copy()).float(),
                    })

                # NARRE 3D
                if pooling == "review_sequence" and U.ndim == 3 and I.ndim == 3:
                    return ((uid, iid), score, {
                        "user_vec": torch.from_numpy(U.copy()).float(),
                        "item_vec": torch.from_numpy(I.copy()).float(),
                    })

                print(f"[ReviewData] ERROR: invalid embedding shape in Default: "
                      f"pooling={pooling}, U.ndim={U.ndim}, I.ndim={I.ndim}")
                return sample

            # BPR setup 
            uid, pos, neg = sample
            uid = int(uid)
            pos = int(pos)
            neg = int(neg)
            U = self.get_user_vec(uid)
            P = self.get_item_vec(pos)
            N = self.get_item_vec(neg)

            if U is None or P is None or N is None:
                print(f"[ReviewData] ERROR: missing embeddings in BPR sample "
                      f"u={uid}, pos={pos}, neg={neg}")
                return sample

            # DeepCoNN 2D → forza shape 3D
            if pooling == "sequence":
                if U.ndim == 2:
                    U = U[np.newaxis, :]
                if P.ndim == 2:
                    P = P[np.newaxis, :]
                if N.ndim == 2:
                    N = N[np.newaxis, :]

            # DeepCoNN / NARRE
            if (pooling == "sequence" and U.ndim == 3 and P.ndim == 3 and N.ndim == 3) or \
               (pooling == "review_sequence" and U.ndim == 3 and P.ndim == 3 and N.ndim == 3):
                return (uid, pos, neg, {
                    "user_vec": torch.from_numpy(U.copy()).float(),
                    "pos_vec": torch.from_numpy(P.copy()).float(),
                    "neg_vec": torch.from_numpy(N.copy()).float(),
                })

            print(f"[ReviewData] ERROR: invalid NARRE/DeepCoNN embedding in BPR: "
                  f"pooling={pooling}, U.ndim={U.ndim}, P.ndim={P.ndim}, N.ndim={N.ndim}")
            return sample


        # TOKENIZED PATH (non usato qui)

        if self.return_tokenized:
            print("[ReviewData] Error: tokenization path non implementato.")
            return sample

        return sample

    def __len__(self):
        if self.setup == "BPR" and self.mode != "Train":
            return self.len
        return len(self.x)
