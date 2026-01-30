# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCoNN(nn.Module):
    """
    DeepCoNN unificato:

    - use_precomputed = False:
        word ids -> nn.Embedding -> CNN -> maxpool -> FC (standard DeepCoNN)

    - use_precomputed = True:
        (user_embs, item_embs) con shape (B, L, D) -> mean pool -> FC
        (NO CNN, NO nn.Embedding)
    """

    def __init__(self, opt, uori="user"):
        super().__init__()
        self.opt = opt
        self.num_fea = 1

        self.use_precomputed = getattr(opt, "use_precomputed", False)

        if not self.use_precomputed:
            # ===== STANDARD DEEPCONN =====
            self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
            self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)

            cnn_input_dim = opt.word_dim
            self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, cnn_input_dim))
            self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, cnn_input_dim))

            self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
            self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)

        else:
            # ===== PRECOMPUTED (NO CNN) =====
            self.user_word_embs = None
            self.item_word_embs = None
            self.user_cnn = None
            self.item_cnn = None

            vec_dim = opt.precomputed_dim  # es. 2304
            self.user_fc_linear = nn.Linear(vec_dim, opt.fc_dim)
            self.item_fc_linear = nn.Linear(vec_dim, opt.fc_dim)

        self.dropout = nn.Dropout(getattr(opt, "drop_out", 0.0))
        self.reset_para()

    # -----------------------------
    # STANDARD DeepCoNN: IDs -> Emb -> CNN -> MaxPool -> FC
    # -----------------------------
    def forward_classic(self, datas):
        if self.use_precomputed:
            raise RuntimeError("forward_classic chiamato ma use_precomputed=True")

        # compatibile col tuo datas unpacking
        _, _, uids, iids, _, _, user_doc, item_doc = datas
        device = next(self.parameters()).device

        user_doc = user_doc.to(device)
        item_doc = item_doc.to(device)

        # (B, L) -> (B, L, D)
        user_doc = self.user_word_embs(user_doc)
        item_doc = self.item_word_embs(item_doc)

        # Conv2d input: (B, 1, L, D)
        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # (B, F, L')
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)

        # MaxPool su L'
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)  # (B, F)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)

        u_fea = self.dropout(self.user_fc_linear(u_fea))  # (B, fc_dim)
        i_fea = self.dropout(self.item_fc_linear(i_fea))

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    # -----------------------------
    # PRECOMPUTED: (B,L,D) -> MeanPool -> FC (NO CNN)
    # -----------------------------
    def forward_precomputed(self, input_tuple):
        if not self.use_precomputed:
            raise RuntimeError("forward_precomputed chiamato ma use_precomputed=False")

        user_doc_embs, item_doc_embs = input_tuple  # (B, L, D)
        device = next(self.parameters()).device

        user_doc_embs = user_doc_embs.to(device)
        item_doc_embs = item_doc_embs.to(device)

        # Mean pooling su L -> (B, D)
        user_vec = user_doc_embs.mean(dim=1)
        item_vec = item_doc_embs.mean(dim=1)

        u_fea = self.dropout(self.user_fc_linear(user_vec))  # (B, fc_dim)
        i_fea = self.dropout(self.item_fc_linear(item_vec))  # (B, fc_dim)

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    # -----------------------------
    # FORWARD unico
    # -----------------------------
    def forward(self, x):
        if self.use_precomputed:
            # x = (user_embs, item_embs)
            return self.forward_precomputed(x)
        else:
            # x = datas (deepconn standard)
            return self.forward_classic(x)

    # -----------------------------
    # INIT pesi
    # -----------------------------
    def reset_para(self):
        if not self.use_precomputed:
            # CNN init
            for cnn in [self.user_cnn, self.item_cnn]:
                nn.init.xavier_normal_(cnn.weight)
                nn.init.constant_(cnn.bias, 0.1)

            # FC init
            for fc in [self.user_fc_linear, self.item_fc_linear]:
                nn.init.uniform_(fc.weight, -0.1, 0.1)
                nn.init.constant_(fc.bias, 0.1)

            # Word2Vec optional
            if getattr(self.opt, "use_word_embedding", False):
                w2v_path = self.opt.w2v_path + str(self.opt.word_dim) + ".npy"
                if os.path.exists(w2v_path):
                    w2v = torch.from_numpy(np.load(w2v_path))
                    if getattr(self.opt, "use_gpu", False):
                        self.user_word_embs.weight.data.copy_(w2v.cuda())
                        self.item_word_embs.weight.data.copy_(w2v.cuda())
                    else:
                        self.user_word_embs.weight.data.copy_(w2v)
                        self.item_word_embs.weight.data.copy_(w2v)
                    print(f"Caricati Word2Vec da {w2v_path}")
                else:
                    print(f"Word2Vec non trovato in {w2v_path}. Init casuale.")
                    nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
                    nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)
            else:
                nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
                nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

        else:
            # Precomputed: init solo FC
            for fc in [self.user_fc_linear, self.item_fc_linear]:
                nn.init.uniform_(fc.weight, -0.1, 0.1)
                nn.init.constant_(fc.bias, 0.1)

            print("DeepCoNN in modalità PRECOMPUTED: no Embedding layer, no CNN, mean-pooling + FC.")

