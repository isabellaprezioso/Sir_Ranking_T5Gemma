# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.net = Net(opt)

        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea
            else:
                feature_dim = self.opt.id_emb_size

        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward_old(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        user_feature, item_feature = self.net(datas)
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)
        return output

    def forward_oldt5(self, datas):
        if isinstance(datas, dict) and 'user_input_ids' in datas:
            # --- Modalità T5-GEMMA ---
            user_input_ids = datas['user_input_ids']
            user_attention_mask = datas['user_attention_mask']
            item_input_ids = datas['item_input_ids']
            item_attention_mask = datas['item_attention_mask']

            # Ottieni embeddings dai T5 encoder interni del net (DeepCoNN)
            user_emb = self.net.get_t5_embeddings(user_input_ids, user_attention_mask)
            item_emb = self.net.get_t5_embeddings(item_input_ids, item_attention_mask)

            # Passa le embedding attraverso le CNN e i layer fully-connected
            user_feature = self.net._process_with_cnn(user_emb, self.net.user_cnn, self.net.user_fc_linear)
            item_feature = self.net._process_with_cnn(item_emb, self.net.item_cnn, self.net.item_fc_linear)

            # Fusion + Dropout + Prediction
            ui_feature = self.fusion_net(user_feature, item_feature)
            ui_feature = self.dropout(ui_feature)
            output = self.predict_net(ui_feature, None, None).squeeze(1)
            return output

        else:
            # --- Modalità classica ---
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
            user_feature, item_feature = self.net(datas)

            ui_feature = self.fusion_net(user_feature, item_feature)
            ui_feature = self.dropout(ui_feature)
            output = self.predict_net(ui_feature, uids, iids).squeeze(1)
            return output


    # Versione unificata: supporta PRECOMPUTED / T5 / classico
    def forward(self, datas):
        """
        Compatibile con:
        - PRECOMPUTED: la net può restituire direttamente lo score (Tensor) o un dict con 'scores'
        - T5/classico: la net restituisce (user_feature, item_feature)
        """
        out = self.net(datas)

        # 1) Score già pronto (Tensor): es. DeepCoNN/NARRE in modalità PRECOMPUTED
        if isinstance(out, torch.Tensor):
            # squeeze eventuale dimensione finale 1x
            if out.dim() == 2 and out.size(-1) == 1:
                return out.squeeze(1)
            return out

        # 2) Dict con 'scores'
        if isinstance(out, dict):
            if 'scores' in out:
                return out['scores']
            # oppure dict con feature esplicite
            if 'user_feature' in out and 'item_feature' in out:
                user_feature = out['user_feature']
                item_feature = out['item_feature']
            else:
                raise TypeError(f"Model net() returned dict without 'scores' or features keys: {list(out.keys())}")
        else:
            # 3) Coppia (user_feature, item_feature)
            try:
                user_feature, item_feature = out
            except Exception as e:
                raise TypeError(f"Model net() returned unsupported type: {type(out)}") from e

        # Fusion + predizione (comune)
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)

        # Passa uids/iids a predict_net solo se disponibili (modalità classica)
        uids = iids = None
        if not isinstance(datas, dict):
            try:
                _, _, uids, iids, *_ = datas
            except Exception:
                pass

        output = self.predict_net(ui_feature, uids, iids).squeeze(1)
        return output

    def load(self, path):

        self.load_state_dict(torch.load(path), strict=False)

    def save(self, epoch=None, name=None, opt=None):

        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name
