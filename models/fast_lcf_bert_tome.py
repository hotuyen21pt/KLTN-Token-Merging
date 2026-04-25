# -*- coding: utf-8 -*-
"""FAST_LCF_BERT + optional Token Merging (ToMe-style) after the BERT backbone."""

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.sa_encoder import Encoder

from thesis_apc_baseline.token_merging.tome_1d import ToMeSequenceMerger


class FAST_LCF_BERT_TOME(nn.Module):
    """Same inputs/outputs as ``FAST_LCF_BERT``; merges tokens after ``bert4global``.

    Extra config attributes (optional, set before training):

    - ``use_tome`` (bool): enable merging.
    - ``tome_merge_steps`` (int): bipartite merge rounds per sample (default 2).
    - ``pad_token_id`` (int): mask padding; defaults to tokenizer.pad_token_id if missing.
    """

    inputs = ["text_indices", "text_raw_bert_indices", "lcf_vec"]

    def __init__(self, bert, config):
        super().__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.bert_SA = Encoder(bert.config, config)
        self.linear2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.bert_SA_ = Encoder(bert.config, config)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.embed_dim, config.output_dim)

        steps = int(getattr(config, "tome_merge_steps", 2))
        self.tome = ToMeSequenceMerger(
            num_merge_steps=steps,
            protect_cls=True,
            protect_sep=True,
        )
        self._use_tome = bool(getattr(config, "use_tome", False))

    def forward(self, inputs):
        if self.config.use_bert_spc:
            text_indices = inputs["text_indices"]
        else:
            text_indices = inputs["text_raw_bert_indices"]

        pad_id = int(getattr(self.config, "pad_token_id", 0))
        attn_mask = (text_indices != pad_id).float()

        global_context_features = self.bert4global(text_indices)["last_hidden_state"]

        lcf_vec = inputs["lcf_vec"].float()

        if self._use_tome:
            trace, merged_h, merged_lcf = self.tome.forward_with_trace(
                global_context_features,
                lcf_vec,
                attn_mask,
            )
            _ = trace  # optional: inspect during debugging
            global_context_features = merged_h
            lcf_matrix = merged_lcf.unsqueeze(2)
        else:
            lcf_matrix = lcf_vec.unsqueeze(2)

        # LCF layer (same as FAST_LCF_BERT)
        lcf_features = torch.mul(global_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        cat_features = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)
        cat_features = self.bert_SA_(cat_features)
        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)
        return {"logits": dense_out, "hidden_state": pooled_out}
