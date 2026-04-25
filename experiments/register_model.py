# -*- coding: utf-8 -*-
"""Attach FAST_LCF_BERT_TOME to PyABSA APCModelList without editing installed pyabsa."""

from thesis_apc_baseline.models.fast_lcf_bert_tome import FAST_LCF_BERT_TOME


def register_fast_lcf_bert_tome():
    from pyabsa.tasks.AspectPolarityClassification.models import APCModelList

    APCModelList.FAST_LCF_BERT_TOME = FAST_LCF_BERT_TOME
    return APCModelList.FAST_LCF_BERT_TOME
