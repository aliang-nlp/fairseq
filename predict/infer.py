#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/2 上午11:07
# @Author  : Aliang

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="zho_Hans")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)
article = "yaoming是世界上最好的篮球运动员"
inputs = tokenizer(article, return_tensors="pt")
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30)
p = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(p)