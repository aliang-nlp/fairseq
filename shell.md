# Dense
python examples/nllb/modeling/train/train_script.py cfg=flores_200_full cfg.fairseq_root=$(pwd) cfg.output_dir=$OUTPUT_DIR cfg.dataset.eval_lang_pairs_file=examples/nllb/modeling/scripts/flores200/eval_lang_pairs_eng400_noneng20.txt cfg.dropout=0.7
