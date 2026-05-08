cd ..

# Chạy full baseline và tome: 1.0 là 100% dataset
thesis_apc_baseline/experiments/compare_baseline_tome.py --data-fraction 1.0

# inference
python thesis_apc_baseline/experiments/infer_one.py --checkpoint thesis_apc_baseline/checkpoints/baseline_fast_lcf_bert --sentence "The `$T$` was cold ." --aspect "pizza"

#full pipeline
python thesis_apc_baseline/experiments/infer_clause_sentiment.py --checkpoint "thesis_apc_baseline/checkpoints/fast_lcf_bert_tome/fast_lcf_bert_tome_custom_dataset_acc_77.68_f1_62.83" --sentence "The pizza was great but the staff was rude and the restaurant was noisy." --show-intermediate --hf-token ""

#show step merge
python thesis_apc_baseline/experiments/infer_one.py --checkpoint "thesis_apc_baseline/checkpoints/fast_lcf_bert_tome/fast_lcf_bert_tome_custom_dataset_acc_77.68_f1_62.83" --sentence "The $T$ was great but the staff was rude and the restaurant was noisy." --aspect "pizza" --show-steps
