cd ..

# Chạy full baseline và tome: 1.0 là 100% dataset
thesis_apc_baseline/experiments/compare_baseline_tome.py --data-fraction 1.0

# inference
python thesis_apc_baseline/experiments/infer_one.py --checkpoint thesis_apc_baseline/checkpoints/baseline_fast_lcf_bert --sentence "The `$T$` was cold ." --aspect "pizza"
