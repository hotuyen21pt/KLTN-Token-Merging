PyABSA APC expects blocks of exactly 3 lines per example:
  Line 1: sentence with exactly one "$T$" marking the aspect placeholder
  Line 2: aspect term (same words as replaced by $T$ in plain text)
  Line 3: polarity — Positive | Negative | Neutral

Raw SemEval-style names like Restaurants_Train.xml.seg are NOT auto-detected.

Use filenames that contain BOTH keywords "train" AND "APC", and ".apc" extension
(e.g. train.APC.restaurants.apc). Same for test: "test" + "APC".

Run once after adding new .seg sources:
  python thesis_apc_baseline/dataset/ensure_pyabsa_names.py

Training scripts resolve thesis_apc_baseline/dataset automatically when train/test sources exist.
