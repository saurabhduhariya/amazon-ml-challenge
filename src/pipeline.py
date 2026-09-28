import os
import gc
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from src.config import (
    DATA_DIR, OUTPUT_DIR, TFIDF_TOP_K, TFIDF_MIN_SIM,
    RULE_MIN_TOKENS, NEG_RATIO, VAL_SPLIT_RATIO, RANDOM_SEED
)
from src.preprocess import preprocess_dataframe
from src.blocking import tfidf_blocking, rule_based_blocking, merge_blocking_channels, evaluate_blocking_recall
from src.dataset import parse_ground_truth, build_training_pairs
from src.features import compute_pair_features
from src.models import train_xgboost, train_lightgbm, predict_ensemble
from src.metrics import compute_f05_macro, optimize_threshold

def run_pipeline(train_mode=True, sample_size=30000):
    print("=" * 70)
    print("🚀 AMAZON ML CHALLENGE: SCALABLE BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Training Data
    train_dir = DATA_DIR / "train"
    print(f"\n[1/5] Loading datasets from {train_dir}...")
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t")
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t")
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t")
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")
    print(f"  Loaded: S1 ({len(s1):,}), S2 ({len(s2):,}), S3 ({len(s3):,}), GT ({len(gt):,})")

    # 2. Preprocess
    print("\n[2/5] Preprocessing entity names, suffixes, and addresses...")
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    s1.set_index('entity_id', drop=False, inplace=True)
    s23.set_index('entity_id', drop=False, inplace=True)

    # 3. Train/Validation Split
    print("\n[3/5] Splitting validation entities & parsing ground truth...")
    gt_dict = parse_ground_truth(gt)
    all_s1_ids = list(s1['entity_id'].values)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * VAL_SPLIT_RATIO)
    val_ids = set(all_s1_ids[:n_val])
    train_ids = set(all_s1_ids[n_val:])

    train_gt_dict = {k: v for k, v in gt_dict.items() if k in train_ids}
    val_gt_dict = {k: v for k, v in gt_dict.items() if k in val_ids}

    # Sample for training efficiency if specified
    if sample_size and sample_size < len(train_ids):
        train_sample_ids = list(train_ids)[:sample_size]
    else:
        train_sample_ids = list(train_ids)

    val_sample_ids = list(val_ids)[:min(5000, len(val_ids))]

    # 4. Multi-Channel Candidate Blocking
    print("\n[4/5] Executing Multi-Channel Candidate Blocking (Sparse TF-IDF + Token Overlap)...")
    s1_train_sample = s1.loc[train_sample_ids]
    s1_val_sample   = s1.loc[val_sample_ids]

    cands_tfidf = tfidf_blocking(s1_val_sample, s23, top_k=TFIDF_TOP_K, min_sim=TFIDF_MIN_SIM)
    cands_rule  = rule_based_blocking(s1_val_sample, s23, min_shared_tokens=RULE_MIN_TOKENS)
    cands_val_merged = merge_blocking_channels(cands_tfidf, cands_rule)

    blocking_stats = evaluate_blocking_recall(cands_val_merged, val_gt_dict)
    print(f"  Validation Blocking Recall Ceiling: {blocking_stats['recall']*100:.2f}% | Avg Candidates/Entity: {blocking_stats['avg_cands']:.1f}")

    # 5. Feature Engineering & Model Training
    print("\n[5/5] Extracting 18+ Pairwise Similarity Features & Mining Hard Negatives...")
    cands_train_tfidf = tfidf_blocking(s1_train_sample, s23, top_k=TFIDF_TOP_K, min_sim=TFIDF_MIN_SIM)
    cands_train_rule  = rule_based_blocking(s1_train_sample, s23, min_shared_tokens=RULE_MIN_TOKENS)
    cands_train_merged = merge_blocking_channels(cands_train_tfidf, cands_train_rule)

    X_train, y_train, pairs_train = build_training_pairs(train_sample_ids, train_gt_dict, cands_train_merged, s1, s23, neg_ratio=NEG_RATIO)
    X_val, y_val, pairs_val       = build_training_pairs(val_sample_ids, val_gt_dict, cands_val_merged, s1, s23, neg_ratio=NEG_RATIO)

    print(f"  Training Feature Matrix: {X_train.shape} | Positive Class Ratio: {(y_train == 1).mean():.3f}")

    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

    val_probs = predict_ensemble(xgb_model, lgb_model, X_val)
    best_thresh, best_f05, results = optimize_threshold(val_probs, pairs_val, val_gt_dict)

    print("\n" + "=" * 70)
    print(f"🏆 EVALUATION COMPLETE: Optimal Macro F_0.5 = {best_f05:.4f} at Decision Threshold {best_thresh:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Business Entity Resolution")
    parser.add_argument("--sample", type=int, default=30000, help="Number of training samples")
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample)
