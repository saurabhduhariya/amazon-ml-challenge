from collections import defaultdict
import numpy as np

def compute_f05_macro(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute Macro-Averaged F_0.5 score across all Source 1 entities.
    Penalizes false positive merges 2x more than missed matches.
    Correctly accounts for singletons (entities with 0 matches).
    """
    f05_scores = []
    precisions = []
    recalls = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if isinstance(pred_set, list):
            pred_set = set(pred_set)
        if isinstance(true_set, list):
            true_set = set(true_set)

        # Case 1: True singleton
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            continue

        # Case 2: Predicted empty on entity with true matches
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # Case 3: Both non-empty
        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)

        if (0.25 * p + r) > 0:
            f05 = (1.25 * p * r) / (0.25 * p + r)
        else:
            f05 = 0.0

        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)

    return {
        'f05': float(np.mean(f05_scores)) if f05_scores else 0.0,
        'precision': float(np.mean(precisions)) if precisions else 0.0,
        'recall': float(np.mean(recalls)) if recalls else 0.0,
        'n_entities': len(ground_truth)
    }

def optimize_threshold(probs, pairs, gt_dict, start=0.2, end=0.9, step=0.05):
    """Grid search for decision threshold optimizing Macro F0.5."""
    best_f05 = -1.0
    best_thresh = 0.5
    results = []

    for t in np.arange(start, end, step):
        preds = defaultdict(set)
        for idx, (s1_id, cand_id) in enumerate(pairs):
            if probs[idx] >= t:
                preds[s1_id].add(cand_id)
        
        metrics = compute_f05_macro(preds, gt_dict)
        results.append((t, metrics))
        if metrics['f05'] > best_f05:
            best_f05 = metrics['f05']
            best_thresh = t

    return best_thresh, best_f05, results
