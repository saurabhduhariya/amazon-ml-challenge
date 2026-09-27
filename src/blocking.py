import gc
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

def tfidf_blocking(s1_df, s23_df, top_k=25, min_sim=0.08, batch_size=100):
    """
    Memory-efficient Sparse Character N-Gram TF-IDF Blocking.
    Processes partitions by country and avoids dense memory spikes.
    """
    candidates = {eid: set() for eid in s1_df['entity_id']}
    countries = s1_df['country'].dropna().unique()

    for c in countries:
        s1_c = s1_df[s1_df['country'] == c]
        s23_c = s23_df[s23_df['country'] == c]
        if len(s1_c) == 0 or len(s23_c) == 0:
            continue

        print(f"  [Country: {c}] S1: {len(s1_c):,} | Target Pool: {len(s23_c):,}")
        sample_size = min(300_000, len(s23_c))
        sample_texts = s23_c['combined_text'].sample(sample_size, random_state=42)

        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32
        )
        vectorizer.fit(sample_texts)

        s23_matrix = vectorizer.transform(s23_c['combined_text'])
        s23_ids = s23_c['entity_id'].values

        n_batches = int(np.ceil(len(s1_c) / batch_size))
        for b in tqdm(range(n_batches), desc=f"    Sparse TF-IDF ({c})"):
            batch = s1_c.iloc[b * batch_size : (b + 1) * batch_size]
            s1_matrix = vectorizer.transform(batch['combined_text'])
            sim = s1_matrix.dot(s23_matrix.T)

            for i, s1_id in enumerate(batch['entity_id']):
                row = sim.getrow(i)
                if row.nnz == 0:
                    continue
                col_indices = row.indices
                data_values = row.data
                mask = data_values >= min_sim
                if not np.any(mask):
                    continue
                valid_cols = col_indices[mask]
                valid_vals = data_values[mask]
                if len(valid_cols) > top_k:
                    top_idx = np.argpartition(valid_vals, -top_k)[-top_k:]
                    valid_cols = valid_cols[top_idx]
                candidates[s1_id].update(s23_ids[valid_cols])
            del sim, s1_matrix
            gc.collect()

        del s23_matrix, vectorizer
        gc.collect()

    return candidates

def rule_based_blocking(s1_df, s23_df, min_shared_tokens=2):
    """
    Inverted-index Token Overlap Blocking with Country Filtering.
    """
    print("  Building inverted index...")
    token_index = defaultdict(list)
    s23_countries = dict(zip(s23_df['entity_id'], s23_df['country']))

    for eid, tokens in zip(s23_df['entity_id'], s23_df['name_tokens']):
        for tok in tokens:
            if len(tok) >= 3:
                token_index[tok].append(eid)

    candidates = {eid: set() for eid in s1_df['entity_id']}
    s1_countries = dict(zip(s1_df['entity_id'], s1_df['country']))

    for eid, tokens in tqdm(zip(s1_df['entity_id'], s1_df['name_tokens']), total=len(s1_df), desc="  Rule blocking"):
        s1_country = s1_countries.get(eid, '')
        counts = defaultdict(int)
        for tok in tokens:
            if len(tok) >= 3 and tok in token_index:
                for cand_id in token_index[tok]:
                    counts[cand_id] += 1

        cands = {
            cand_id for cand_id, cnt in counts.items()
            if cnt >= min_shared_tokens and s23_countries.get(cand_id, '') == s1_country
        }
        candidates[eid] = cands

    return candidates

def merge_blocking_channels(*channels):
    """Compute union of candidates across multiple blocking channels."""
    all_ids = set().union(*[ch.keys() for ch in channels])
    merged = {s1_id: set().union(*[ch.get(s1_id, set()) for ch in channels]) for s1_id in all_ids}
    return merged

def evaluate_blocking_recall(candidates, ground_truth):
    """Evaluate blocking candidate recall ceiling and average candidate count."""
    total_true, found = 0, 0
    for s1_id, true_matches in ground_truth.items():
        if s1_id not in candidates:
            continue
        cands = candidates[s1_id]
        for m_id in true_matches:
            total_true += 1
            if m_id in cands:
                found += 1
    recall = found / total_true if total_true > 0 else 0.0
    avg_cands = sum(len(v) for v in candidates.values()) / max(1, len(candidates))
    return {'recall': recall, 'found': found, 'total_true': total_true, 'avg_cands': avg_cands}
