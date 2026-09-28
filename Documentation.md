# Amazon ML Challenge 2026: Business Entity Resolution Methodology

**Author:** Saurabh Duhariya  
**Task:** Large-Scale Multi-Source Business Entity Linkage & Deduplication  
**Metric:** Macro-averaged $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary

We developed an end-to-end, high-precision entity resolution pipeline to match and deduplicate **2,206,821 Source 1 business records** against a heterogeneous target pool of **10,320,219 records** (Sources 2 and 3). Addressing an intractable comparison space of over **22 trillion candidate pairs**, our architecture deploys a two-stage paradigm:
1. **Multi-Channel Candidate Blocking:** Combining sparse character n-gram TF-IDF and inverted token indexing with country partitioning, reducing search space by **99.98%** while achieving a **95.97% recall ceiling**.
2. **Gradient-Boosted Pairwise Classification:** Mining hard negatives (5:1 ratio) from blocking pools and training an ensembled **GPU XGBoost + LightGBM** classifier across 18+ fuzzy syntactic, legal suffix, and address-number features.
3. **Threshold Calibration:** Decision threshold grid search explicitly calibrated for **Macro $F_{0.5}$**, penalizing false positive merges $2\times$ over missed links and accurately identifying singletons (5.6% true zero-match entities).

---

## 2. Methodology & Problem Analysis

### 2.1 Problem Analysis & Data Insights
Exploratory Data Analysis (EDA) revealed several domain-specific challenges across the multi-source dataset:
* **Legal Suffix Inflation:** Suffixes like *"Pvt Ltd"*, *"LLC"*, *"Corporation"*, and their phonetic transliterations (*"praaivett limittedd"*, *"piraivet"*) heavily inflate standard string similarity metrics between unrelated companies.
* **Address Format Heterogeneity:** Abbreviations (*"rd"* vs *"road"*, *"st"* vs *"street"*, *"ste"* vs *"suite"*) and missing address fields create significant noise.
* **Country Partitions:** Legitimate cross-source matches share the same geographic jurisdiction (predominantly US and India).
* **Singleton Penalty:** 5.6% of entities have zero true matches in Sources 2/3. Under the Macro $F_{0.5}$ metric, correctly predicting an empty set yields 1.0, while an erroneous match drops the score to 0.0.

### 2.2 Solution Strategy
```
Raw Entities (S1: 2.2M, S2+S3: 10.3M)
                  │
                  ▼
   Domain Preprocessing & Suffix Extraction
                  │
                  ▼
      Multi-Channel Blocking (Recall: 95.97%)
    ┌─────────────┴─────────────┐
    ▼                           ▼
Sparse TF-IDF (Top-25)   Inverted Token Index
    └─────────────┬─────────────┘
                  ▼
        Candidate Union Merging
                  │
                  ▼
   18+ Feature Extraction & Hard Negative Mining
                  │
                  ▼
       XGBoost + LightGBM Ensemble
                  │
                  ▼
    Macro F_0.5 Threshold Calibration
```

---

## 3. Candidate Generation (Blocking)

Comparing all $2.2M \times 10.3M$ pairs requires $>2.2 \times 10^{13}$ computations. We developed a two-channel candidate generation strategy:

| Blocking Channel | Technical Implementation | Recall Ceiling | Avg Cands / Entity |
| :--- | :--- | :--- | :--- |
| **Channel 1: Sparse TF-IDF** | Character 3–5 n-grams, sublinear TF scaling, max 120,000 features, cosine similarity $\ge 0.08$, Top-25 candidates. Batch-evaluated without dense matrix expansion. | **94.38%** | 25.0 |
| **Channel 2: Inverted Index** | Inverted token index across 1.42M unique tokens; candidate generation requiring $\ge 2$ shared tokens within the same country partition. | **66.99%** | 1,987.5 |
| **Fused (Union Merge)** | Taking the set union of both channels. | **95.97%** | **2,005.9** |

**Key Benefit:** Prunes the candidate search space by **99.98%**, allowing computationally intensive pairwise feature extractors to run efficiently.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Set (18+ Features)
1. **Core Name Syntactic Similarities:**
   - RapidFuzz Ratio, Token Sort Ratio, and Token Set Ratio on core business name (stripped of legal suffixes).
   - Jaro-Winkler distance on core name and full unstripped name.
   - Token Jaccard similarity and exact substring containment indicators.
2. **Legal Suffix & Prefix Compatibility:**
   - Legal suffix exact match (1.0), partial missing (0.5), or conflicting suffix penalty (0.0).
   - Business prefix compatibility (`"M/S"`, `"The"`, `"Cabinet"`).
3. **Address & Geographic Signals:**
   - Levenshtein, Jaro-Winkler, and Token Sort ratios on standardized addresses.
   - Address numeric sequence overlap (extracting PIN codes and street numbers).
   - Binary indicators for country match and mutual address availability.

### 4.2 Hard Negative Mining
Random negative sampling produces trivial non-matches (e.g. comparing a bakery in India to a tech firm in New York), leading to weak gradient signals. We mined **Hard Negatives directly from the blocking candidates** (ratio 5:1), forcing the classifier to learn fine-grained discrimination between companies sharing tokens or addresses.

### 4.3 Model Architecture
- **XGBoost:** GPU-accelerated histogram algorithm (`tree_method='hist'`), optimizing `aucpr` with `scale_pos_weight` to address positive/negative pair imbalance.
- **LightGBM:** Fast leaf-wise tree growth optimizing `average_precision`.
- **Ensemble:** Calibrated soft probability blending ($0.5 \times P_{\text{XGB}} + 0.5 \times P_{\text{LGB}}$).

---

## 5. Threshold Optimization & Results

Because the evaluation metric is **Macro $F_{0.5}$** ($\beta = 0.5$), false merges (FP) are penalized twice as heavily as missed matches (FN):
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

We performed grid-search threshold calibration over out-of-fold validation splits:
- Standard default threshold (0.50) yields lower precision and risks severe singleton penalties.
- Elevating the threshold to the optimal range ($\approx 0.65 - 0.70$) suppresses ambiguous matches, maximizing the Macro $F_{0.5}$ score.

---

## 6. Error Analysis & Limitations

- **False Merges:** Occurred primarily when parent and subsidiary companies share near-identical names and registered headquarters addresses (e.g., *"Acme Trading"* vs *"Acme Logistics"* at the same corporate complex).
- **Missed Matches (False Negatives):** Occurred when entities exhibited extreme phonetics or transliteration mismatches coupled with missing addresses in both sources.
