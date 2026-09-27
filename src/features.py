import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def jaccard_tokens(a: str, b: str) -> float:
    """Calculate token Jaccard similarity."""
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def numbers_match(a: str, b: str) -> float:
    """Calculate overlap of street and pin numbers."""
    na, nb = set(re.findall(r'\b\d+\b', a)), set(re.findall(r'\b\d+\b', b))
    return 0.5 if not (na and nb) else (len(na & nb) / max(len(na), len(nb)))

def compute_pair_features(s1_row, cand_row) -> dict:
    """
    Extract 18+ pairwise syntactic, fuzzy, and structural similarity features.
    """
    n1, n2   = s1_row['clean_name_core'], cand_row['clean_name_core']
    nf1, nf2 = s1_row['clean_name'], cand_row['clean_name']
    a1, a2   = s1_row['clean_address'], cand_row['clean_address']
    
    f = {}
    
    # 1. Core Name Similarity Features
    f['name_lev']        = fuzz.ratio(n1, n2) / 100.0
    f['name_jw']         = JaroWinkler.similarity(n1, n2)
    f['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    f['name_token_set']  = fuzz.token_set_ratio(n1, n2) / 100.0
    f['name_jaccard']    = jaccard_tokens(n1, n2)
    f['name_exact']      = float(n1 == n2)
    f['name_contain']    = float(n1 in n2 or n2 in n1) if (n1 and n2) else 0.0
    f['name_full_jw']    = JaroWinkler.similarity(nf1, nf2)
    
    # 2. Suffix & Prefix Consistency
    suf1, suf2 = s1_row['legal_suffix'], cand_row['legal_suffix']
    if not suf1 or not suf2:
        f['suffix_match'] = 0.5
    elif suf1 == suf2:
        f['suffix_match'] = 1.0
    else:
        f['suffix_match'] = 0.0
        
    pre1, pre2 = s1_row['name_prefix'], cand_row['name_prefix']
    f['prefix_match'] = 1.0 if (not pre1 or not pre2 or pre1 == pre2) else 0.0
    
    # 3. Address Features
    if a1 and a2:
        f['addr_lev']        = fuzz.ratio(a1, a2) / 100.0
        f['addr_jw']         = JaroWinkler.similarity(a1, a2)
        f['addr_token_sort'] = fuzz.token_sort_ratio(a1, a2) / 100.0
        f['addr_jaccard']    = jaccard_tokens(a1, a2)
        f['addr_numbers']    = numbers_match(a1, a2)
    else:
        for k in ['addr_lev', 'addr_jw', 'addr_token_sort', 'addr_jaccard', 'addr_numbers']:
            f[k] = 0.0
            
    # 4. Partition & Metadata Indicators
    f['country_match'] = float(s1_row['country'] == cand_row['country'])
    f['has_addr_both'] = float(bool(a1) and bool(a2))
    
    return f
