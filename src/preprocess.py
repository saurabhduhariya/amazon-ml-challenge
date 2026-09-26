import re
import pandas as pd
from unidecode import unidecode
from src.config import _SUFFIX_PATTERNS, _PREFIX_PATTERNS, ADDRESS_ABBR

def normalize_text(text: str) -> str:
    """Normalize raw text: unicode removal, lowercase, symbol expansion."""
    if pd.isna(text) or not text:
        return ''
    text = unidecode(str(text)).lower().strip()
    text = text.replace('&', ' and ').replace('@', ' at ').replace('#', ' number ')
    text = re.sub(r'\bm/s\b|\bm\s*s\b', 'ms', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_name_components(name: str):
    """Separate business prefix, core company name, and legal suffix."""
    if not name:
        return '', '', ''
    prefix = ''
    for pat, p in _PREFIX_PATTERNS:
        if pat.search(name):
            prefix = p
            name = pat.sub('', name).strip()
            break
    suffixes = []
    for pat, s in _SUFFIX_PATTERNS:
        if pat.search(name):
            suffixes.append(s)
            name = pat.sub('', name).strip()
    core_name = re.sub(r'\s+', ' ', name).strip()
    return core_name, prefix, ' '.join(suffixes)

def expand_address_abbr(address: str) -> str:
    """Standardize street/avenue abbreviations."""
    return ' '.join(ADDRESS_ABBR.get(t, t) for t in address.split())

def extract_numbers(text: str) -> str:
    """Extract numeric sequences for street/PIN verification."""
    return ' '.join(re.findall(r'\b\d+\b', text))

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end normalization of a business entity DataFrame."""
    df = df.copy()
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    
    df['clean_name'] = df['business_name'].apply(normalize_text)
    components = [extract_name_components(n) for n in df['clean_name']]
    df['clean_name_core'] = [c[0] for c in components]
    df['name_prefix']     = [c[1] for c in components]
    df['legal_suffix']    = [c[2] for c in components]
    df['clean_name_nosuffix'] = df['clean_name_core']
    
    df['clean_address'] = df['business_address'].apply(normalize_text).apply(expand_address_abbr)
    df['addr_numbers']  = df['clean_address'].apply(extract_numbers)
    df['combined_text'] = df['clean_name_core'] + ' | ' + df['clean_address']
    df['name_tokens']   = df['clean_name_core'].apply(lambda x: set(x.split()) if x else set())
    return df
