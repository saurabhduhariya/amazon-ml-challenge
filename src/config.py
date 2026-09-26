import os
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset"
OUTPUT_DIR = BASE_DIR / "output"

# Blocking Hyperparameters
TFIDF_TOP_K = 25
TFIDF_MIN_SIM = 0.08
TFIDF_BATCH_SIZE = 100
RULE_MIN_TOKENS = 2

# Negative Sampling & Model Training
NEG_RATIO = 5
VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Legal Suffixes for Business Entity Disambiguation
LEGAL_SUFFIXES = [
    # English (ordered by specificity)
    'private limited', 'pvt ltd', 'pvt limited', 'limited liability partnership',
    'limited liability company', 'incorporated', 'corporation',
    'limited', 'inc', 'corp', 'llc', 'llp', 'lp', 'plc', 'ltd', 'pvt', 'co',
    'private', 'public',
    # Indic transliterated
    'praaivett limittedd', 'praivet limited', 'limittedd', 'praaivett', 'elelpii',
    'piraivet limitet', 'piraiveett limitett', 'limitet',
    # European / French
    'societe a responsabilite limitee', 'societe anonyme',
    'societe par actions simplifiee', 'societe civile immobiliere',
    'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'gie', 'snc', 'scp', 'gmbh', 'ag',
    # Generic corporate qualifiers
    'group', 'holding', 'holdings', 'enterprises', 'enterprise',
    'industries', 'industry', 'services', 'solutions', 'technologies',
    'technology', 'international', 'intl'
]

BUSINESS_PREFIXES = ['ms', 'the', 'ets', 'ste', 'cie', 'cabinet', 'groupe', 'dr', 'shri', 'messrs']

ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'sq': 'square', 'hwy': 'highway', 'pkwy': 'parkway',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    'bldg': 'building', 'dept': 'department', 'mt': 'mount',
    'ft': 'fort', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    'bd': 'boulevard', 'av': 'avenue', 'imp': 'impasse',
}

_SUFFIX_PATTERNS = [(re.compile(r'\b' + re.escape(s) + r'\b'), s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)]
_PREFIX_PATTERNS = [(re.compile(r'^\b' + re.escape(p) + r'\b\s*'), p) for p in sorted(BUSINESS_PREFIXES, key=len, reverse=True)]
