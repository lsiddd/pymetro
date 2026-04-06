# ./systems/ai/ga_cache.py
import os
import pickle
import hashlib
from typing import List, Any

CACHE_PATH = '/tmp/pymetro_ga_cache.pkl'

def make_signature(stations: List[Any]) -> str:
    """Stable fingerprint of the current station set (ids + types)."""
    key = tuple(sorted((s.id, s.type) for s in stations))
    return hashlib.md5(str(key).encode()).hexdigest()

def save_cache(chromosomes: list, sig: str) -> None:
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({'sig': sig, 'chromosomes': chromosomes[:10]}, f)
        print(f"[GA Cache] {len(chromosomes[:10])} cromossomos salvos.")
    except Exception as e:
        print(f"[GA Cache] Falha ao salvar: {e}")

def load_cache(sig: str) -> list:
    if not os.path.exists(CACHE_PATH):
        return []
    try:
        with open(CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        if data.get('sig') == sig:
            return data.get('chromosomes', [])
    except Exception as e:
        print(f"[GA Cache] Falha ao carregar: {e}")
    return []
