import json, hashlib
from typing import Any, Dict

def _array_to_bytes(arr: mx.array) -> bytes:
    """Serialize an MX array to a deterministic byte string (for hashing)."""
    return arr.tobytes()

def memory_to_dict(mem: UnifiedMemory) -> Dict[str, Any]:
    """Flatten the whole memory object into a JSON‑serialisable dict."""
    dump = mem.dump()                     # the dict we already expose
    # Convert every numpy array in the dump to a byte string so we can re‑hydrate later
    for k, v in dump.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, dict) and "sha256" in subv:
                    # already a hex string – keep it as‑is
                    continue
                else:
                    # store raw bytes for later reconstruction
                    dump[k][subk] = subv
        elif isinstance(v, list):
            # list of arrays – we will store their hashes instead of the whole tensor
            dump[k] = [hashlib.sha256(_array_to_bytes(a)).hexdigest() for a in v]
    return dump

def load_memory_from_dict(d: Dict[str, Any], dim: int = 256) -> UnifiedMemory:
    """Re‑create a UnifiedMemory from a dict produced by `memory_to_dict`. """
    # Re‑create a fresh instance with the same config
    mem = UnifiedMemory(
        feature_dim=dim,
        short_max=d.get("short_max", 32),
        long_max=d.get("long_max", 128),
        ema_alpha=d.get("ema_alpha", 0.1),
        consolidate_thresh=d.get("consolidate_thresh", 0.8),
        age_beta=d.get("age_beta", 0.001),
        novelty_thresh=d.get("novelty_thresh", 0.6),
        goal_dim=d.get("goal_dim", 256),
    )
    # Restore the raw buffers
    mem.short_buf = deque(maxlen=mem.short_max)
    mem.short_energy = [0.0] * mem.short_max
    mem.short_ts = [0] * mem.short_max
    mem.short_goal = [mx.zeros(mem.goal_dim, dtype=mx.float32) for _ in range(mem.short_max)]
    mem.long_entries = []

    # Helper to decode a stored hash back into an mx.array
    def decode_hash(h: str) -> mx.array:
        # In a real system you would keep a mapping from hash → array.
        # For this minimal demo we’ll just reconstruct a dummy random array.
        # Replace this with your own lookup table if you want true reproducibility.
        return mx.random.normal((mem.feature_dim,), dtype=mx.float32)

    # Walk through the dumped dict and rebuild the structures
    for entry in d["long_entries"]:
        vec_shape = entry["vec_shape"]
        # Re‑create a vector of the proper shape (here we just use random normal)
        vec = mx.random.normal(vec_shape, dtype=mx.float32)
        weight = entry["weight"]
        mem.long_entries.append((vec, weight))

    # Short‑term buffers are already filled with zeros; you can also
    # load the saved energies / timestamps if you stored them.
    # (For brevity we keep them empty – they will be repopulated on the next `add`.)

    return mem
