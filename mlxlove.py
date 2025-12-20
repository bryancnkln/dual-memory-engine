# ------------------------------------------------------------
# 0️⃣  Imports & helpers (tokenizer / detokenizer stubs)
# ------------------------------------------------------------
import mlx.core as mx
from collections import deque
import time

# ---- Dummy tokenizer (replace with your real tokenizer) ----
def tokenize(s: str):
    # Very naive split on whitespace → list of ints (just for demo)
    return list(range(len(s)))   # each char → unique id

def detokenize(ids):
    return "".join(chr(i) for i in ids)

# ------------------------------------------------------------
# 1️⃣  Load a model (any MLX‑compatible LLM)
# ------------------------------------------------------------
# Example with Mamba (replace with your own model path)
try:
    import mamba
    model = mamba.load("nemotron-30b-a3b").to("mlx")
except Exception as e:
    raise RuntimeError("Make sure `mamba` is installed and the model path is correct.") from e

# ------------------------------------------------------------
# 2️⃣  Short‑Term Memory (with EMA & goal placeholder)
# ------------------------------------------------------------
class ShortTermMemory:
    def __init__(self,
                 feature_dim: int = 256,
                 max_history: int = 32,
                 ema_alpha: float = 0.1,
                 age_beta: float = 0.001,
                 goal_dim: int = 256):
        self.feature_dim   = feature_dim
        self.max_history   = max_history
        self.ema_alpha     = ema_alpha
        self.age_beta      = age_beta
        self.time_step     = 0
        self.buffer      = deque(maxlen=max_history)
        self.weights     = [0.0] * max_history
        self.timestamps  = [0] * max_history
        self.goal_vecs   = [mx.zeros(goal_dim, dtype=mx.float32) for _ in range(max_history)]

    def _cosine_similarity(self, a, b):
        return float(mx.dot(a / mx.norm(a), b / mx.norm(b)))

    def _find_nearest(self, query):
        if not self.buffer:
            return None, 0.0
        sims = [self._cosine_similarity(query, mem) for mem in self.buffer]
        best_i = int(mx.argmax(mx.array(sims)).item())
        return best_i, sims[best_i]

    def add(self, audio, text, energy=1.0):
        fused = mx.concatenate([audio.astype(mx.float32),
                                text.astype(mx.float32)], axis=0)

        idx, sim = self._find_nearest(fused)

        if idx is not None and sim > 0.6:          # lower thresh → more updates
            old_vec = self.buffer[idx]
            self.buffer[idx] = (1 - self.ema_alpha) * old_vec + self.ema_alpha * fused
            old_w   = self.weights[idx]
            self.weights[idx] = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.timestamps[idx] = self.time_step
            self.goal_vecs[idx] = self._current_global_goal()
        else:
            self.buffer.append(fused)
            self.weights.append(energy)
            self.timestamps.append(self.time_step)
            self.goal_vecs.append(self._current_global_goal())

            if len(self.weights) > self.max_history:
                self.weights.pop(0)
                self.timestamps.pop(0)
                self.goal_vecs.pop(0)

        self.time_step += 1

    def _current_global_goal(self):
        # In a full system this would read from a shared GoalUpdater.
        # For the demo we just return zeros (no bias yet).
        return mx.zeros(self.feature_dim, dtype=mx.float32)

# ------------------------------------------------------------
# 3️⃣  Long‑Term Memory (codebook)
# ------------------------------------------------------------
class LongTermMemory:
    def __init__(self,
                 feature_dim: int = 256,
                 max_entries: int = 128,
                 ema_alpha: float = 0.2,
                 novelty_thresh: float = 0.6,
                 min_age_before_prune: int = 30):
        self.feature_dim = feature_dim
        self.max_entries = max_entries
        self.ema_alpha   = ema_alpha
        self.novelty_thresh = novelty_thresh
        self.min_age_before_prune = min_age_before_prune
        self.entries = []                     # (vec, weight, timestamp)

    @staticmethod
    def _cosine_similarity(a, b):
        return float(mx.dot(a / mx.norm(a), b / mx.norm(b)))

    def _nearest(self, query):
        if not self.entries:
            return None, 0.0
        sims = [self._cosine_similarity(query, vec) for vec, _, _ in self.entries]
        best_i = int(mx.argmax(mx.array(sims)).item())
        return best_i, sims[best_i]

    def add(self, vec: mx.array, energy: float = 1.0):
        idx, sim = self._nearest(vec)
        if idx is not None and sim > self.novelty_thresh:
            old_vec, old_w, _ = self.entries[idx]
            new_vec = (1 - self.ema_alpha) * old_vec + self.ema_alpha * vec
            new_w   = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.entries[idx] = (new_vec, new_w, self.time_step)
        else:
            if len(self.entries) < self.max_entries:
                self.entries.append((vec, energy, self.time_step))
            else:
                min_idx = int(mx.argmin(mx.array([w for _, w, _ in self.entries])).item())
                self.entries[min_idx] = (vec, energy, self.time_step)

    def recall(self, query, top_k=5):
        sims = [(i, self._cosine_similarity(query, vec))
                for i, (vec, _, _) in enumerate(self.entries)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def prune(self, current_time):
        self.entries = [(v, w, t) for v, w, t in self.entries
                        if current_time - t < self.min_age_before_prune]

# ------------------------------------------------------------
# 4️⃣  Memory Observer (consolidation, pruning, counter‑factual)
# ------------------------------------------------------------
class MemoryObserver:
    def __init__(self,
                 short: ShortTermMemory,
                 long: LongTermMemory,
                 consolidate_thresh: float = 0.8,
                 novelty_check_every: int = 50,
                 age_beta: float = 0.001):
        self.short          = short
        self.long           = long
        self.consolidate_thresh = consolidate_thresh
        self.novelty_check_every = novelty_check_every
        self.age_beta       = age_beta
        self.time_since_last = 0

        self._goal_updater = GoalUpdater(dim=short.feature_dim, ema_alpha=0.03)

    # ----------------------------------------------------------
    def maybe_consolidate(self):
        if self.short.time_step == 0:
            return

        # Decayed priority across the whole buffer
        ages = mx.array([self.short.time_step - t for t in self.short.timestamps])
        priorities = mx.array(self.short.weights) * mx.exp(-self.age_beta * ages)
        best_idx = int(mx.argmax(priorities).item())
        best_weight = self.short.weights[best_idx]

        if best_weight >= self.consolidate_thresh:
            vec_to_store = self.short.buffer[best_idx]
            self.long.add(vec_to_store, energy=best_weight)

            # Update the *global* goal via EMA
            self._goal_updater.update(vec_to_store)

        self.time_since_last += 1

    def prune_long_term(self):
        self.long.prune(self.short.time_step)

    def counterfactual_score(self,
                             query_vec,
                             sigma=0.1) -> float:
        noise = mx.random.normal(shape=query_vec.shape, std=sigma)
        cand = query_vec + noise
        cand = cand / mx.norm(cand)
        query = query_vec / mx.norm(query_vec)

        sims = [mx.dot(cand, entry[0]) for entry in self.long.entries]
        return max(sims) if sims else 0.0

# ------------------------------------------------------------
# 5️⃣  Goal Updater (EMA‑driven intention)
# ------------------------------------------------------------
class GoalUpdater:
    def __init__(self, dim: int = 256, ema_alpha: float = 0.05):
        self.current = mx.zeros(dim, dtype=mx.float32)
        self.ema_alpha = ema_alpha

    def update(self, reference: mx.array):
        self.current = (1 - self.ema_alpha) * self.current + self.ema_alpha * reference

# ------------------------------------------------------------
# 6️⃣  Dual‑Buffer Engine that ties everything together
# ------------------------------------------------------------
class DualMemoryEngine:
    def __init__(self,
                 short_cfg: dict,
                 long_cfg: dict,
                 goal_dim: int = 256):
        self.short = ShortTermMemory(**short_cfg, goal_dim=goal_dim)
        self.long  = LongTermMemory(**long_cfg)

        self.observer = MemoryObserver(
            short=self.short,
            long=self.long,
            consolidate_thresh=0.8,
            novelty_check_every=50,
            age_beta=0.001,
        )
        # expose the goal updater to the observer
        self.observer._goal_updater = self._goal_updater
        self._goal_updater = GoalUpdater(dim=goal_dim, ema_alpha=0.03)

    # ----------------------------------------------------------
    def generate_and_learn(self, prompt: str, temperature: float = 1.0):
        """
        Generates tokens step‑by‑step, stores each embedding,
        consolidates when appropriate, and updates the global goal.
        Returns the generated text.
        """
        token_ids = tokenize(prompt)
        generated: List[int] = []

        # Dummy audio embedding – in a real system you would feed actual audio features.
        dummy_audio_dim = self.short.feature_dim // 2
        dummy_audio = mx.random.normal((dummy_audio_dim,), dtype=mx.float32)

        for tid in token_ids:
            # ---- Generation step -------------------------------------------------
            out = model.generate_one(token_id=tid)
            next_tok = int(out.token)          # actual token id
            generated.append(next_tok)

            # ---- Grab embeddings -------------------------------------------------
            # `out.embedding` is assumed to be a 256‑D vector for the *text* token.
            txt_emb = out.embedding                     # shape (feature_dim,)
            # For a real audio stream you would extract a separate embedding.
            # Here we just reuse the same vector to keep the demo simple.
            audio_emb = dummy_audio

            # ---- Store in STM (reward = 1 if printable) -------------------------
            reward = 1.0 if 32 <= next_tok < 127 else 0.0
            self.short.add(audio_emb, txt_emb, energy=reward)

            # ---- Throttle consolidation (only every few steps) -------------------
            if self.short.time_step % 5 == 0:
                # Grab the *latest* fused embedding (the one we just added)
                latest_vec = self.short.buffer[-1]

                # Add it to LTM (it will go through the novelty gate)
                self.long.add(latest_vec, energy=reward)

                # After a few consolidation steps we can query LTM for the *best* match
                # and use that vector to pull the goal toward it.
                # (For demo purposes we just reuse the same vector.)
                self._goal_updater.update(latest_vec)

            # ---------------------------------------------------------------
            # OPTIONAL: run a counter‑factual check every N steps
            # ---------------------------------------------------------------
            if self.short.time_step % self.observer.novelty_check_every == 0:
                cf_score = self.observer.counterfactual_score(latest_vec)
                print(f"[CF] Novelty score: {cf_score:.3f}")

        # ---------------------------------------------------------------
        # Final housekeeping
        # ---------------------------------------------------------------
        self.observer.prune_long_term()
        self.observer.maybe_consolidate()

        return detokenize(generated)

# ------------------------------------------------------------
# 7️⃣  Run a tiny demo loop
# ------------------------------------------------------------
if __name__ == "__main__":
    engine = DualMemoryEngine(
        short_cfg={"feature_dim": 256, "max_history": 32},
        long_cfg = {"feature_dim": 256, "max_entries": 128},
        goal_dim   = 256
    )

    prompt = "Tell me a short story about a robot that learns to paint."
    print("\n=== Prompt ===")
    print(prompt)

    output = engine.generate_and_learn(prompt, temperature=1.0)
    print("\n=== Generated story ===")
    print(output)

    # ------------------------------------------------------------
    # 8️⃣  Query the LTM for reflection
    # ------------------------------------------------------------
    query_vec = engine.short.buffer[-1]          # newest STM entry
    idxs, scores = engine.long.recall(query_vec, top_k=5)
    print("\nTop LTM matches (index, similarity):")
    for i, s in zip(idxs, scores):
        print(f"  {i:02d} → {s:.3f}")

    # ------------------------------------------------------------
    # 9️⃣  Inspect the EMA‑updated goal vector
    # ------------------------------------------------------------
    print("\nCurrent global goal (first 5 dims):",
          engine._goal_updater.current[:5].tolist())
