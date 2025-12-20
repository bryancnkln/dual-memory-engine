class UnifiedMemory:
    """
    • Holds a short‑term ring buffer (maxlen = short_max) that stores
      (vector, energy, timestamp, goal_vector) tuples.
    • Maintains a long‑term codebook of up to long_max entries.
    • The *observer* logic lives here: when an entry’s energy exceeds
      `consolidate_thresh` we copy it into the codebook, and we prune
      stale entries automatically.
    • All EMA updates are done in‑place, so the same vector can be
      refreshed many times without creating a new object.
    """
    def __init__(
        self,
        feature_dim: int = 256,
        short_max: int = 32,                # how many recent turns we keep
        long_max: int = 128,                # max codebook size
        ema_alpha: float = 0.1,             # smoothing for EMA updates
        consolidate_thresh: float = 0.8,    # energy needed to promote
        age_beta: float = 0.001,            # exponential decay with age
        novelty_thresh: float = 0.6,        # similarity to consider “known”
        goal_dim: int = 256                 # dimension of optional goal vector
    ):
        self.fd           = feature_dim
        self.short_max    = short_max
        self.long_max     = long_max
        self.ema_alpha    = ema_alpha
        self.consolidate_thresh = consolidate_thresh
        self.age_beta     = age_beta
        self.novelty_thresh = novelty_thresh
        self.goal_dim     = goal_dim

        # ---- short‑term storage -------------------------------------------------
        self.short_buf    = deque(maxlen=short_max)          # vectors
        self.short_energy = [0.0] * short_max                # scalar reward per slot
        self.short_ts     = [0] * short_max                  # timestamps
        self.short_goal   = [mx.zeros(goal_dim, dtype=mx.float32) for _ in range(short_max)]

        # ---- long‑term codebook -------------------------------------------------
        self.long_entries : List[Tuple[mx.array, float]] = []   # (vector, weight)

    # ----------------------------------------------------------------------
    #  Utility: cosine similarity (fast, no grads needed)
    # ----------------------------------------------------------------------
    @staticmethod
    def _cosine(a: mx.array, b: mx.array) -> float:
        na, nb = mx.norm(a), mx.norm(b)
        return 0.0 if na < 1e-8 or nb < 1e-8 else float(mx.sum(a * b) / (na * nb))

    # ----------------------------------------------------------------------
    #  Find nearest neighbour inside the *short‑term* buffer
    # ----------------------------------------------------------------------
    def _nearest_in_short(self, query: mx.array) -> Tuple[Optional[int], float]:
        if not self.short_buf:
            return None, 0.0
        sims = [self._cosine(query, mem) for mem in self.short_buf]
        idx  = mx.argmax(mx.array(sims)).item()
        return idx, sims[idx]

    # ----------------------------------------------------------------------
    #  Promote a short‑term entry to the long‑term codebook
    # ----------------------------------------------------------------------
    def _promote(self, idx: int, energy: float):
        """Take the vector stored at `idx` and push it into the codebook."""
        vec = self.short_buf[idx]               # shape (feature_dim,)
        # EMA‑update the weight (importance) of the new codebook entry
        self.long_entries.append((vec, energy))

    # ----------------------------------------------------------------------
    #  Public API – called after every new token / observation
    # ----------------------------------------------------------------------
    def add(
        self,
        vec: mx.array,          # the fused (audio+text) embedding
        energy: float = 1.0,    # scalar reward / confidence
        goal: Optional[mx.array] = None   # optional goal vector for this slot
    ):
        """
        1️⃣  Try to merge with an *already‑existing* slot (EMA update).  
        2️⃣  If no good match, push a brand‑new slot.  
        3️⃣  If the slot’s energy crosses `consolidate_thresh`,
            immediately copy it into the long‑term codebook.
        """
        # ----- 1️⃣  Find a similar existing slot -----
        idx, sim = self._nearest_in_short(vec)

        if idx is not None and sim > self.novelty_thresh:
            # EMA blend with the existing vector
            old_vec = self.short_buf[idx]
            new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * vec
            self.short_buf[idx] = new_vec

            # EMA‑blend the scalar energy as well
            old_w = self.short_energy[idx]
            new_w = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.short_energy[idx] = new_w

            # Refresh timestamp (age matters for later decay)
            self.short_ts[idx] = self.time_step

            # Optional per‑slot goal vector
            if goal is not None:
                self.short_goal[idx] = goal

        else:
            # ----- 2️⃣  Insert a fresh slot -----
            self.short_buf.append(vec)
            self.short_energy.append(energy)
            self.short_ts.append(self.time_step)
            if goal is not None:
                self.short_goal.append(goal)
            # If we already hit the max length, pop the *oldest* element
            if len(self.short_buf) > self.short_max:
                self.short_buf.popleft()
                self.short_energy.pop(0)
                self.short_ts.pop(0)
                self.short_goal.pop(0)

        # ----- 3️⃣  Consolidation check -----
        # The *most* recent slot is always at the right‑hand side of the deque.
        latest_idx = len(self.short_buf) - 1   # because we just appended
        if self.short_energy[latest_idx] >= self.consolidate_thresh:
            # Promote that slot to the long‑term codebook
            self._promote(latest_idx, self.short_energy[latest_idx])

        # ----- bookkeeping -----
        self.time_step += 1

    # ----------------------------------------------------------------------
    #  Query the long‑term codebook (returns indices & weighted scores)
    # ----------------------------------------------------------------------
    def recall(self, query: mx.array, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Return the `top_k` most similar codebook entries.
        The similarity is weighted by the stored energy (weight).
        """
        if not self.long_entries:
            return [], []

        scores = []
        for vec, w in self.long_entries:
            scores.append(self._cosine(query, vec) * w)

        sorted_idx = mx.argsort(mx.array(scores), descending=True)
        top_idx = [sorted_idx[i].item() for i in range(min(top_k, len(sorted_idx)))]
        top_score = [scores[i].item() for i in range(len(top_idx))]
        return top_idx, top_score

    # ----------------------------------------------------------------------
    #  Simple diagnostics – call whenever you want to see what’s inside
    # ----------------------------------------------------------------------
    def dump(self) -> dict:
        """Return a plain‑dict that can be JSON‑serialised for logging."""
        return {
            "short_count": len(self.short_buf),
            "short_energy": list(self.short_energy),
            "short_ts": list(self.short_ts),
            "long_count": len(self.long_entries),
            "long_entries": [
                {
                    "vec_shape": v.shape.as_tuple(),
                    "weight": w,
                    "sha256": hashlib.sha256(v.tobytes()).hexdigest()
                }
                for v, w in self.long_entries
            ],
        }
#Ring buffer + per‑slot weight
#Codebook of distilled vectors
#When energy is high, move to long‑term; prune old entries.”
## Watching the _energy thresholds_ in real time

# create the manager once (e.g. at program start)
mem = UnifiedMemory(
    feature_dim=256,
    short_max=32,
    long_max=128,
    ema_alpha=0.1,
    consolidate_thresh=0.8,
    age_beta=0.001,
    novelty_thresh=0.6,
    goal_dim=256,
)

# … inside your generation loop, after you have called `mem.add(...)`
if mem.time_step % 10 == 0:          # every 10 tokens, print a snapshot
    print("\n=== MEMORY SNAPSHOT ===")
    dump = mem.dump()
    print(f"Short‑term slots : {dump['short_count']}")
    print(f"Latest short energy: {dump['short_energy'][-1]:.3f}")
    print(f"Long‑term entries : {dump['long_count']}")
    print(f"Current consolidation threshold: {mem.consolidate_thresh:.3f}")
    print("---")
##Short‑term slots : 27
##Latest short energy: 1.23
##Long‑term entries : 45
