def add(self, audio: mx.array, text: mx.array, energy: float = 1.0):
    """
    Fuse audio+text embedding and store it.
    If a similar slot already exists, blend via EMA;
    otherwise insert a fresh slot.
    """
    # 1️⃣  Fuse
    fused = mx.concatenate(
        [audio.astype(mx.float32), text.astype(mx.float32)], axis=0
    )

    # 2️⃣  Find nearest neighbour
    idx, sim = self._find_nearest(fused)

    if idx is not None and sim > 0.6:                # lower threshold → more updates
        # EMA‑blend the vector
        old_vec = self.buffer[idx]
        new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * fused
        self.buffer[idx] = new_vec

        # EMA‑blend the scalar energy
        old_w   = self.weights[idx]
        new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
        self.weights[idx] = new_w

        # Refresh timestamp (age matters for later decay)
        self.timestamps[idx] = self.time_step
    else:
        # Append a brand‑new slot
        self.buffer.append(fused)
        self.weights.append(energy)          # keep parallel lists in sync
        self.timestamps.append(self.time_step)

        # Trim auxiliary lists if we exceeded the ring size
        if len(self.weights) > self.max_history:
            self.weights.pop(0)
            self.timestamps.pop(0)

    self.time_step += 1
