def add(self, audio, text, energy: float = 1.0) -> None:
    fused = mx.concat([audio, text], axis=0)

    idx, sim = self._find_nearest(fused)

    if idx is not None and sim > 0.75:                # similarity threshold
        # ----- EMA‑update the existing slot -----
        old_vec = self.buffer[idx]
        new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * fused
        self.buffer[idx] = new_vec

        # ----- Update *both* weight and gate parameter -----
        old_w   = self.weights[idx]
        new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
        self.weights[idx] = new_w

        # NEW: apply the sigmoid gate to the *new* energy
        #   gated = σ(A) * energy
        gated_energy = mx.mul(
            mx.sigmoid(mx.from_float(self.A[idx])),   # σ(A)
            mx.from_float(energy)                     # energy
        )
        self.weights[idx] = gated_energy            # store the *gated* weight

        self.timestamps[idx] = self.time_step
    else:
        # ----- CREATE NEW SLOT -----
        self.buffer.append(fused)
        self.weights.append(energy)          # store raw energy for now
        self.timestamps.append(self.time_step)
        self.A.append(0.0)                    # NEW: initialise gate param

        # Optional: if you want the *initial* weight to already be gated,
        # you could do:
        #   gated = mx.mul(mx.sigmoid(mx.from_float(0.0)), mx.from_float(energy))
        #   self.weights[-1] = gated
