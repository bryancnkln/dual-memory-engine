def maybe_consolidate(self):
    if not self.short.buffer:
        return
    # Grab the *latest* slot's raw energy first
    raw_energy = self.short.weights[-1]

    # NEW: apply the sigmoid gate *before* we decide to consolidate
    #   gated = σ(A) * raw_energy
    gated_energy = mx.mul(
        mx.sigmoid(mx.from_float(self.short.A[-1])),   # σ(A) for the newest slot
        mx.from_float(raw_energy)
    )

    if gated_energy >= self.consolidate_thresh:
        latest_vec = self.short.buffer[-1]            # the fused vector
        self.long.add(latest_vec, energy=gated_energy)   # <-- use gated energy
