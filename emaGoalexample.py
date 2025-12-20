class GoalUpdater:
    """
    Holds a single 256‑D intention vector that is updated by EMA
    whenever a high‑weight LTM entry is consolidated.
    """
    def __init__(self, dim: int = 256, ema_alpha: float = 0.05):
        self.current = mx.zeros(dim, dtype=mx.float32)
        self.ema_alpha = ema_alpha

    def update(self, reference: mx.array):
        """
        Move the current goal a little toward `reference`.
        """
        self.current = (1.0 - self.ema_alpha) * self.current + self.ema_alpha * reference
