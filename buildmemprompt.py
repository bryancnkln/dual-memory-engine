def _build_memory_prompt(self, slots: List[Tuple[mx.array, mx.array, float]]):
    weighted_vecs = []
    for audio_vec, text_vec, energy in slots:
        fused = mx.concat([audio_vec, text_vec], axis=0)          # (2*feat_dim,)
        # NEW: multiply by the *sigmoid* of the learned gate parameter
        gate = mx.sigmoid(mx.from_float(self.short.A[   # the gate for this slot
            len(self.short.buffer) - len(weighted_vecs) - 1])   # index of this slot
        weighted = mx.mul(fused, gate)                # <-- gated vector
        weighted_vecs.append(weighted)

    # Stack into a dense tensor that will become the KV tensor
    mem_tensor = mx.stack(weighted_vecs)               # [num_mem, embed_dim]
    # Optional: also return the list of gated token ids if you use a discrete memory vocab
    mem_prompt_ids = []                                 # still empty in this version
    return mem_prompt_ids, mem_tensor
