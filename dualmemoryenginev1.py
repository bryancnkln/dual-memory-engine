class DualMemoryEngine:
    def __init__(self, short_cfg, long_cfg, model, voice, tokenizer, device="mlx"):
        # ----- memory objects -------------------------------------------------
        self.short  = ShortTermMemory(**short_cfg)          # <-- now stores A
        self.long   = LongTermMemory(**long_cfg)
        self.observer = MemoryObserver(self.short, self.long,
                                       consolidate_thresh=0.8,
                                       prune_every=50)

        # ----- model & tokenizer -------------------------------------------
        self.model  = model.to(device)
        self.model.eval()
        self.voice  = voice
        self.tok    = tokenizer
        self.device = device

        # ----- make sure the model knows about the extra bias -----------------
        # If you are using a custom decoder, expose a method that accepts `mem_bias`
        # (see the `forward` signature below).  If you are using a HF‑style model,
        # you can add the bias in the forward call with `bias=mem_bias`.
        # Example for a custom decoder:
        #   self.model.forward = self._forward_with_mem_bias   # bind later

    # ----------------------------------------------------------------------
    # 2️⃣  Build the memory prompt (now with gate)
    # ----------------------------------------------------------------------
    def _build_memory_prompt(self, slots):
        """
        slots: List[(audio_vec, text_vec, energy)]   # raw (un‑gated) values
        Returns:
            mem_prompt_ids – list[int] (if you use discrete memory tokens)
            mem_embs       – dense tensor [num_mem, embed_dim] (gated)
        """
        weighted_vecs = []
        for i, (audio_vec, text_vec, energy) in enumerate(slots):
            fused   = mx.concat([audio_vec, text_vec], axis=0)   # (2*feat_dim,)
            # ----- APPLY THE GATE -----
            #   gate = σ(A)   where A is a learned scalar per slot
            gate = mx.sigmoid(mx.from_float(self.short.A[len(weighted_vecs)]))
            weighted = mx.mul(fused, gate)                     # <-- gated vector
            weighted_vecs.append(weighted)

        mem_tensor = mx.stack(weighted_vecs)                # [num_mem, embed_dim]
        mem_prompt_ids = []                                 # still empty here
        return mem_prompt_ids, mem_tensor

    # ----------------------------------------------------------------------
    # 3️⃣  Get past KV + bias (the gate lives here)
    # ----------------------------------------------------------------------
    def _get_mem_past(self, mem_embs):
        # Linear projections (reuse the model's own projection layers)
        key   = self.model.W_k(mem_embs)          # [num_mem, d_k]
        value = self.model.W_v(mem_embs)          # [num_mem, d_v]

        # ----- Compute the gated bias for each memory slot -----
        # raw energies are stored in `self.short.weights` (still raw, not gated yet)
        raw_energies = [self.short.weights[i] for i in range(len(mem_embs))]
        gate_vals = [mx.sigmoid(mx.from_float(self.short.A[i]))
                     for i in range(len(raw_energies))]
        bias = mx.stack(gate_vals)                # shape [num_mem]

        # The model’s forward method must be able to consume `bias`.
        # If you have a custom forward:
        #   out = self.model(input_ids, past_key_values=(key, value), bias=bias)
        # If you are using a HF‑style model that already supports `bias`,
        # just pass `bias` there.
        return key, value, bias                     # tuple that matches your model's API
