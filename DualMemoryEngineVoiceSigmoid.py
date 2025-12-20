import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Tuple

# --------------------------------------------------------------
# 1️⃣  Short‑Term Memory with gate parameter A
# --------------------------------------------------------------
class ShortTermMemory:
    def __init__(self, feature_dim: int, max_history: int = 32,
                 ema_alpha: float = 0.1, age_beta: float = 0.001):
        self.feature_dim   = feature_dim
        self.max_history   = max_history
        self.ema_alpha     = ema_alpha
        self.age_beta      = age_beta

        self.buffer        = deque(maxlen=max_history)   # stores fused vectors
        self.weights       = []                           # raw energy (float)
        self.timestamps    = []                           # step count
        self.A             = nn.Parameter(torch.zeros(max_history))  # learnable gate
        self.time_step     = 0

    # --------------------------------------------------------------
    # 2️⃣  Add a new observation (audio + text) with energy
    # --------------------------------------------------------------
    def add(self, audio_vec: torch.Tensor, text_vec: torch.Tensor, energy: float = 1.0):
        fused = torch.cat([audio_vec, text_vec])          # (2*feature_dim,)

        # nearest‑neighbour similarity
        sims = [F.cosine_similarity(fused, mem) for mem in self.buffer]
        if sims and max(sims) > 0.75:
            idx = int(np.argmax(sims))
            # ----- EMA‑update vector -----
            old = self.buffer[idx]
            new = (1 - self.ema_alpha) * old + self.ema_alpha * fused
            self.buffer[idx] = new

            # ----- EMA‑update weight (energy) -----
            old_w = self.weights[idx]
            new_w = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.weights[idx] = new_w

            # ----- UPDATE GATE PARAMETER (A) -----
            #   gated_energy = σ(A) * energy   → store back as weight
            gated = torch.sigmoid(self.A[idx]) * energy
            self.weights[idx] = gated

            self.timestamps[idx] = self.time_step
        else:
            # ----- CREATE NEW SLOT -----
            self.buffer.append(fused)
            self.weights.append(energy)
            self.timestamps.append(self.time_step)
            self.A.append(nn.Parameter(torch.zeros(1)))   # new learnable gate

        self.time_step += 1

    # --------------------------------------------------------------
    # 3️⃣  Retrieve the most recent N slots
    # --------------------------------------------------------------
    def get_recent(self, N: int = 2) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        start = max(0, len(self.buffer) - N)
        return [
            (self.buffer[i][:self.feature_dim],        # audio part
             self.buffer[i][self.feature_dim:],       # text part
             self.weights[i])                         # raw energy (still stored)
            for i in range(start, len(self.buffer))
        ]

    # --------------------------------------------------------------
    # 4️⃣  Helper that builds the *gated* dense tensor
    # --------------------------------------------------------------
    def get_gated_mem_tensor(self, N: int = 2) -> torch.Tensor:
        weighted_vecs = []
        for i, (audio_vec, text_vec, _) in enumerate(self.get_recent(N)):
            fused = torch.cat([audio_vec, text_vec], dim=0)
            gate = torch.sigmoid(self.A[len(weighted_vecs)])   # σ(A_i)
            weighted = fused * gate
            weighted_vecs.append(weighted)
        return torch.stack(weighted_vecs)                    # [num_mem, embed_dim]

    # --------------------------------------------------------------
    # 5️⃣  Helper: compute gated bias for the attention logits
    # --------------------------------------------------------------
    def get_mem_bias(self) -> torch.Tensor:
        # raw energies of the recent slots (still stored as plain floats)
        raw_energies = [self.weights[i] for i in range(len(self.buffer))]
        # gated bias = σ(A_i) * raw_energy_i
        gate_vals = [torch.sigmoid(self.A[i]) * raw for i in range(len(raw_energies))]
        return torch.stack(gate_vals)               # shape [num_mem]

# --------------------------------------------------------------
# 2️⃣  Long‑Term Memory (unchanged, just keep it)
# --------------------------------------------------------------
class LongTermMemory:
    def __init__(self, feature_dim: int, max_entries: int = 128,
                 ema_alpha: float = 0.2, novelty_thresh: float = 0.6):
        self.feature_dim = feature_dim
        self.max_entries = max_entries
        self.ema_alpha   = ema_alpha
        self.thr         = novelty_thresh
        self.entries: List[Tuple[torch.Tensor, float]] = []   # (vector, weight)

    def add(self, vec: torch.Tensor, energy: float):
        # find nearest neighbour
        if self.entries:
            sims = [F.cosine_similarity(vec, e[0]) for e in self.entries]
            idx  = int(np.argmax(sims))
            if sims[idx] > self.thr:
                # EMA‑update existing entry
                old_vec, old_w = self.entries[idx]
                new_vec = (1 - self.ema_alpha) * old_vec + self.ema_alpha * vec
                new_w   = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
                self.entries[idx] = (new_vec, new_w)
                return
        # otherwise add / replace
        if len(self.entries) < self.max_entries:
            self.entries.append((vec, energy))
        else:
            # replace the smallest‑weight entry
            min_i = int(np.argmin([w for _, w in self.entries]))
            self.entries[min_i] = (vec, energy)

    def recall(self, query: torch.Tensor, top_k: int = 5
               ) -> Tuple[List[int], List[float]]:
        if not self.entries:
            return [], []
        sims = [F.cosine_similarity(query, e[0]) * e[1] for e in self.entries]
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = sorted(top_idx)[::-1]                # descending
        return top_idx.tolist(), [sims[i] for i in top_idx]

# --------------------------------------------------------------
# 3️⃣  Observer (unchanged – just uses the gated energy now)
# --------------------------------------------------------------
class MemoryObserver:
    def __init__(self, short: ShortTermMemory, long: LongTermMemory,
                 consolidate_thresh: float = 0.8, prune_every: int = 50):
        self.short  = short
        self.long   = long
        self.consolidate_thresh = consolidate_thresh
        self.prune_every = prune_every

    def maybe_consolidate(self):
        if not self.short.buffer:
            return
        raw_energy = self.short.weights[-1]                # still raw
        gated_energy = torch.sigmoid(self.short.A[-1]) * raw_energy
        if gated_energy >= self.consolidate_thresh:
            latest_vec = self.short.buffer[-1]            # fused vector
            self.long.add(latest_vec, energy=gated_energy)

    def prune_long_term(self):
        if self.short.time_step % self.prune_every != 0:
            return
        if not self.long.entries:
            return
        min_idx = int(np.argmin([w for _, w in self.long.entries]))
        del self.long.entries[min_idx]

    def step(self):
        self.maybe_consolidate()
        self.prune_long_term()

# --------------------------------------------------------------
# 3️⃣  DualMemoryEngine – now with gate‑aware memory
# --------------------------------------------------------------
class DualMemoryEngine:
    def __init__(self, short_cfg, long_cfg, model, voice_synth,
                 tokenizer, device="mlx"):
        # ---- memory ----------------------------------------------------
        self.short  = ShortTermMemory(**short_cfg)
        self.long   = LongTermMemory(**long_cfg)
        self.observer = MemoryObserver(self.short, self.long,
                                       consolidate_thresh=0.8,
                                       prune_every=50)

        # ----- model / tokenizer / voice ---------------------------------
        self.model  = model.to(device)
        self.model.eval()
        self.voice  = voice_synth
        self.tokenizer = tokenizer
        self.device = device

        # If your LM expects a custom `bias` argument, store it for later:
        self._last_mem_bias = None

    # ------------------------------------------------------------------
    # 4️⃣  Main entry point – called each time a new audio chunk arrives
    # ------------------------------------------------------------------
    async def process_chunk(self, audio_np: np.ndarray, user_text: str = ""):
        # --------------------------------------------------------------
        # 4.1  Encode audio → embedding (float tensor)
        # --------------------------------------------------------------
        audio_emb = torch.from_numpy(voice.encode(audio_np)).float()   # (feature_dim,)

        # --------------------------------------------------------------
        # 5️⃣  Store it in STM (energy = 1.0 for demo; you can replace)
        # --------------------------------------------------------------
        # No text embedding yet – we pass an empty vector (zeros)
        self.short.add(audio_vec=audio_emb,
                       text_vec=torch.zeros_like(audio_emb),
                       energy=1.0)

        # --------------------------------------------------------------
        # 6️⃣  Let the observer decide whether to move it to LTM
        # --------------------------------------------------------------
        self.observer.step()

        # --------------------------------------------------------------
        # 7️⃣  Build the *gated* memory prompt
        # --------------------------------------------------------------
        slots = self.short.get_recent(N=2)               # List[(audio, text, energy)]
        mem_prompt_ids, mem_embs = self._build_memory_prompt(slots)

        # --------------------------------------------------------------
        # 8️⃣  Tokenise the user utterance
        # --------------------------------------------------------------
        user_ids = self.tok.encode(user_text)            # List[int]
        inp_ids = torch.tensor(mem_prompt_ids + user_ids,
                               dtype=torch.long, device=self.device).unsqueeze(0)

        # --------------------------------------------------------------
        # 9️⃣  Retrieve past KV + bias (the bias now contains the gate)
        # --------------------------------------------------------------
        key, value, bias_tensor = self._get_mem_past(self.short.get_gated_mem_tensor(2))

        # --------------------------------------------------------------
        # 10️⃣  Forward the LM (note the extra `bias` argument)
        # --------------------------------------------------------------
        with torch.no_grad():
            out = self.model(input_ids=inp_ids,
                             past_key_values=(key, value),
                             bias=bias_tensor)                # <-- NEW
        # --------------------------------------------------------------
        # 11️⃣  Sample the next token(s)
        # --------------------------------------------------------------
        reply_ids = self._sample_from_logits(out.logits)

        # --------------------------------------------------------------
        # 12️⃣  Synthesize speech and play it
        # --------------------------------------------------------------
        wav = self.voice.decode(reply_ids)               # numpy array
        sd.play(wav, self.voice.sample_rate, blocking=True)

        # --------------------------------------------------------------
        # 13️⃣  Store the reply back into STM (so the model can remember it)
        # --------------------------------------------------------------
        reply_text = self.tok.decode(reply_ids)
        reply_emb  = self._encode_text(reply_ids)        # shape (feature_dim,)
        self.short.add(audio_vec=audio_emb,
                       text_vec=reply_emb,
                       energy=1.0)

        return reply_ids

    # ------------------------------------------------------------------
    # Helper: build the memory prompt (now with gate)
    # ------------------------------------------------------------------
    def _build_memory_prompt(self, slots: List[Tuple[torch.Tensor, torch.Tensor, float]]
                             ) -> Tuple[List[int], torch.Tensor]:
        weighted_vecs = []
        for i, (audio_vec, text_vec, _) in enumerate(slots):
            fused  = torch.cat([audio_vec, text_vec], dim=0)          # (2*feat_dim,)
            # ---- GATE !---
            gate = torch.sigmoid(self.short.A[len(weighted_vecs)])   # σ(A_i)
            weighted = fused * gate                               # <-- gated vector
            weighted_vecs.append(weighted)

        mem_tensor = torch.stack(weighted_vecs)               # [num_mem, embed_dim]
        mem_prompt_ids = []                                   # still empty (no discrete ids)
        return mem_prompt_ids, mem_tensor

    # ------------------------------------------------------------------
    # Helper: get KV + bias (the bias now contains the sigmoid gate)
    # ------------------------------------------------------------------
    def _get_mem_past(self, mem_embs: torch.Tensor):
        # Linear projections from the model (they are stored inside the model)
        key   = self.model.W_k(mem_embs)          # [num_mem, d_k]
        value = self.model.W_v(mem_embs)          # [num_mem, d_v]

        # ---- Compute the gated bias (σ(A) * raw_energy) ----
        raw_energies = [self.short.weights[i] for i in range(len(mem_embs))]
        gate_vals = [torch.sigmoid(self.short.A[i]) * raw for i in range(len(raw_energies))]
        bias = torch.stack(gate_vals)               # shape [num_mem]

        return key, value, bias                     # tuple for the model
