import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Tuple

# --------------------------------------------------------------
# 1️⃣  Helper classes (very similar to the ones you already have)
# --------------------------------------------------------------
class ShortTermMemory:
    def __init__(self, feature_dim: int, max_history: int = 32,
                 ema_alpha: float = 0.1, age_beta: float = 0.001):
        self.embed_dim = feature_dim                # dimension of one modality
        self.max_history = max_history
        self.ema_alpha = ema_alpha
        self.age_beta = age_beta

        self.buffer: deque = deque(maxlen=max_history)   # stores fused vectors (mlx or torch)
        self.weights: List[float] = []                  # scalar energy per slot
        self.timestamps: List[int] = []                 # tick when slot entered
        self.t = 0                                      # global step counter

    # ------------------------------------------------------------------
    #   PUBLIC API
    # ------------------------------------------------------------------
    def add(self, audio_vec: torch.Tensor, text_vec: torch.Tensor,
            energy: float = 1.0) -> None:
        """Fuse audio+text, store or EMA‑update an existing slot."""
        fused = torch.cat([audio_vec, text_vec])          # (2*feature_dim,)
        # find nearest neighbour (cosine similarity)
        sims = [F.cosine_similarity(fused, mem) for mem in self.buffer]
        if sims and max(sims) > 0.75:                       # similarity threshold
            idx = int(np.argmax(sims))
            # EMA‑update vector
            old = self.buffer[idx]
            new = (1 - self.ema_alpha) * old + self.ema_alpha * fused
            self.buffer[idx] = new
            # EMA‑update energy
            old_w = self.weights[idx]
            new_w = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.weights[idx] = new_w
            self.timestamps[idx] = self.t
        else:
            self.buffer.append(fused)
            self.weights.append(energy)
            self.timestamps.append(self.t)
        self.t += 1

    def get_recent(self, N: int = 2) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Return the last N slots as (audio_vec, text_vec, energy)."""
        start = max(0, len(self.buffer) - N)
        return [(self.buffer[i][:self.embed_dim],      # audio part
                 self.buffer[i][self.embed_dim:],      # text part
                 self.weights[i])                       # energy
                for i in range(start, len(self.buffer))]

    # ------------------------------------------------------------------
    #   OPTIONAL: retrieve a *dense* tensor for KV injection
    # ------------------------------------------------------------------
    def get_mem_tensor(self, N: int = 2) -> torch.Tensor:
        """Return a tensor of shape [num_mem, embed_dim] ready for KV."""
        slots = self.get_recent(N)
        tensors = []
        for a, t, _ in slots:
            # concatenate audio+text and scale by energy
            vec = torch.cat([a, t]) * np.mean([e for _,_,e in slots])   # simple scaling
            tensors.append(vec)
        return torch.stack(tensors)        # [num_mem, embed_dim]
# --------------------------------------------------------------

class LongTermMemory:
    def __init__(self, feature_dim: int, max_entries: int = 128,
                 ema_alpha: float = 0.2, novelty_thresh: float = 0.6):
        self.embed_dim = feature_dim
        self.max_entries = max_entries
        self.ema_alpha = ema_alpha
        self.thr = novelty_thresh
        self.entries: List[Tuple[torch.Tensor, float]] = []   # (vector, weight)

    def add(self, vec: torch.Tensor, energy: float):
        # find nearest neighbour
        if self.entries:
            sims = [F.cosine_similarity(vec, e[0]) for e in self.entries]
            idx = int(np.argmax(sims))
            if sims[idx] > self.thr:
                # EMA‑update existing entry
                old_vec, old_w = self.entries[idx]
                new_vec = (1 - self.ema_alpha) * old_vec + self.ema_alpha * vec
                new_w   = (1 - self.ema_alpha) * old_w + self.ema_alpha * energy
                self.entries[idx] = (new_vec, new_w)
                return
        # otherwise add a new entry (or replace the least‑important)
        if len(self.entries) < self.max_entries:
            self.entries.append((vec, energy))
        else:
            # replace the entry with the smallest weight
            min_i = int(np.argmin([w for _, w in self.entries]))
            self.entries[min_i] = (vec, energy)

    def recall(self, query: torch.Tensor, top_k: int = 5
               ) -> Tuple[List[int], List[float]]:
        """Return top‑k indices and their weighted similarity scores."""
        if not self.entries:
            return [], []
        sims = [F.cosine_similarity(query, e[0]) * e[1] for e in self.entries]
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = sorted(top_idx)[::-1]      # descending order
        return top_idx.tolist(), [sims[i] for i in top_idx]

# --------------------------------------------------------------
# 2️⃣  Memory‑Observer (same as before, but now with torch)
# --------------------------------------------------------------
class MemoryObserver:
    def __init__(self, short: ShortTermMemory, long: LongTermMemory,
                 consolidate_thresh: float = 0.8, prune_every: int = 50):
        self.short = short
        self.long  = long
        self.consolidate_thresh = consolidate_thresh
        self.prune_every = prune_every

    def maybe_consolidate(self):
        if not self.short.buffer:
            return
        latest_energy = self.short.weights[-1]
        if latest_energy >= self.consolidate_thresh:
            # Move the *latest* fused vector into LTM
            latest_vec = self.short.buffer[-1]          # already fused (audio+text)
            self.long.add(latest_vec, energy=latest_energy)

    def prune_long_term(self):
        if self.short.t % self.prune_every != 0:
            return
        if not self.long.entries:
            return
        # Find entry with the smallest weight and delete it
        min_i = int(np.argmin([w for _, w in self.long.entries]))
        del self.long.entries[min_i]

    def step(self):
        self.maybe_consolidate()
        self.prune_long_term()

# --------------------------------------------------------------
# 3️⃣  DualMemoryEngine – the glue that ties everything together
# --------------------------------------------------------------
class DualMemoryEngine:
    """
    Top‑level orchestrator:
      * owns STM & LTM,
      * owns the observer,
      * knows how to talk to the LM,
      * owns the voice‑synthesiser (SuperTonic in your case).
    """
    def __init__(self,
                 short_cfg: dict,
                 long_cfg:  dict,
                 model,                # the LM (nn.Module) already moved to device
                 voice_synth,          # object with .encode(audio) and .decode(tokens)
                 tokenizer,           # tokenizer that maps str → token ids
                 device: str = "cpu"):
        # ---- memory objects -------------------------------------------------
        self.short  = ShortTermMemory(**short_cfg)
        self.long   = LongTermMemory(**long_cfg)
        self.observer = MemoryObserver(self.short, self.long)

        # ---- model & tokenizer --------------------------------------------
        self.model = model.to(device)
        self.model.eval()                     # inference mode
        self.tokenizer = tokenizer
        self.device = device

        # ---- voice synthesiser -----------------------------------------------
        self.voice = voice_synth
        self.voice_device = voice_synth.device   # usually "mlx" or "cpu"

    # --------------------------------------------------------------
    # 2️⃣  Public entry point – called each time a new audio chunk arrives
    # --------------------------------------------------------------
    async def process_chunk(self, audio_chunk_np: np.ndarray,
                            user_text: str):
        """
        1️⃣ Encode audio → embedding,
        2️⃣ Store it in STM,
        3️⃣ Run the observer,
        4️⃣ Build a memory‑prompt,
        5️⃣ Generate reply token ids,
        6️⃣ Synthesize speech and play it.
        """
        # ----------------------------------------------------------
        # 2️⃣  Encode audio → embedding (float32 Tensor)
        # ----------------------------------------------------------
        audio_tensor = torch.from_numpy(audio_chunk_np).float()   # (samples,)
        # SuperTonic expects a 1‑D float32 array; we assume it returns a
        # torch Tensor of shape (feature_dim,).  If it returns an MLX array,
        # convert with `torch.from_numpy(mx_array.tolist())`.
        audio_emb = torch.from_numpy(self.voice.encode(audio_chunk_np)).float()

        # ----------------------------------------------------------
        # 3️⃣  Store in STM (energy can be a simple reward, e.g. 1.0 for speech)
        # ----------------------------------------------------------
        # Here we *don’t* have a text embedding yet – we only have an audio embedding.
        # For the first turn we can pass an *empty* text vector (zeros) and give it energy=1.
        self.short.add(audio_vec=audio_emb,
                       text_vec=torch.zeros_like(audio_emb),   # placeholder
                       energy=1.0)

        # ----------------------------------------------------------
        # 4️⃣  Let the observer decide whether to move anything to LTM
        # ----------------------------------------------------------
        self.observer.step()

        # ----------------------------------------------------------
        # 5️⃣  Build the memory‑prompt and generate a reply
        # ----------------------------------------------------------
        # a) Grab the most recent N memory slots (e.g., N=2)
        slots = self.short.get_recent(N=2)          # list of (audio_vec, text_vec, energy)
        mem_prompt_ids, mem_embs = self._build_memory_prompt(slots)

        # b) Tokenise the *user utterance* (the current turn)
        user_ids = self.tokenizer.encode(user_text)   # List[int]

        # c) Concatenate memory tokens + user tokens
        inp_ids = torch.tensor(mem_prompt_ids + user_ids, dtype=torch.long,
                               device=self.device).unsqueeze(0)   # (1, seq_len)

        # d) Get past KV cache that already contains memory KV from previous turns
        past_kv = self._get_mem_past(mem_embs)       # returns (key, value) tuple

        # e) Forward the model (we keep the KV cache to avoid recomputing everything)
        with torch.no_grad():
            out = self.model(input_ids=inp_ids,
                             past_key_values=past_kv)

        # f) Sample the next token(s)
        reply_ids = self._sample_from_logits(out.logits)

        # ----------------------------------------------------------
        # 6️⃣  Convert token ids → speech waveform → play
        # ----------------------------------------------------------
        # SuperTonic expects a *list* of token ids (int) rather than a tensor.
        wav = self.voice.decode(reply_ids)          # returns a np.ndarray (waveform)
        # Play the waveform (non‑blocking example):
        import sounddevice as sd
        sd.play(wav, self.voice.sample_rate, blocking=True)

        # ----------------------------------------------------------
        # 7️⃣  Store the newly generated reply back into STM (optional)
        # ----------------------------------------------------------
        # Encode the reply text back into an embedding and push it into STM.
        # This makes the model able to “remember” what it just said.
        reply_text = self.tokenizer.decode(reply_ids)   # back to string
        reply_emb = self._encode_text(reply_text)      # shape (feature_dim,)
        # We reuse the *same* audio embedding that just arrived (or you can get a fresh one)
        self.short.add(audio_vec=audio_emb,
                       text_vec=reply_emb,
                       energy=1.0)

        # Finally return the generated ids for any downstream logging you want
        return reply_ids

    # --------------------------------------------------------------
    # 3️⃣  Helper: build the memory prompt (dense tensors + token ids)
    # --------------------------------------------------------------
    def _build_memory_prompt(self, slots: List[Tuple[torch.Tensor, torch.Tensor, float]]
                            ) -> Tuple[List[int], torch.Tensor]:
        """
        * slots – list of (audio_vec, text_vec, energy)
        * Returns:
            - mem_prompt_ids  – list[int] that can be concatenated with the user prompt
            - mem_embs        – dense tensor [num_mem, embed_dim] that will become KV
        """
        # 1️⃣  Scale each fused vector by its energy
        weighted_vecs = []
        for audio_vec, text_vec, energy in slots:
            fused = torch.cat([audio_vec, text_vec]) * energy   # shape (embed_dim*2)
            weighted_vecs.append(vec)

        # 2️⃣  Stack into a dense tensor
        mem_tensor = torch.stack(weighted_vecs)   # [num_mem, embed_dim]

        # 3️⃣  Optional: map each memory vector to a few discrete tokens.
        #     If you have a *memory tokeniser* (tiny classifier) you can do:
        #         mem_token_ids = self.memory_tokeniser.encode(mem_tensor)
        #     Here we simply keep the dense representation for KV injection.
        mem_prompt_ids = []                       # we don't have discrete ids yet
        return mem_prompt_ids, mem_tensor

    # --------------------------------------------------------------
    # 4️⃣  Helper: create past_key_values that include the memory KV
    # --------------------------------------------------------------
    def _get_mem_past(self, mem_embs: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Takes a dense memory tensor (shape [num_mem, embed_dim]) and returns
        a pair of tensors suitable for `model(..., past_key_values=…)`.

        If your LM expects separate key/value caches for *each* modality,
        you can split them here.  The crucial part is adding the **energy bias**:
        """
        # Linear projections – these are parameters of the LM we reuse.
        # Assume the model exposes `self.W_k` and `self.W_v` as nn.Linear layers.
        # If you are using a pure MLX model you can implement the same math with `mx`.
        key   = self.model.W_k(mem_embs)            # [num_mem, d_k]
        value = self.model.W_v(mem_embs)            # [num_mem, d_v]

        # ----- Energy bias (optional but recommended) -----
        # Convert the scalar energies back to a tensor of shape [num_mem, 1]
        energies = torch.tensor([e for _, _, e in self.short.get_recent(len(slots))],
                                device=self.device, dtype=torch.float)
        bias = energies[:, None]                     # broadcast over head dimension later
        # Add bias *before* the softmax that computes attention weights
        # (the bias will be added inside the attention routine of the model,
        #  so we just store it together with key/value.)
        # If your model does not support an explicit bias argument, you can
        # inject it directly into the attention logits later (see §3.2).

        # Return a tuple that matches the signature expected by `model`.
        # Many transformers expect a tuple of (key, value) for each layer.
        return (key, value)          # each is [num_mem, d_k/d_v]; they will be
                                     # concatenated with the *previous* KV cache
                                     # that the model keeps internally.

    # --------------------------------------------------------------
    # 5️⃣  Helper: simple sampling from logits
    # --------------------------------------------------------------
    def _sample_from_logits(self, logits: torch.Tensor) -> List[int]:
        """Greedy sampling of the *next* token (you can replace with top‑p)."""
        probs = F.softmax(logits[:, -1], dim=-1)          # (1, vocab)
        next_id = torch.multinomial(probs, num_samples=1).item()
        return [next_id]

    # --------------------------------------------------------------
    # 6️⃣  Helper: encode a *text* string into the same embedding space
    # --------------------------------------------------------------
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a string with the same tokenizer you used for the LM."""
        ids = self.tokenizer.encode(text)            # List[int]
        # Convert to dense embedding – you probably have a separate text encoder.
        # For simplicity we reuse the same linear projection that maps token ids
        # to the model's hidden size, then take the *mean* of the hidden states.
        # This is a placeholder – replace with your actual text encoder.
        with torch.no_grad():
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
            hidden = self.model.transformer.wte(ids_tensor)   # token embeddings
            emb = hidden.mean(dim=0)                         # (hidden_dim,)
        return emb

    # --------------------------------------------------------------
    # 7️⃣  Public method to start the whole loop (e.g., from a button)
    # --------------------------------------------------------------
    async def start(self):
        """Open the microphone, then enter the async listening loop."""
        # Open an AudioWorklet / ScriptProcessor that will feed us chunks.
        # The details depend on the browser; the outline is:
        #   - create an AudioContext,
        #   - connect a ScriptProcessor,
        #   - on each `onaudioprocess` push the chunk into a queue.
        #   - when the queue is non‑empty, call `process_chunk`.
        # Here we only illustrate the *high‑level* flow:

        # 1️⃣  Open the microphone
        stream = await self._open_mic()
        # 2️⃣  Main async loop:
        while True:
            chunk = await self._get_next_chunk_from_queue()   # blocks until a chunk arrives
            if chunk is None:
                await asyncio.sleep(0.001)
                continue
            # `user_text` is whatever you have transcribed from the chunk.
            # For a pure voice‑only pipeline you could skip the transcription
            # and directly call `self.process_chunk(chunk, user_text="")`.
            await self.process_chunk(chunk, user_text="")   # you can pass a transcription here

    # --------------------------------------------------------------
    # 8️⃣  Low‑level microphone helpers (pseudo‑code)
    # --------------------------------------------------------------
    async def _open_mic(self):
        ctx = await torch.audio.open_stream(
            source="mic",
            channels=1,
            sample_rate=self.voice.sample_rate,
            blocksize=int(self.voice.sample_rate * 0.2)   # 200 ms blocks
        )
        # Store the stream somewhere so `process_chunk` can read from it.
        # (Implementation details omitted – see the MDN “AudioWorklet” docs.)
        return stream

    async def _get_next_chunk_from_queue(self):
        """Blocking‑ish wait until a chunk is available, then pop it."""
        # Pseudo‑implementation:
        if not self._chunk_queue:
            await asyncio.sleep(0.001)
            return None
        return self._chunk_queue.pop(0)          # numpy array of shape (samples,)

    # --------------------------------------------------------------
    # 9️⃣  Utility: convert a dense tensor to a list of Python ints
    # --------------------------------------------------------------
    @staticmethod
    def _tensor_to_int_list(t: torch.Tensor) -> List[int]:
        return t.cpu().numpy().tolist()
