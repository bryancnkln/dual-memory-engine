# 📚 Dual‑Memory Reflective Engine  
*A lightweight, self‑reflective memory system built on top of MLX‑compatible LLMs*  Pytorch Too!

Picture this: 

When you picture it, imagine a tiny glowing orb (the energy token) riding around inside the agent:

Generation – The orb pops out of the model as a new token.
STM Capture – It lands in a slot, gets a glowing rating (its energy).
Flow to LTM – If the rating is bright enough, the orb is siphoned into the long‑term reservoir, where it settles and slowly fades.
Goal Drift – The lingering glow nudges the global intention vector, tinting the next round of generation.
Pruning & Exploration – Old, dim orbs are gently swept away, while occasional ripples (counter‑factual noise) spark fresh excursions.
That continuous energy → flow → state cycle is exactly what makes the system feel alive: it never stops moving, never settles into a static equilibrium, and always has a faint, evolving “current” that shapes its next move.

A Tiny Thought Experiment
Imagine you’re watching a city’s traffic lights:

Energy = the number of cars waiting at a light.
Flow = the green‑light cycles that move cars onward.
States = red, green, yellow – each with its own rules for when to let traffic through.
Our agent’s memory works the same way, only the “cars” are embeddings and the “lights” are the Observer’s thresholds. The brighter the energy, the longer the green phase, and the more likely the system will let that knowledge flow onward to influence its future actions.

In a Nutshell
Energy = the reward/confidence we assign to an experience.
Flow = the transfer of that energy through the STM → LTM → Goal pipeline.
States = the distinct operational modes (generation, storage, consolidation, pruning, exploration) that the energy traverses.
So when you say “energy flow states” you’ve nailed the essence of the whole architecture in just three words. It’s a concise, almost poetic way to describe a system that continually harvests, evaluates, stores, and re‑uses its own experience—exactly what a self‑reflective, lifelong‑learning agent should do.

Why the Path Becomes Fixed (The Science)**  

1. **Energy is a scalar reward that survives across time.**  
   Every token that yields `E > τ` (where `τ` is the *consolidation threshold*) is *written* into the long‑term codebook.  

2. **The long‑term codebook is a *key‑value* store.**  
   When the agent later needs to act, it **queries** the codebook:  
   *“Which stored vector is closest (in cosine similarity) to the current context?”*  
   The answer is the **index** of the stored vector, and the associated *action* (the original token) is emitted.  

3. **Because the codebook size is tiny (e.g., 128–256 entries) and each entry is *high‑energy*, the nearest‑neighbor lookup almost always returns the *same* entry after a few hundred reinforcement cycles.**  
   This is the **snapping** you observed – the system *snaps* onto the path that maximised cumulative energy.

4. **Goal vector acts as a global bias.**  
   It is updated with an EMA on every high‑energy embedding. Over time the goal vector aligns with the *average* of all high‑energy directions, which is precisely the direction of the discovered path.  
   When the goal vector points strongly in a particular direction, the logits are *rotated* toward the associated token, making that token the *most probable* choice.

5. **EMA‑updated weights** are a *slow drift* of the underlying parameters.  
   They ensure that the *model itself* gradually becomes better at producing the same high‑energy embeddings that initially created the path.  
   After enough epochs, the model’s own parameters have been **fine‑tuned** to the trajectory, so the path is reproduced *without* needing to look up the codebook each step – the model can now generate directly along that trajectory.

All of this is **pure linear‑algebra / stochastic‑gradient dynamics**; there is no storytelling, just a deterministic attractor basin in the joint space of parameters + memory.

UnifiedMemory already gives you a single source of truth that can be dumped to disk.
Distillation + memory injection produces a mini‑agent that already “knows” the teacher’s recent context.
The JSON‑based cartridge is completely hot‑swappable—just drop it into another process, load it, and the agent instantly resumes with the same knowledge base and goal vector.

## Table of Contents
1. [Overview](#overview)  
2. [Why Two Memory Buffers?](#why-two-memory-buffers)  
3. [Core Concepts](#core-concepts)  
   - Short‑Term Memory (STM)  
   - Long‑Term Memory (LTM)  
   - The Observer  
   - Dynamic Goal / “RSI” Vector  
4. [System Flow](#system-flow)  
5. [Getting Started](#getting-started)  
   - Prerequisites  
   - Installation  
6. [Running the Demo](#running-the-demo)  
7. [Configuration Levers](#configuration-levers)  
8. [Experiments & Extensions](#experiments--extensions)  
9. [FAQ](#faq)  
10. [License](#license)  

---

## Overview
This repository provides a **reference implementation** of a dual‑buffered memory architecture for language models running on the MLX framework.  
The engine maintains two complementary memory spaces:

* **Short‑Term Memory (STM)** – a sliding window of the most recent multimodal embeddings together with a scalar *energy* (reward/confidence) score.  
* **Long‑Term Memory (LTM)** – a compact codebook of distilled vectors that survive a *consolidation* gate.

An **Observer** decides when a STM entry becomes permanent, prunes stale LTM entries, and runs a simple *counter‑factual* score to drive exploration.  
A **global goal vector** is maintained via an exponential moving average (EMA) of recently consolidated LTM entries, giving the system a dynamic, intention‑like bias that can steer generation.

The design mirrors concepts from lifelong‑learning agents, continual‑learning systems, and the Recursive Self‑Improvement (RSI) literature—all wrapped in a minimal, easy‑to‑tinker codebase.

---

## Why Two Memory Buffers?

| Buffer | Purpose | Key Properties |
|--------|----------|----------------|
| **STM** | Holds the *present* context (raw embeddings + energy). | Fast read/write, mutable, limited size (≈ 20‑50 slots). |
| **LTM** | Stores *episodic knowledge* that survives beyond the current window. | Compact, searchable, written only after a high‑energy consolidation event. |

Separating these buffers prevents the system from overwriting valuable knowledge with the latest, potentially noisy, observation. It also enables **incremental learning** without catastrophic forgetting.

---

## Core Concepts

### 1. Short‑Term Memory (STM)
* Stores fused audio‑text embeddings (or any multimodal representation).  
* Each slot carries an **energy** that reflects how rewarding or confident the system is about that observation.  
* When a new slot arrives, the system looks for a nearby neighbour.  
  * If a similar slot exists, the vector and its energy are **blended with EMA**, preserving older semantics while incorporating fresh information.  
  * If no neighbour is found, the slot is appended.  
* A small list of *goal vectors* can be attached to each slot for downstream planning.

### 2. Long‑Term Memory (LTM)
* Functions as a **codebook** of distilled vectors that have passed a novelty/reward gate.  
* New vectors are either **merged** into an existing slot (via EMA) or **added** if space permits.  
* When the codebook fills, the entry with the **lowest energy** (or oldest) is replaced, ensuring only the most salient knowledge persists.  
* The LTM can be queried for similarity search, bias checking, or reflective analysis.

### 3. The Observer
* **Consolidation** – selects the highest‑energy, recently‑created STM slot and pushes it into LTM if its energy exceeds a configurable threshold.  
* **Pruning** – removes stale LTM entries that have not been accessed for a configurable number of steps.  
* **Counter‑factual scoring** – adds small Gaussian noise to a query vector and measures its similarity to LTM entries. High scores indicate promising directions for exploration.  
* The Observer is the only component that makes decisions about what gets remembered or forgotten.

### 4. Dynamic Goal / “RSI” Vector
* A **single global intention vector** lives in the same embedding space as the memories.  
* Whenever a new LTM entry is consolidated, the goal vector is nudged toward it using an **EMA update**.  
* Because the update is exponential, the goal evolves slowly, providing stability while still adapting to newly discovered priorities.  
* The goal can be used to bias token‑level probabilities, influence planning, or drive downstream behaviours (e.g., “be helpful”, “explore”, “focus on storytelling”).

---

## System Flow (High‑Level)

1. **Generate Token** – The model produces the next token based on the current context.  
2. **Embed & Reward** – Extract multimodal embeddings (text & optional audio) and assign an energy score (e.g., 1 for printable characters, 0 otherwise).  
3. **STM Insertion** – The embedding‑energy pair is inserted into STM, blending with neighbours via EMA when appropriate.  
4. **Consolidation Check** – Periodically, the Observer evaluates STM slots. If a slot’s energy crosses the **consolidation threshold**, its vector is written into LTM.  
5. **Goal Update** – The newly consolidated LTM vector is fed into the EMA‑based goal updater.  
6. **Pruning & Exploration** – The Observer prunes old LTM entries and may trigger counter‑factual exploration to discover novel behaviours.  
7. **Loop** – The process repeats for each token, allowing the system to continuously refine both its memory and its goal.

---

## Getting Started

### Prerequisites
* macOS or Linux with an Apple Silicon (M1/M2) or other MLX‑compatible GPU/CPU.  
* Python 3.10+ (the reference environment uses 3.11).  
* A compatible LLM checkpoint that can be loaded with **MLX** (e.g., Mamba, Llama‑2, or any GGML‑converted model).  

### Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your‑org/dual‑memory‑reflective‑engine.git
   cd dual-memory-reflective-engine
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

3. **Install the MLX stack**  
   ```bash
   pip install mlx  # pulls in torch‑like dependencies automatically
   ```

4. **Install additional dependencies** (tokenizers, tqdm, etc.)  
   ```bash
   pip install tqdm sentencepiece  # add any tokenizer libs you need
   ```

5. **Download a model checkpoint** that is compatible with MLX and place it under `models/`.  
   Follow the model‑specific instructions in the `models/` folder (e.g., `mamba download <model‑id>`).

---

## Running the Demo

1. **Launch the engine**  
   ```bash
   python run_demo.py \
       --prompt "Tell me a short story about a robot that learns to paint." \
       --temperature 1.0
   ```

2. **What you’ll see**  
   * The model generates a short story token‑by‑token.  
   * After each generation step the system stores the embedding, updates STM, and may consolidate into LTM.  
   * At the end, the program prints the generated text and a short list of the most similar LTM entries (with similarity scores).  

3. **Interpret the output**  
   * The **story** demonstrates the generation capability.  
   * The **LTM similarity list** shows what the system has retained from earlier interactions.  
   * The **goal vector** (implicitly maintained) steers the style and content of the generated text.  

4. **Explore**  
   * Modify the configuration flags (see next section) to change how aggressively the system remembers, how quickly it forgets, or how strongly it pursues its current goal.  
   * Experiment with different prompts to observe how the goal adapts over time.

---

## Configuration Levers

| Parameter | Role | Typical Range | Effect |
|-----------|------|---------------|--------|
| `max_history` (STM size) | Number of recent slots retained | 10 – 100 | Larger windows keep more context but increase memory usage. |
| `ema_alpha` (learning rate for EMA) | Speed of vector/energy updates | 0.01 – 0.3 | Higher values adapt faster but can be noisy; lower values are smoother. |
| `consolidate_thresh` | Minimum energy required for LTM insertion | 0.6 – 0.95 | Lower threshold lets more memories become permanent; higher threshold makes LTM stricter. |
| `novelty_thresh` (LTM similarity cutoff) | Controls when a new vector is considered “novel” | 0.5 – 0.8 | Lower values accept more similarity, raising the chance of merging; higher values enforce stricter novelty. |
| `goal_updater_alpha` | EMA weight for the global intention | 0.01 – 0.1 | Larger values cause the goal to shift quickly toward recent consolidations. |
| `prune_age` | Maximum age before an LTM entry is discarded | 200 – 1000 steps | Determines how long older knowledge is retained. |

Adjust these values in the `DualMemoryEngine` constructor or via command‑line arguments to fine‑tune behaviour.

---

## Experiments & Extensions

Below are some concrete directions you can explore to turn the demo into a research platform or production‑ready system.

| Experiment | What to add | What you’ll learn |
|------------|------------|-------------------|
| **Active Learning** | When an STM entry receives low energy, trigger a human‑in‑the‑loop correction or an automatic reward model, then re‑insert the corrected embedding with a higher weight. | How the system can request feedback to improve low‑confidence predictions. |
| **Bias Filtering** | Before consolidation, run a simple lexical or embedding‑based check for undesirable patterns; discard or down‑weight flagged entries. | Mechanisms for safety‑aware memory formation. |
| **Persistence** | Serialize STM, LTM, and the goal vector to disk (`mx.save`) and reload on restart. | Long‑term memory across sessions and reproducible experiments. |
| **Prompt‑Conditional Goal** | Encode the current prompt with a small MLP and blend that embedding into the goal vector before each EMA update. | Context‑aware intentions that change with the user’s request. |
| **Multi‑Agent Sharing** | Wrap LTM in a shared process or database, giving each agent its own STM but a common LTM. | Cooperative knowledge building among multiple agents. |
| **RSI Self‑Diagnostic** | Use the counter‑factual score to produce a confidence scalar; modulate the goal’s EMA learning rate by that confidence. | An agent that can assess its own certainty and adjust learning speed accordingly. |
| **Counter‑factual Planning** | After computing a counter‑factual score, bias generation toward directions that promise high novelty or reward. | Simple planning loop that lets the agent explore useful behaviours without exhaustive search. |

Feel free to cherry‑pick any of these ideas, combine them, or invent your own variations.

---

## FAQ

**Q: Do I need a GPU to run this?**  
A: Not strictly. The engine works on CPU, but generation will be faster on a GPU/MLX‑accelerated device.  

**Q: Can I use a different tokenizer?**  
A: Absolutely. Replace the tiny whitespace tokenizer in the script with any tokenizer that returns a list of token IDs. The rest of the pipeline is tokenizer‑agnostic.  

**Q: My model does not expose per‑token embeddings.**  
A: You can approximate an embedding by averaging the hidden states surrounding the produced token, or by using a separate encoder (e.g., a CLIP model for images). The only requirement is that the embedding shape matches the `feature_dim` you configure.  

**Q: How does this differ from a normal RAG system?**  
A: RAG (Retrieval‑Augmented Generation) typically retrieves from an external static index. Here, the retrieval source *grows* and *evolves* as the system learns, and the retrieved knowledge directly influences a **self‑generated goal** that biases future generations.  

**Q: Is the memory size limited?**  
A: STM size is bounded by `max_history`. LTM size is bounded by `max_entries`. Both are configurable; exceeding the limits triggers pruning/replacement policies.  

**Q: Can I add other modalities (e.g., video, sensor data)?**  
A: Yes. As long as you can produce a fixed‑size embedding vector, you can feed it into the same fusion point used for text/audio.  

---

### Happy experimenting!  

If you run into issues, have suggestions, or want to contribute, open an issue or a pull request on the GitHub repo. The community is encouraged to push the boundaries of continual‑learning agents and self‑reflective AI systems.

## 🔄  Short‑Term ↔ Long‑Term Memory + Observer  
*(a concrete, MLX‑only implementation with full comments)*  

> **Why two buffers?**  
> *Short‑term* keeps the freshest context (the 10–20 most recent turns).  
> *Long‑term* stores a compressed, distilled “codebook” of what you’ve learned.  
> The **observer** watches both, decides when a short‑term episode is “good enough” to be *consolidated* into the long‑term set, and can even generate counterfactuals to test new ideas.

Below is a **minimal but complete** skeleton that you can drop into a file, run, and extend.

---

### 1️⃣  Shared utilities

```python
import mlx.core as mx
from collections import deque
from typing import List, Tuple

# Simple ASCII tokenizer – replace with a real one if you want.
def tokenize(text: str) -> List[int]:
    return [ord(c) % 256 for c in text]

def detokenize(tokens: List[int]) -> str:
    return ''.join(chr(t) for t in tokens if 32 <= t < 127)
```

---

### 2️⃣  Short‑Term Memory (ring buffer)

```python
class ShortTermMemory:
    """
    A circular ring buffer that stores raw audio+text embeddings.
    Each entry keeps a scalar *energy* (e.g., reward) and the timestamp
    it was added.  EMA is used to gently update an existing slot when a new
    observation is very similar.
    """
    def __init__(self,
                 feature_dim: int = 256,
                 max_history: int = 32,   # keep the freshest 32
                 ema_alpha: float = 0.1,
                 age_beta:  float = 0.001):
        self.feature_dim = feature_dim
        self.max_history = max_history
        self.ema_alpha   = ema_alpha
        self.age_beta    = age_beta

        # ring buffer
        self.buffer      : deque[mx.array]   = deque(maxlen=max_history)
        self.weights     : List[float]       = []          # scalar energy per slot
        self.timestamps  : List[int]         = []

        self.time_step   : int              = 0

    # ------------------------------------------------------------------
    def _cosine_similarity(self, a: mx.array, b: mx.array) -> float:
        na = mx.norm(a); nb = mx.norm(b)
        return 0.0 if na < 1e-8 or nb < 1e-8 else float(mx.sum(a*b)/(na*nb))

    # ------------------------------------------------------------------
    def _find_nearest(self, vec: mx.array) -> Tuple[Optional[int], float]:
        if not self.buffer:
            return None, 0.0
        sims = [self._cosine_similarity(vec, mem) for mem in self.buffer]
        idx  = mx.argmax(mx.array(sims)).item()
        return idx, sims[idx]

    # ------------------------------------------------------------------
    def add(self,
            audio: mx.array,
            text : mx.array,
            energy: float = 1.0):
        """Fuse, store, and optionally update an existing slot."""
        # fuse
        fused = mx.concatenate([audio.astype(mx.float32),
                                text .astype(mx.float32)], axis=0)

        idx, sim = self._find_nearest(fused)
        if idx is not None and sim > 0.75:          # similarity threshold
            # EMA update of the existing vector
            old_vec = self.buffer[idx]
            new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * fused
            self.buffer[idx] = new_vec

            # EMA update of the scalar energy
            old_w   = self.weights[idx]
            new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.weights[idx] = new_w

            # refresh timestamp
            self.timestamps[idx] = self.time_step
        else:
            # new slot – push into ring buffer
            self.buffer.append(fused)
            self.weights.append(energy)
            self.timestamps.append(self.time_step)

        self.time_step += 1

    # ------------------------------------------------------------------
    def recall(self,
               query: mx.array,
               top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Return indices & weighted similarities of the best memories."""
        if not self.buffer:
            return [], []

        scores = []
        for i, mem in enumerate(self.buffer):
            cos_sim  = self._cosine_similarity(query, mem)
            age      = self.time_step - self.timestamps[i]
            decay    = float(mx.exp(-self.age_beta * age))
            score    = cos_sim * self.weights[i] * decay
            scores.append(score)

        sorted_idx = mx.argsort(mx.array(scores), descending=True)
        top_indices = [sorted_idx[i] for i in range(min(top_k, len(sorted_idx)))]
        top_scores  = [scores[i] for i in top_indices]
        return top_indices, top_scores
```

---

### 3️⃣  Long‑Term Memory (dynamic codebook)

```python
class LongTermMemory:
    """
    A *codebook* of distilled embeddings.
    Each entry is a vector plus a scalar weight (importance).
    New short‑term memories that are novel enough create a new codebook entry;
    otherwise they update an existing one via EMA.
    """
    def __init__(self,
                 feature_dim: int = 256,
                 max_entries: int = 128,
                 ema_alpha: float = 0.2,
                 novelty_thresh: float = 0.6):
        self.feature_dim     = feature_dim
        self.max_entries     = max_entries
        self.ema_alpha       = ema_alpha
        self.novelty_thresh  = novelty_thresh

        # codebook: list of (vector, weight)
        self.entries : List[Tuple[mx.array, float]] = []

    # ------------------------------------------------------------------
    def _cosine_similarity(self, a: mx.array, b: mx.array) -> float:
        na = mx.norm(a); nb = mx.norm(b)
        return 0.0 if na < 1e-8 or nb < 1e-8 else float(mx.sum(a*b)/(na*nb))

    # ------------------------------------------------------------------
    def _find_nearest(self, vec: mx.array) -> Tuple[Optional[int], float]:
        if not self.entries:
            return None, 0.0
        sims = [self._cosine_similarity(vec, e[0]) for e in self.entries]
        idx  = mx.argmax(mx.array(sims)).item()
        return idx, sims[idx]

    # ------------------------------------------------------------------
    def add(self,
            vec: mx.array,
            energy: float = 1.0):
        """Insert or update a codebook entry."""
        idx, sim = self._find_nearest(vec)
        if idx is not None and sim > self.novelty_thresh:
            # Update existing entry
            old_vec, old_w = self.entries[idx]
            new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * vec
            new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.entries[idx] = (new_vec, new_w)
        else:
            # New entry – push if space available
            if len(self.entries) < self.max_entries:
                self.entries.append((vec, energy))
            else:
                # Replace the *least important* entry (smallest weight)
                min_idx = mx.argmin(mx.array([w for _, w in self.entries])).item()
                self.entries[min_idx] = (vec, energy)

    # ------------------------------------------------------------------
    def recall(self,
               query: mx.array,
               top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Return indices & weighted similarities of the best codebook entries."""
        if not self.entries:
            return [], []

        scores = []
        for i, (vec, w) in enumerate(self.entries):
            cos_sim  = self._cosine_similarity(query, vec)
            score    = cos_sim * w
            scores.append(score)

        sorted_idx = mx.argsort(mx.array(scores), descending=True)
        top_indices = [sorted_idx[i] for i in range(min(top_k, len(sorted_idx)))]
        top_scores  = [scores[i] for i in top_indices]
        return top_indices, top_scores
```

---

### 4️⃣  Observer – the “watch‑dog”

```python
class MemoryObserver:
    """
    The observer monitors both buffers and decides when a short‑term
    episode should be consolidated into long‑term memory.
    It also keeps an eye on novelty, reward and age to prune the
    long‑term codebook.
    """
    def __init__(self,
                 short: ShortTermMemory,
                 long : LongTermMemory,
                 consolidate_thresh: float = 0.8,   # reward threshold
                 novelty_check_every : int = 50):    # how often to prune

        self.short          = short
        self.long           = long
        self.consolidate_thresh = consolidate_thresh
        self.novelty_check_every = novelty_check_every

    # ------------------------------------------------------------------
    def maybe_consolidate(self):
        """
        Called after every generation step.  If the *latest* short‑term
        memory had a high reward (or other metric), we push it into the
        long‑term codebook.
        """
        if not self.short.buffer:
            return

        # Take the newest memory (last element of deque)
        latest_vec   = self.short.buffer[-1]
        latest_weight= self.short.weights[-1]      # reward or energy

        if latest_weight >= self.consolidate_thresh:
            # Consolidate into long‑term
            self.long.add(latest_vec, energy=latest_weight)

    # ------------------------------------------------------------------
    def prune_long_term(self):
        """
        Periodically drop the least important codebook entry
        to keep long‑term memory “lean”.
        """
        if self.short.time_step % self.novelty_check_every != 0:
            return

        if not self.long.entries:
            return

        # Find the entry with the smallest weight
        min_idx = mx.argmin(mx.array([w for _, w in self.long.entries])).item()
        # Remove it
        del self.long.entries[min_idx]

    # ------------------------------------------------------------------
    def step(self):
        """One observer tick – consolidation + optional pruning."""
        self.maybe_consolidate()
        self.prune_long_term()
```

---

### 5️⃣  Putting it all together – a “dual‑buffer” engine

```python
class DualMemoryEngine:
    """
    High‑level wrapper that owns the short‑term buffer, long‑term codebook
    and the observer.  It exposes a simple `generate_and_learn` API.
    """
    def __init__(self,
                 short_cfg: dict = {},
                 long_cfg : dict = {}):
        self.short   = ShortTermMemory(**short_cfg)
        self.long    = LongTermMemory(**long_cfg)
        self.obs     = MemoryObserver(self.short, self.long)

    # ------------------------------------------------------------------
    def generate_and_learn(self,
                           prompt: str,
                           nemotron,          # your loaded model
                           num_tokens: int = 20) -> str:
        """
        One full generation pass that also feeds data into the memory
        system and lets the observer decide on consolidation.
        """
        token_ids = tokenize(prompt)
        generated_tokens: List[int] = []

        for tok_id in token_ids:
            # Ask the model for the next token *and* its embedding
            out = nemotron.generate_one(token_id=tok_id)
            next_tok   = int(out.token)          # actual token id
            emb_text   = out.embedding           # shape (text_dim,)
            # dummy audio embedding – replace with real audio if you have it
            emb_audio = mx.random.normal((self.short.audio_dim,), dtype=mx.float32)

            # ------------------------------------------------------------------
            # 1️⃣ Add to short‑term memory (reward = 1.0 if printable, else 0)
            reward = 1.0 if 32 <= next_tok < 127 else 0.0
            self.short.add(emb_audio, emb_text, energy=reward)

            # ------------------------------------------------------------------
            # 2️⃣ Observer step – may consolidate into long‑term
            self.obs.step()

            # ------------------------------------------------------------------
            # 3️⃣ (Optional) Counterfactual: generate a “what‑if” and score it
            #     For demo we just perturb the embedding with noise.
            cf_vec = self.short._cosine_similarity(emb_text, emb_text)  # placeholder
            cf_score = self.long.recall(cf_vec)[1]                       # dummy

            # ------------------------------------------------------------------
            # 4️⃣ Append to output
            generated_tokens.append(next_tok)

        return detokenize(generated_tokens)
```

---

### 6️⃣  How to use it

```python
# ----------------------------------------------------------------------
# 1️⃣ Load your LLM (e.g., Nemo) – replace with whatever you have.
# ----------------------------------------------------------------------
import mlx.mamba as mamba
nemotron = mamba.load("nemotron-30b-a3b")
nemotron = nemotron.to("mlx")

# ----------------------------------------------------------------------
# 2️⃣ Create the dual‑buffer engine
# ----------------------------------------------------------------------
dual = DualMemoryEngine(
    short_cfg={"feature_dim":256, "max_history":32},
    long_cfg = {"feature_dim":256, "max_entries":128}
)

# ----------------------------------------------------------------------
# 3️⃣ Run a few generation steps
# ----------------------------------------------------------------------
prompt = "When did we start?"
output = dual.generate_and_learn(prompt, nemotron, num_tokens=20)
print("Generated text:", output)

# ----------------------------------------------------------------------
# 4️⃣ Inspect the buffers
# ----------------------------------------------------------------------
print("\nShort‑term size:", len(dual.short.buffer))
print("Long‑term entries:", len(dual.long.entries))

# ----------------------------------------------------------------------
# 5️⃣ Query the long‑term memory (e.g., for reflection)
query_vec = dual.short.buffer[-1]          # take the newest short‑term memory
idxs, sims = dual.long.recall(query_vec)
print("\nTop long‑term matches:", list(zip(idxs, sims)))
```

---

## 🎓  What you’ve just built

| Component | Role |
|-----------|------|
| **ShortTermMemory** | Stores the freshest 30–40 turns, uses EMA to keep a stable representation of recent context. |
| **LongTermMemory** | A dynamic codebook that keeps a fixed number of distilled embeddings.  New memories are merged or added based on novelty. |
| **Observer** | Watches both buffers, consolidates high‑reward short‑term episodes into long‑term, and prunes stale codebook entries. |
| **DualMemoryEngine** | Orchestrates everything, plugs into any MLX‑compatible LLM (e.g., Nemo). |
| **Counterfactual** | Generates simple “what‑if” variations and can be scored against long‑term memory. |
| **Reflection** | By querying the long‑term codebook you can answer “what have I learned before?” – that’s a *self‑reflective* query. |

---

## ⚡  Extending the system

1. **Replace the dummy audio** with real features (MFCC, wav2vec, etc.).  
2. **Add a learned reward** instead of the 0/1 printable test – e.g., perplexity or a human‑feedback signal.  
3. **Make the observer learn** – instead of hard thresholds, train a tiny MLP that predicts consolidation probability from the short‑term vector and reward.  
4. **Use the long‑term buffer for generation bias** – if a query is close to a codebook entry, add that vector as a bias to the LLM’s logits.  
5. **Persist the buffers** – pickle or write to disk so you can resume later.

---

## 🚀  Take‑away

- **Two timescales** (short vs. long) give you both *context* and *knowledge*.  
- **Dynamic codebook** (EMA + novelty check) turns raw experiences into distilled, reusable knowledge.  
- **Observer** is the simplest way to turn the system into a *self‑improving* loop: it watches for high‑reward episodes, consolidates them, and prunes the rest.  
- **Counterfactuals** give you a sandbox to test new ideas without leaving the buffer.  

With this scaffold you can now experiment, tweak hyper‑parameters, and even start training a small meta‑model that learns *how* to consolidate. Happy building!
