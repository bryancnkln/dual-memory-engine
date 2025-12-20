# ------------------------------------------------------------
# 0️⃣  Imports (same as before)
# ------------------------------------------------------------
import mlx.core as mx
import mamba          # your MLX‑compatible model wrapper
import time, hashlib, json
# (tokenize / detokenize helpers from the previous answer)

# ------------------------------------------------------------
# 1️⃣  Load a model
# ------------------------------------------------------------
model = mamba.load("nemotron-30b-a3b").to("mlx")

# ------------------------------------------------------------
# 2️⃣  Initialise the unified memory manager
# ------------------------------------------------------------
mem = UnifiedMemory(
    feature_dim=256,
    short_max=32,
    long_max=128,
    ema_alpha=0.12,
    consolidate_thresh=0.85,
    age_beta=0.0015,
    novelty_thresh=0.55,
    goal_dim=256,
)

# ------------------------------------------------------------
# 3️⃣  Simple generation + memory hook
# ------------------------------------------------------------
def generate_one_step(prompt: str) -> str:
    tokens = tokenize(prompt)
    out_ids = []

    for t in tokens:
        out = model.generate_one(token_id=t)
        next_tok = int(out.token)

        # Fake audio embedding – replace with real audio features later
        audio_emb = mx.random.normal((mem.fd // 2,), dtype=mx.float32)

        # Grab a *goal* vector for this step (you could compute it from the prompt)
        goal_vec = mem.short_goal[0]          # just reuse the first goal slot for demo

        # Store in memory
        mem.add(audio_emb, energy=1.0, goal=goal_vec)

        out_ids.append(next_tok)

    return detokenize(out_ids)

# ------------------------------------------------------------
# 4️⃣  Run a few steps and watch the snapshot
# ------------------------------------------------------------
prompt = "Write a haiku about a robot learning to paint."
print("\n>>> Prompt:", prompt)

generated = generate_one_step(prompt)
print("\n>>> Generated:", generated)

# Print a memory snapshot every 15 generated tokens
for i in range(5):
    time.sleep(1)                     # slow it down so you can see the prints
    dump = mem.dump()
    print("\n--- Snapshot after token", len(generated.split()) + i * 5, "---")
    print(json.dumps(dump, indent=2))
##**What you’ll see:**

#- The printed story.
#- After each 5‑token chunk a JSON block showing how many short‑term slots are left, the _latest_ energy value, and how many entries are now sitting #in the long‑term codebook.
#- As the energy grows (because you’re rewarding printable tokens), the `long_count` will start to increase – that’s the consolidation you were #watching for.
##
##
