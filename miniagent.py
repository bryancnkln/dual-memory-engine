def train_mini_agent(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    dump_path: str = "mini_agent_cartridge.json",
):
    model = mamba.load(model_name).to("mlx")
    # Build an empty UnifiedMemory that will grow as we train
    mem = UnifiedMemory(
        feature_dim=256,
        short_max=32,
        long_max=128,
        ema_alpha=0.15,
        consolidate_thresh=0.7,
        novelty_thresh=0.5,
        goal_dim=256,
    )

    optimizer = mx.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        # Generate a batch of prompts (or sample from a corpus)
        prompts = random_prompts(batch_size)

        for p in prompts:
            # Generate a chunk and feed each token into the memory engine
            generated = mem.short_mem.generate_and_learn(p, model, n_tokens=12)

            # Store the generated chunk into the memory (so the agent can
            # later query it for self‑reflection)
            mem.add(
                vec=mem.short_mem.buffer[-1],           # just reuse the last embedding
                energy=1.0,
                goal=mem.short_goal[-1]               # optional goal
            )

            # Simple SGD step on the *next* token prediction
            # (here we use teacher‑free auto‑regressive loss)
            loss = model.loss_on_sequence(generated)    # pseudo‑loss fn you implement
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # After each epoch, dump the whole cartridge to disk
        save_cartridge(
            Agent(model=model, short_mem=mem, ...),   # assemble a minimal Agent wrapper
            dump_path,
        )
        print(f"=> Epoch {ep} checkpoint saved")
