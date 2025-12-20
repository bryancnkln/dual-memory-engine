def save_cartridge(agent: Agent, path: str):
    """
    Serialize the model weights, the unified memory, and the goal vector.
    Returns a dictionary that can be loaded later.
    """
    # 1️⃣  Serialize the model weights
    model_state = {}
    for k, v in agent.model.state_dict().items():
        model_state[k] = mx.array(v)      # convert to MX array
    # 2️⃣  Serialize the unified memory
    mem_dict = memory_to_dict(agent.short_mem)     # `agent.short_mem` is the UnifiedMemory instance
    # 3️⃣  Pack everything
    payload = {
        "model_state": model_state,
        "memory": mem_dict,
        "persona_A": agent.persona_A.tolist(),
        "goal_vector": agent.goal_updater.current.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load_cartridge(path: str, device: str = "mlx") -> Agent:
    """Inverse of `save_cartridge`. Returns a ready‑to‑run Agent."""
    with open(path, "r") as f:
        payload = json.load(f)

    # ---- 1️⃣  Re‑create the model -------------------------------------------------
    # The model architecture is assumed to be known; here we load the same family.
    model = mamba.load("nemotron-30b-a3b")          # <-- you can replace with any model you stored
    model.set_parameters([payload["model_state"][k] for k in payload["model_state"]])
    model = model.to(device)

    # ---- 2️⃣  Re‑create the UnifiedMemory -----------------------------------------
    saved_mem = load_memory_from_dict(payload["memory"])
    # ---- 3️⃣  Assemble a new Agent instance ---------------------------------------
    agent = Agent(
        model_name="custom",                # just a placeholder – we already have `model`
        persona_A=mx.array(payload["persona_A"], dtype=mx.float32),
        self_awareness=True,
        shared_long=restore_long(saved_mem)       # a tiny helper that extracts the long‑term store
    )
    # Restore the short‑term and long‑term buffers directly:
    agent.short_mem = restored_mem
    agent.long_mem   = restored_mem

    # Restore the goal vector
    goal_vec = mx.array(payload["goal_vector"], dtype=mx.float32)
    agent.goal_updater.current = goal_vec

    return agent
