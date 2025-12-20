import mamba
import numpy as np

def distill_student(
    teacher_name: str,
    student_name: str,
    prompt: str,
    n_teacher_steps: int = 200,
    student_train_steps: int = 50,
    distillation_temp: float = 1.0,
    device: str = "mlx"
):
    """
    1️⃣  Load teacher (big) and student (small) models.
    2️⃣  Run the teacher for `n_teacher_steps` while storing its
        short‑term + long‑term memory.
    3️⃣  For each teacher step, generate a *soft target* distribution
        from the teacher’s logits and train the student on that
        distribution (knowledge distillation).
    4️⃣  After training, copy the teacher’s memory into the student’s
        `UnifiedMemory` so the student starts with the same “knowledge”
        cartridge.
    """
    # ---- 1️⃣  Load models -------------------------------------------------
    teacher = mamba.load(teacher_name).to(device)
    student = mamba.load(student_name).to(device)

    # ---- 2️⃣  Create memory objects ---------------------------------------
    short_cfg = {"feature_dim": teacher.output_dim, "max_history": 32}
    long_cfg  = {"feature_dim": teacher.output_dim, "max_entries": 256}
    mem_teacher = UnifiedMemory(**short_cfg, **long_cfg)

    # The teacher will fill its own memory automatically when we call
    # `generate_and_learn` later (see the demo below).

    # ---- 3️⃣  Distillation loop -----------------------------------------
    # We will generate token‑by‑token from the teacher,
    # compute a softened probability distribution over the vocabulary,
    # and train the student with a cross‑entropy loss that also
    # incorporates the memory‑derived reward.
    optimizer = mx.optim.Adam(student.parameters(), lr=1e-4)

    for step in range(n_teacher_steps):
        # 3a️⃣  Generate one token from the teacher
        out = teacher.generate_one(token_id=0)       # start token
        # 3b️⃣  Forward the hidden state through the memory engine
        # (here we reuse the unified engine from the previous answer)
        # For demo purposes we just call a dummy step:
        mem_teacher.add(
            audio=mx.random.normal((short_cfg["feature_dim"]//2,), dtype=mx.float32),
            text=out.embedding,
            energy=1.0
        )

        # 3c️⃣  Build a soft target distribution
        #   – use teacher's logits at the next position
        logits = out.logits                     # shape (vocab,)
        probs  = mx.softmax(logits / distillation_temp)

        # 3d️⃣  Sample a token for the student (or just use argmax)
        next_tok = int(mx.random.choice(len(probs), p=probs))
        # 3e️⃣  Train the student on that token
        optimizer.zero_grad()
        student_out = student.generate_one(token_id=next_tok)
        # Compute cross‑entropy against the *teacher* distribution
        loss = -mx.dot(probs, mx.log(student_out.prob_next))   # simplified
        loss.backward()
        optimizer.step()

        # 3f️⃣  Periodically copy teacher memory into the student
        if step % 20 == 0:
            # Grab a copy of the teacher memory (deep copy)
            mem_copy = UnifiedMemory(**mem_teacher.dump())
            # Merge it into the student's own buffer:
            student.short = mem_copy.short          # overwrite with teacher’s short buffer
            student.long  = mem_copy.long           # overwrite with teacher’s long buffer
            # (Optionally also copy the EMA‑updated goal vector)
            student.goal_updater.current = mem_copy.short_goal[-1]  # last goal slot

    # ---- 4️⃣  Return the distilled student model -------------------------
    return student
