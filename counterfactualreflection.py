##countfactual reflection optional but handy! 
def counterfactual_score(self,
                         query_vec: mx.array,
                         sigma: float = 0.1) -> float:
    """
    Perturb `query_vec` with small Gaussian noise and return the highest
    cosine similarity to any LTM entry.  This can be used as an
    exploration signal (higher → “I want to try something new”).
    """
    noise = mx.random.normal(shape=query_vec.shape, std=sigma)
    cand = query_vec + noise
    cand = cand / mx.norm(cand)
    query = query_vec / mx.norm(query_vec)

    sims = [mx.dot(cand, entry[0]) for entry in self.long.entries]
    return max(sims) if sims else 0.0


#You can call this from the observer (or from a separate planning module) to decide whether to **explore** a new direction or to **re‑consolidate** #a memory that looks promising.
