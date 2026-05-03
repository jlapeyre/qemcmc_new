from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np
from qemcmc import build_proposer_qc
from qemcmc.circuits_core import _normalize_counts_keys_to_bitstrings as _norm


def _sample_counts(
    bitstr: str, model, pm, sampler, shots: int, *, gamma: float, time: int, delta_time: float
) -> Dict[str, int]:
    """
    Build, transpile, and sample a proposer circuit for the given bitstring.

    Parameters
    - bitstr: input bitstring to prepare (big-endian).
    - model: EnergyModel used to build the proposer.
    - pm: a pass manager (e.g., from generate_preset_pass_manager(...)).
    - sampler: a configured Qiskit SamplerV2 (or compatible) bound to a backend.
    - shots: number of measurement shots to request.
    - gamma, time, delta_time: evolution parameters for the proposer.

    Returns
    - dict[str, int]: normalized counts keyed by zero-padded bitstrings of width n.
    """
    qc = build_proposer_qc(model, bitstr, gamma=gamma, time=time, delta_time=delta_time)
    qc.measure_all()
    isa = pm.run(qc)
    try:
        job = sampler.run([isa], shots=int(shots))
    except TypeError:
        job = sampler.run([isa])
    pub = job.result()[0]
    counts = pub.data.meas.get_counts()
    #    return _norm(counts, width=isa.num_qubits)
    first_key = next(iter(counts.keys()), None)
    if isinstance(first_key, str):
        width = len(first_key)
    else:
        width = isa.num_clbits or qc.num_clbits or model.n_spins
    return _norm(counts, width=width)


def _metropolis_accept(
    e_old: float, e_new: float, *, beta: float, rng: np.random.Generator
) -> bool:
    """
    Standard Metropolis acceptance rule.

    Accept if e_new <= e_old, else with probability exp(-beta * (e_new - e_old)).
    """
    if e_new <= e_old:
        return True
    p = math.exp(-float(beta) * float(e_new - e_old))
    return bool(rng.random() < p)


from typing import Dict, Any, Optional, Tuple, Callable


def mcmc_optimize(
    initial_bitstr: str,
    model,
    pm,
    sampler,
    shots: int,
    *,
    max_iters: int = 200,
    gamma: float = 0.5,
    time: int = 6,
    delta_time: float = 0.8,
    beta: float = 1.0,
    patience: int = 30,
    rng: Optional[np.random.Generator] = None,
    callback: Optional[Callable[[int, str, float, str, float], None]] = None,
    log_every: int = 10,
) -> Dict[str, Any]:
    """
    Metropolis MCMC loop over proposal circuits; track and return the best energy found.

    Workflow per iteration
    - Build and transpile proposer for current bitstring; sample counts.
    - Evaluate energies of observed bitstrings; pick the one with lowest energy.
    - Metropolis accept/reject against current state.
    - Track best-so-far; stop if no improvement for 'patience' iterations or max_iters hit.

    Parameters
    - initial_bitstr: starting configuration (big-endian, length = model.n_spins).
    - model: EnergyModel used to build H_problem and evaluate energies.
    - pm: pass manager bound to a backend for transpilation.
    - sampler: configured SamplerV2 (or compatible) for execution.
    - shots: number of shots per iteration.
    - max_iters: maximum MCMC iterations.
    - gamma, time, delta_time: proposer evolution parameters.
    - beta: inverse temperature for Metropolis acceptance.
    - patience: stop if best energy does not improve for this many iterations.
    - rng: optional NumPy Generator (default: np.random.default_rng()).

    Returns
    - dict with keys:
      best_s, best_E, last_s, last_E, iters, accepted, accept_rate, history_E, history_s.
    """
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    cur_s = str(initial_bitstr)
    cur_E = float(model.energy(cur_s))
    best_s = cur_s
    best_E = cur_E
    no_improve = 0
    accepted = 0
    hist_E = [cur_E]
    hist_s = [cur_s]
    for it in range(1, int(max_iters) + 1):
        counts = _sample_counts(
            cur_s, model, pm, sampler, shots, gamma=gamma, time=time, delta_time=delta_time
        )
        cand_s, cand_E = None, None
        for s, _cnt in counts.items():
            e = float(model.energy(s))
            if cand_E is None or e < cand_E:
                cand_s, cand_E = s, e
        if cand_s is None:
            continue
        if _metropolis_accept(cur_E, cand_E, beta=beta, rng=g):
            cur_s, cur_E = cand_s, cand_E
            accepted += 1
        if cur_E < best_E:
            best_s, best_E = cur_s, cur_E
            no_improve = 0
        else:
            no_improve += 1
        hist_E.append(cur_E)
        hist_s.append(cur_s)
        if callback is not None:
            callback(it, cur_s, cur_E, best_s, best_E)
        elif log_every and (it % int(log_every) == 0):
            print(f"{it} {cur_s} {cur_E}")
        if no_improve >= int(patience):
            break
    iters = len(hist_E) - 1
    return {
        "best_s": best_s,
        "best_E": best_E,
        "last_s": cur_s,
        "last_E": cur_E,
        "iters": iters,
        "accepted": accepted,
        "accept_rate": (accepted / iters) if iters > 0 else 0.0,
        "history_E": hist_E,
        "history_s": hist_s,
    }
