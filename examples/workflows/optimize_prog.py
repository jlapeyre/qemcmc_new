import argparse
from qemcmc import load_dimacs_edge, mis_energy_model, random_bitstring, mcmc_optimize
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="aves-sparrow-social.gph")
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--time", type=int, default=6)
    p.add_argument("--delta-time", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--opt-level", type=int, default=1)
    args = p.parse_args()

    g = load_dimacs_edge(args.graph)
    model, _ = mis_energy_model(g, lam=args.lam)
    init_s = random_bitstring(model.n_spins, rng=args.seed)

    service = QiskitRuntimeService()
    backend = service.least_busy(min_num_qubits=model.n_spins)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=args.opt_level)
    sampler = SamplerV2(backend)

    res = mcmc_optimize(
        init_s,
        model,
        pm,
        sampler,
        args.shots,
        max_iters=args.max_iters,
        gamma=args.gamma,
        time=args.time,
        delta_time=args.delta_time,
        beta=args.beta,
        patience=args.patience,
        rng=args.seed,
    )

    print("backend:", backend.name)
    print("start:", init_s, "E=", model.energy(init_s))
    print("best_s:", res["best_s"], "best_E:", res["best_E"])
    print("iters:", res["iters"], "accept_rate:", f'{res["accept_rate"]:.3f}')

if __name__ == "__main__":
    main()
