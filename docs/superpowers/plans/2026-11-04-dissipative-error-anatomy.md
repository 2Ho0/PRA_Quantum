# Dissipative Error Anatomy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the theory note, a tested numerics library and the figures that verify Theorems A/A2/B1–B3, Lemma (γ_c*), and Proposition C of the spec.

**Architecture:**
- Small dense-matrix Lindblad simulator for N ≤ 8. Dissipators are applied elementwise in the z basis, and the same simulator handles both Schrödinger and Heisenberg pictures.
- Charge tools project operators onto U(1) sectors.
- Bound functions implement the spec's formulas.
- A separate free-fermion covariance module reaches N ≈ 200.
- Experiments are standalone scripts that write PNG and JSON files to `results/`.

**Tech Stack:** Python 3.14, NumPy 2.5, SciPy 1.18 (`solve_ivp` DOP853), matplotlib 3.11, pytest 9.2. Interpreter: `C:/Users/mwill/.venvs/pra-qno/Scripts/python.exe` (written `$PY` below).

**Spec:** `docs/superpowers/specs/2026-11-04-dissipative-error-anatomy-design.md`

## Global Constraints

- Basis convention: |0⟩ = spin up (σ^z = +1). Site 0 is the leftmost Kronecker factor, so basis index b = Σ_i bit_i 2^(n-1-i). Emission jump σ^- = |1⟩⟨0|.
- Dephasing generator: D(ρ) = Σ_ij Γ_ij (σ^z_i ρ σ^z_j − ½{σ^z_j σ^z_i, ρ}), with Γ real symmetric PSD (n×n). Collective rate γ_c means Γ_ij = γ_c for all i, j. The resulting charge-q damping is 2 γ_c q² (q = change of S^z = ½Σσ^z).
- XY Hamiltonian: H = ½ Σ_{i<j} J_ij (σ^x σ^x + σ^y σ^y) + Σ_i B_i σ^z_i, with J_ij = J0 / |i−j|^α, J0 = 1, α = 1.
- Default ODE tolerances: rtol = 1e-10, atol = 1e-12.
- No git commits (the project is not a repository). Every step that would commit is replaced by running the full test suite.

## Review Focus

1. Heisenberg vs Schrödinger consistency: Tr[O e^{tL}ρ] must equal Tr[e^{tL†}(O) ρ]. Test `test_heisenberg_matches_schrodinger`.
2. Non-Hermitian operators (charged sectors): evolution and charge projection must not assume Hermiticity. Tests `test_charge_projector_nonhermitian` and `test_evolve_nonhermitian_linear`.
3. Singular Γ in `protective_collective_rate`: Γ without a collective component must give 0, not raise. Test `test_protective_rate_singular`.
4. Bound functions at t = 0 and for γ_c → 0 must return finite values. `charged_observable_bound` with γ_c = 0 must reduce to the linear 2‖δH‖‖O‖t. Test `test_bounds_limits`.
5. Free-fermion module vs many-body JW at N = 4 with pairing and both dephasings: max deviation must be < 1e-8. Test `test_fermion_covariance_matches_manybody`.

---

### Task 1: Literature and data-availability check

**Files:** Create `docs/literature/novelty-check.md`

- [ ] **Step 1:** Run WebSearch on papers citing arXiv:2511.23392, 2311.14818 and 2404.11081, and on "collective dephasing Zeno symmetry-breaking error bound quantum simulation".
- [ ] **Step 2:** Use the web-fetch agent on the BEQS Zenodo record (doi 10.5281/zenodo.17750419) to get its file list, and check whether learned parameters (mean, covariance, Γ) are included.
- [ ] **Step 3:** Write the note: closest works, how each differs from Theorems A/B, open risks, and the data verdict (usable / raw-only).

### Task 2: Theory note

**Files:** Create `docs/theory/error-anatomy-theory.md` (English, paper-ready)

- [ ] **Step 1:** Setup and notation. Duhamel identities (one-step and two-step).
- [ ] **Step 2:** Theorem A with proof, and Corollary showing BEQS Eq. (A26) is recovered.
- [ ] **Step 3:** Theorem A2 (expected error: leading term + third-order remainder) and Theorem A3 (concentration, stated under a uniform-forgetting assumption).
- [ ] **Step 4:** Lemma on the protective collective rate, Theorems B1–B3 with proofs, and the SU(2) Casimir remark.
- [ ] **Step 5:** Proposition C with proof, and a list of the numerical checks that map to each statement.

### Task 3: Core simulator

**Files:** Create `pyproject.toml`, `src/qerr/__init__.py`, `src/qerr/spins.py`, `src/qerr/lindblad.py`, `tests/test_lindblad.py`

**Interfaces:**
- Produces:
  - `spins.pauli_op(n:int, site:int, kind:str)->np.ndarray`, with kind ∈ {'x','y','z','+','-'}
  - `spins.zvals(n)->np.ndarray[(2**n,n)]`
  - `spins.total_sz(n)->np.ndarray[(2**n,)]`
  - `spins.basis_state(n, bits:str)->np.ndarray`
  - `lindblad.OpenModel(H, Gamma, emission, jumps=())`, a frozen dataclass with property `n`
  - `lindblad.lindblad_rhs(model, X)->ndarray`
  - `lindblad.adjoint_rhs(model, X)->ndarray`
  - `lindblad.evolve(model, X0, times, heisenberg=False, rtol=1e-10, atol=1e-12)->ndarray[(T,D,D)]`
  - `lindblad.superop(model)->ndarray[(D²,D²)]`
  - `lindblad.expect(Xs, O)->ndarray[(T,)]` (complex)

- [ ] **Step 1: Write failing tests** in `tests/test_lindblad.py`:
  - `test_pauli_algebra`: σ^x σ^y = i σ^z on each site; σ^+ = |0⟩⟨1|.
  - `test_rhs_matches_superop`: n = 3, random H, PSD Γ, emission, one extra jump. Require ‖vec(lindblad_rhs(ρ)) − superop @ vec(ρ)‖ < 1e-12.
  - `test_trace_and_hermiticity_preserved`: evolve to t = 2. Require |Tr ρ − 1| < 1e-9, ‖ρ − ρ†‖ < 1e-9, and min eig > −1e-8.
  - `test_heisenberg_matches_schrodinger`: random O. Require |Tr[O ρ_t] − Tr[O_t ρ0]| < 1e-8.
  - `test_evolve_nonhermitian_linear`: evolve(X + iY) = evolve(X) + i·evolve(Y) within 1e-9.
- [ ] **Step 2:** Run `$PY -m pytest tests/test_lindblad.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**
  - Dephasing is applied elementwise as W∘X, with W_bb' = −½ (z_b − z_b')ᵀ Γ (z_b − z_b'). This W is self-adjoint, so the same W is used in both pictures.
  - Emission uses index maps: Schrödinger (σ^-ρσ^+)[idx1,idx1] = ρ[idx0,idx0]; Heisenberg (σ^+Aσ^-)[idx0,idx0] = A[idx1,idx1]. The anticommutator term is elementwise ½(p0_b + p0_b').
  - `evolve` integrates the flattened complex array with `solve_ivp(method='DOP853')`.
- [ ] **Step 4:** Run the tests. Expected: PASS.

### Task 4: Charge tools, model builders, perturbation ensembles

**Files:** Create `src/qerr/charge.py`, `src/qerr/models.py`, `tests/test_charge.py`

**Interfaces:**
- Consumes: Task 3 `spins`, `lindblad`.
- Produces:
  - `charge.charge_projector(n, X, q:int)->ndarray`
  - `charge.charge_decompose(n, X)->dict[int, ndarray]`
  - `charge.protective_collective_rate(Gamma, tol=1e-10)->float`
  - `models.xy_couplings(n, J0=1.0, alpha=1.0)->ndarray`
  - `models.xy_hamiltonian(n, J, B)->ndarray`
  - `models.Term`, a NamedTuple `(label:str, op:ndarray, abs_charge:int)`
  - `models.charge_basis_terms(n, max_range:int|None=None)->list[Term]`. Hermitian 1- and 2-local operators of definite |q|:
    - |q| = 0: z_i; zz; (xx+yy)/2; (xy−yx)/2
    - |q| = 1: x_i, y_i; xz, zx, yz, zy
    - |q| = 2: (xx−yy)/2; (xy+yx)/2
  - `models.delta_h(terms, g:ndarray)->ndarray`
  - `models.sample_coeffs(M, sigma, n_pairs, rng)->ndarray[(2*n_pairs, M)]` (antithetic: rows come in pairs g, −g)

- [ ] **Step 1: Write failing tests** in `tests/test_charge.py`:
  - `test_charge_projector_nonhermitian`: for random X, the sum over q of P_q X equals X, and [S^z, P_q X] = q P_q X.
  - `test_basis_terms_have_declared_charge`: every Term op is Hermitian, and its support is only in sectors ±abs_charge.
  - `test_xy_model_is_u1_covariant`: for the XY + z-field + Γ + emission superop, P_q(L(P_p X)) = 0 for q ≠ p.
  - `test_protective_rate_formula`: for Γ = 0.3·11ᵀ + diag(0.1,0.2,0.05), the rate is 1/(1ᵀΓ⁻¹1), and Γ − γ_c* 11ᵀ has min eig ≥ −1e-10.
  - `test_protective_rate_singular`: Γ = diag(1,0,1) gives 0.0.
  - `test_collective_damping_rate`: with H = 0 and Γ = γ_c 11ᵀ, an operator in B_q decays as exp(−2γ_c q² t).
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Projector: keep X_bb' where M_b − M_b' = q.
  - Protective rate:
    - If 1 is not in range(Γ) (residual > tol), return 0.
    - Otherwise return 1/(1ᵀ pinv(Γ) 1).
- [ ] **Step 4:** Run the full suite. Expected: PASS.

### Task 5: Bound functions

**Files:** Create `src/qerr/bounds.py`, `tests/test_bounds.py`

**Interfaces:**
- Produces:
  - `opnorm(A)->float`
  - `beqs_expected_bound(t, sigma, M, O_norm=1.)`, which is (exp(2t²σ²M) − 1)·‖O‖
  - `beqs_concentration_bound(t, sigma, M, delta=0.05, O_norm=1.)`, which is 2t√M·c·σ·‖O‖ with c = √(2 ln(2/δ))
  - `beqs_bias_bound(t, dH_norm, O_norm=1.)`, which is 2t‖δH‖‖O‖
  - `forgetting_bound(model, O, dH, times, n_fine=401)->ndarray`, which is ∫₀ᵗ ‖[δH, O_u]‖ du (trapezoid on a fine grid)
  - `charged_observable_bound(t, dH_norm, gamma_c, q, O_norm=1.)`
  - `zeno_bound(t, dH_norm, dHq_norms:dict[int,float], gamma_c, O_norm=1.)` = Σ_{q≠0} 4‖O‖‖δH‖‖δH_q‖·φ(a,t), with a = 2γ_c q² and φ(a,t) = t/a − (1−e^{−at})/a². Use a series form for at < 1e-4, since φ → t²/2. dHq_norms holds both signs of q.

- [ ] **Step 1: Write failing tests** in `tests/test_bounds.py`:
  - `test_bounds_limits`: all bounds are 0 at t = 0. `charged_observable_bound(t, 1, 1e-14, 1, 1)` ≈ 2t (rel 1e-6). `zeno_bound` with γ_c → 0 is finite and ≈ 2‖δH‖Σ‖δH_q‖t² (rel 1e-3).
  - `test_forgetting_bound_dominates`: for N = 3 with random weak δH, |Δ(t)| from `evolve` is ≤ `forgetting_bound` ≤ `beqs_bias_bound` at all grid times (+1e-9).
- [ ] **Step 2:** Run. FAIL. **Step 3:** Implement. **Step 4:** Run the full suite. PASS.

### Task 6: Experiments E1 (charge anatomy) and E2 (forgetting)

**Files:** Create `experiments/common.py`, `experiments/e1_charge_anatomy.py`, `experiments/e2_forgetting.py`. Outputs go to `results/e1_*.png|json` and `results/e2_*.png|json`.

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: `common.base_model(n, gamma_c, gamma_loc, gamma_em, B_inhom=0.1, seed=0)->OpenModel` and `common.save_json(path, dict)`.

- [ ] **Step 1: E1.**
  - Setup: N = 6, Néel |010101⟩. Perturbations have fixed norm ε = 0.05: P0 (|q| = 0 terms), P1 (|q| = 1), P2 (|q| = 2), each a random combination of `charge_basis_terms` rescaled to ‖δH‖ = ε. Observables: O0 = σ^z_2 σ^z_3 (q = 0); O1 = σ^x_2 (|q| = 1); O2 = (σ^x_2σ^x_3 − σ^y_2σ^y_3)/2 (|q| = 2). Sweep γ_c ∈ {0, 0.5, 2, 8}, t ∈ [0, 20].
  - Record |Δ(t)| with bound overlays (Theorem A, B2, B3).
  - Fit the late-time slope of |Δ| for (O0, P1/P2) against ε²/γ_c, and the plateau of (O1, P1) against ε/γ_c. Store both in JSON.
  - Acceptance:
    - (O0, P1/P2) first-order vanishing: |Δ| ∝ ε² (ratio between ε and 2ε ≈ 4 ± 0.4).
    - Every bound is ≥ the observed |Δ| at all times.
- [ ] **Step 2: E2.**
  - Setup: N = 6 with local dephasing 0.2 and emission 0.05 (primitive).
  - Show that |Δ(t)|, `forgetting_bound` and `beqs_bias_bound` give a saturating bound vs a linear one. Plot osc(O_u).
  - Acceptance: the forgetting bound saturates (its value at t = 20 differs from its value at t = 10 by < 5%).
- [ ] **Step 3:** Run both scripts. Check the PNGs visually and that the JSON acceptance flags are all true.

### Task 7: Experiments E3 (BEQS-like ensemble) and E4 (CP pathology)

**Files:** Create `experiments/e3_ensemble.py`, `experiments/e4_cp_pathology.py`.

- [ ] **Step 1: E3.**
  - Setup: N = 6, all `charge_basis_terms` with range ≤ 2, σ = 0.03, 100 antithetic pairs.
  - Compute:
    - expected error |E⟨O'⟩ − ⟨O⟩| and the 95% interval for O0, for (a) all terms, (b) |q| = 0 only, (c) |q| > 0 only
    - BEQS bounds with M = total count and with M = |q| = 0 count
    - γ_c ∈ {0, 2}
  - Acceptance: with γ_c = 2, the |q| > 0 contribution to the O0 expected error at t = 20 is smaller than with γ_c = 0.
- [ ] **Step 2: E4.**
  - Compare the analytic E[P_exc] = exp(−μt + s²t²/2) with Monte Carlo (Gaussian rates) and with log-normal rates (same mean and sd).
  - Acceptance: the Gaussian Monte Carlo exceeds 1 for t > 2μ/s², and the log-normal stays ≤ 1.

### Task 8: Free-fermion module and E5

**Files:** Create `src/qerr/fermions.py`, `tests/test_fermions.py`, `experiments/e5_free_fermion.py`.

**Interfaces:**
- Produces:
  - `bdg_matrix(h, Delta)->ndarray[(2N,2N)]`
  - `damping_matrix(gamma_loc:ndarray, gamma_c:float)->ndarray[(2N,2N)]`
  - `evolve_covariance(h, Delta, gamma_loc, gamma_c, G0, times)->ndarray[(T,2N,2N)]`. Here G = ⟨ΨΨ†⟩, Ψ = (c, c†), and dG/dt = −i[H_BdG, G] + Λ∘G.
  - `product_covariance(occ:ndarray)->ndarray`
  - Accessors `occupation(G)`, `hopping(G,i,j)`, `pairing(G,i,j)`.

- [ ] **Step 1: Write the failing test** `test_fermion_covariance_matches_manybody`:
  - Setup: N = 4. Build JW c_i, H = Σh c†c + ½Σ(Δc†c† + h.c.), jumps √γ_i n_i and √γ_c N_tot. Evolve via `lindblad.evolve` with the `jumps` field.
  - Compare ⟨c_i†c_j⟩ and ⟨c_i c_j⟩ against the covariance for t ∈ {0, 0.5, 1.5}. Require < 1e-8.
- [ ] **Step 2:** Run. FAIL. **Step 3:** Implement Λ:
  - normal blocks, i ≠ j: −½(γ_i+γ_j)
  - normal-block diagonal: 0
  - anomalous blocks, i ≠ j: −½(γ_i+γ_j) − 2γ_c
- [ ] **Step 4:** Run. PASS.
- [ ] **Step 5: E5.**
  - Setup: N ∈ {20, 50, 100, 200}, nearest-neighbour hopping 1, half-filled Néel occupation, ε = 0.05 random pairing (|q| = 2) or random on-site (|q| = 0), γ_c ∈ {0, 1, 4}, γ_loc = 0.05.
  - Observables: center occupation (q = 0) and center pairing (|q| = 2).
  - Acceptance:
    - |Δ| for center occupation under pairing perturbations is N-independent within 5% for N ≥ 50 at t = 10.
    - It decreases monotonically with γ_c.

### Task 9: Results summary and doc update

**Files:** Create `docs/results/summary.md` (Korean). Modify the spec §4 status column if needed.

- [ ] **Step 1:** Run the full test suite plus all experiment scripts fresh, and collect the JSON acceptance flags.
- [ ] **Step 2:** Write the summary:
  - findings per experiment, with figure paths
  - what is proven vs what is only checked numerically
  - honest limits (for example, gains are small when γ_c t ≲ 1)
  - next steps
