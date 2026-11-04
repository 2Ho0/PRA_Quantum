# Phase 2: Local Zeno Bound, BEQS Data Re-learning, PRA Draft — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Theorem B3 system-size independent, re-learn the BEQS 10-ion generator from the public raw shots and apply the charge anatomy to it, and assemble everything into a compiled PRA manuscript draft.

**Architecture:** Three tracks share the `qerr` library.
- (i) A Majorana engine for quadratic fermion operators. It computes the commutator-form certificate (Thm B3′) exactly at N ≤ 200.
- (ii) An ion-data pipeline: decode → basis inference → Pauli estimation → integral-method least squares → bootstrap.
- (iii) A revtex manuscript that pulls PDF figures from `results/`.

**Tech Stack:** Python 3.14, NumPy, SciPy, matplotlib, pytest (venv `C:\Users\mwill\.venvs\pra-qno`); TeX Live 2026 (`latexmk`, revtex4-2).

**Spec:** `docs/superpowers/specs/2026-11-04-phase2-local-data-draft-design.md` (phase-1 spec: `docs/superpowers/specs/2026-11-04-dissipative-error-anatomy-design.md`)

## Global Constraints

- No new Python dependencies beyond numpy/scipy/matplotlib/pytest.
- Raw data never enters git. It is downloaded to `data/raw/` (already ignored) by `data/fetch_beqs.py`, and the script verifies the Zenodo md5 checksums.
- Units:
  - Theory experiments: $J_0=1$.
  - Ion data: time in ms and rates in rad/ms internally; reported in rad/s, the BEQS convention.
- Bit conventions (ReadMe): qubit 1 is the least significant bit; bit 0 means eigenvalue +1 and bit 1 means −1. For the Néel data, "1 is spin down".
- Free-fermion conventions are those of `qerr.fermions` and E5. The collective jump is $\sqrt{\gamma_c}N$; pairing blocks decay at $2\gamma_c$.
- Figures:
  - Use the palettes in `experiments/common.py` (ORDINAL, CAT).
  - Every figure is saved as both PNG and PDF.
- Wording: BEQS is the work we build on, never a target. Inferred bases are described as inferred.

## Review Focus

1. **Settings whose t=0 means are ambiguous.** Basis inference must flag them by margin, never assign them silently. The count is reported in E6. *(Task 4 test `test_infer_bases_flags_ambiguous`.)*
2. **Bit-order or sign-convention slips.** These would silently corrupt every estimate. *(Task 4 tests `test_decode_readme_example`, `test_neel_decimal_decodes_alternating`.)*
3. **Learned Γ that is not PSD.** γ_c* must be computed on the PSD projection, and the projection size must be reported. *(Task 6 checks the projected-out weight and writes it to JSON.)*
4. **Quadrature error of the B3′ certificate.** The grid must be checked by halving the step for one configuration. *(Task 3 flag `certificate grid convergence < 2%`.)*
5. **The manuscript.** Undefined references or citations in the compiled PDF are a failure. *(Task 8 greps the log.)*

---

### Task 1: Majorana engine for quadratic fermion operators

**Files:** Create `src/qerr/quadratic.py`; Test `tests/test_quadratic.py`

**Interfaces:**
- Consumes: `qerr.fermions.bdg_matrix`, `damping_matrix` (conventions only).
- Produces:
  - `bdg_operator(h, Delta) -> M`: Σ M_ab Ψ_a Ψ_b† equals H = Σ h_ij c_i†c_j + ½Σ(Δ_ij c_i†c_j† + h.c.) up to a constant.
  - `number_operator(N, i) -> M`: the same representation for n_i.
  - `bdg_to_majorana(M) -> A`: complex antisymmetric 2N×2N. Σ M_ab Ψ_aΨ_b† = const + q(A), with q(A) = (i/4)Σ A_cd w_c w_d and c_j = (w_j + i w_{N+j})/2.
  - `majorana_to_bdg(A) -> M`: the inverse.
  - `quad_norm(A) -> float`: ¼(‖Re A‖₁ + ‖Im A‖₁). It is exact when A is real (Hermitian q) or imaginary.
  - `heisenberg_evolve(h, Delta, gamma_loc, gamma_c, A0, times) -> (T, 2N, 2N)`: e^{tL†} q(A0) = q(A(t)) + const. It integrates dM/dt = −i[H_BdGᵀ, M] + Λ∘M (the transpose of the covariance ODE) and converts back.
  - `charge_part(A, q) -> A_q`: q ∈ {0, 2, −2}, particle-number charge.

- [ ] **Step 1: Write the failing tests.** Compare against the many-body Jordan–Wigner construction for N = 3, reusing the helpers in `tests/test_fermions.py`.
  - `test_bdg_operator_matches_many_body`
  - `test_majorana_roundtrip_operator`: q(A) + const = Σ M Ψ Ψ†.
  - `test_commutator_rule`: [q(A1), q(A2)] = i·q([A1, A2]).
  - `test_quad_norm_exact_for_hermitian`: equals the many-body operator norm to 1e-10.
  - `test_heisenberg_duality`: the traceless parts of q(A(t)) and of the many-body e^{tL†}q(A0) agree to 1e-8. Use t = 0.7, local rates (0.1, 0.2, 0.3), γ_c = 0.4, and random h, Δ.
  - `test_charge_part_selects_pairing`: q(charge_part(A, 2)) changes particle number by +2.
- [ ] **Step 2: Run.** `pytest tests/test_quadratic.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `src/qerr/quadratic.py`.**
- [ ] **Step 4: Run.** `pytest -q`. Expected: all pass (16 old + 6 new).
- [ ] **Step 5: Commit** `feat: Majorana engine for quadratic fermion operators`.

### Task 2: Theory — support preservation, Theorem B3′, Corollary B3-loc

**Files:** Modify `docs/theory/error-anatomy-theory.md` (§0 table, new §4 items after B4, §6, §7).

**Interfaces:**
- Produces:
  - **Lemma B.2 (support).** P_q, every D_Γ and on-site terms preserve operator support. The LR constants of L_0 are those of its finite-range part.
  - **Theorem B3′.** $|\Delta(t)|\le\sum_{p>0}\int_{a+b\le t}e^{-2\gamma_cp^2b}\lVert[\delta H,e^{b\mathcal L_0^\dagger}(i[\delta H_{\pm p},O_a])]\rVert\,da\,db$, where $\delta H_{\pm p}=\delta H_p+\delta H_{-p}$.
  - **Corollary B3-loc.** Under (LR) with constants (c, v, ξ), for δH = Σ_x h_x of range ℓ and |h_x| ≤ ε, |h_{x,±p}| ≤ ε_p, an N-independent bound. It is written with light-cone volumes and has the 1D form $C\,\varepsilon\varepsilon_p\,\mathrm{osc}(O)(|A|+2vt)^2\,t/(2\gamma_cp^2)$.

- [ ] **Step 1: Write the statements and proofs.**
  - Proofs: shell decomposition $O_a=\sum_k\Omega_k$ with restricted evolutions; the outer commutator is bounded with LR on strictly supported pieces.
- [ ] **Step 2: Self-check.**
  - Every inequality uses only contraction, Russo–Dye, Lemma B.0 or (LR).
  - The constants are dimensionally consistent.
- [ ] **Step 3: Commit** `docs: local Zeno bound (B3', B3-loc) and support lemma`.

### Task 3: E7 — certificate vs global bound vs N

**Files:** Create `experiments/e7_local_zeno.py`. Output: `results/e7_local_zeno.{png,pdf,json}`.

**Interfaces:**
- Consumes:
  - `e5_free_fermion.couplings`, `GAMMA_LOC`, `EPS` (same model).
  - Task 1 functions.
  - `qerr.fermions.evolve_covariance`.
- Produces:
  - `certificate_table(N, gc, t_max, n) -> f[i, j]`, where f = ‖[δH, e^{b_jL_0†}(i[δH_{±2}, O_{a_i}])]‖ and O = n_c.
  - `triangle_integral(f, dt, rate) -> (n+1,)`: the cumulative certificate on t_k = k·dt.

- [ ] **Step 1: Implement** for N ∈ {20, 50, 100, 200}, γ_c ∈ {0, 1, 4}, t ≤ 10, grid dt = 0.25, parallelized as in E5.
- [ ] **Step 2: Run** `python e7_local_zeno.py`. Expected acceptance flags:
  - certificate ≥ observed |Δn_c| at all grid times.
  - certificate spread over N ∈ {100, 200} at t = 10 is < 5%.
  - global B3 ratio N=200/N=50 is within 16 ± 25%.
  - certificate decreases monotonically in γ_c at t = 10.
  - certificate grid convergence < 2% (dt vs dt/2 at N=50, γ_c=1).
- [ ] **Step 3: Render and inspect the figure. Commit** `feat: E7 local Zeno certificate vs N`.

### Task 4: Ion-data I/O, basis inference, Pauli estimation

**Files:** Create `src/qerr/iondata.py` and `data/fetch_beqs.py`; Test `tests/test_iondata.py`.

**Interfaces:**
- Produces:
  - `decode_shots(dec, n) -> int8 ±1 array (..., n)`
  - `load_10ion(root, state) -> (times_ms (21,), spins (21, 200, 200, 10))`
  - `load_neel51(path) -> (times_ms (11,), spins (11, 1000, 51))`
  - `infer_bases(m1, m2, sem1, sem2) -> (bases (200, 10) int8 in {0:X, 1:Y, 2:Z}, margin (200, 10))`
  - `pauli_means(spins, bases, strings) -> (mean (T, S), sem (T, S), n_settings (S,))`. Each string is a tuple of (site, letter) pairs.
  - `per_setting_means(spins, bases, string) -> (setting_idx, (n_set, T))`, used by the bootstrap.
- Decision rule in `infer_bases`:
  - Y if m2 > 0.5; otherwise X if m1 > 0.2; otherwise Z.
  - margin = distance to the active threshold in units of the larger SEM.

- [ ] **Step 1: Failing tests.**
  - `test_decode_readme_example`: 101 → spins (−1, +1, −1, +1, +1, −1, −1, +1, +1, +1) for qubits 1..10.
  - `test_neel_decimal_decodes_alternating`: 1501199875790165, n = 51 → qubit 1 = −1, alternating.
  - `test_infer_bases_recovers_synthetic`: Bloch vectors (sin 0.33π, 0, −cos 0.33π) and (0, 1, 0), 200 shots, 2% flips → 100% correct.
  - `test_infer_bases_flags_ambiguous`: an injected mean at the threshold → margin < 3.
  - `test_pauli_means_unbiased`: a synthetic product state gives estimates within 4 SEM of the exact values.
- [ ] **Step 2: Run, expect FAIL. Implement. Run `pytest -q`, expect all pass.**
- [ ] **Step 3: Write `data/fetch_beqs.py`** (urllib, md5 check against record.json values hard-coded). Run it. Expected: 4 files verified in `data/raw/`.
- [ ] **Step 4: Commit** `feat: BEQS data loader, basis inference, Pauli estimator`.

### Task 5: Pauli algebra and integral-method learner

**Files:** Create `src/qerr/pauli.py` and `src/qerr/learning.py`; Test `tests/test_learning.py`.

**Interfaces:**
- `pauli.py`
  - `pmul(p, q) -> (phase complex, r)`
  - `comm_i(p, q) -> (coef float, r) | (0.0, None)`: returns i[p, q] = coef·r.
  - `weight(p)`
- `learning.py`
  - `Ansatz(n, h_terms: list[pstring], deph_pairs: list[(i, j)], emission_sites: list[int])` with `.n_params`.
  - `adjoint_terms(O, ansatz) -> list[(k, coef, pstring | ())]`: L†(O) = Σ θ_k coef S, where `()` is the identity.
  - `integral_system(obs, ansatz, series, times, t_learn) -> (A, y, rows)`. `series[(state, pstring)]` is a (T,) array; integrals use a cubic spline on t ≤ t_learn.
  - `fit(A, y) -> θ`
  - `to_openmodel(θ, ansatz) -> OpenModel`
- Parameter conventions:
  - H = Σ_k θ_k P_k.
  - D_Γ(X) = Σ_ij Γ_ij(Z_i X Z_j − ½{Z_jZ_i, X}) with Γ symmetric; the parameters are Γ_ii and Γ_ij for i < j, each pair entering twice.
  - Emission: rate θ_k on σ⁻ = |1⟩⟨0|, the `OpenModel` convention.

- [ ] **Step 1: Failing tests.**
  - `test_comm_i_single_site`: i[X, Y] = −2Z, i[Y, Z] = −2X, i[Z, X] = −2Y.
  - `test_adjoint_terms_match_superop`: n = 3, random θ. Σ coef⟨S⟩_ρ equals Tr[L†(O)ρ] from `qerr.lindblad.adjoint_rhs` for random ρ and every weight-≤2 O.
  - `test_recovers_parameters_exact_data`: n = 4 XY + fields + Γ + emission, exact expectations on dt = 0.01 → max relative error < 1e-3.
  - `test_coarse_grid_bias_reported`: on the coarse 0.5-unit grid the error is < 5%. The measured value is printed.
- [ ] **Step 2: Run, expect FAIL. Implement. Run `pytest -q`, expect all pass.**
- [ ] **Step 3: Commit** `feat: Pauli algebra and integral-method Lindbladian learner`.

### Task 6: E6 — learning on BEQS data, validation, symmetry diagnostics, error budget

**Files:** Create `experiments/e6_ion_data.py`. Output: `results/e6_ion_learning.{png,pdf}`, `results/e6_symmetry.{png,pdf}`, `results/e6_ion_data.json`.

- [ ] **Step 1: Learning.**
  - Ansatz: XX, YY, XY, YX on all pairs; X, Y, Z on all sites; full Γ; per-site emission.
  - Observables: all weight-1 and weight-2 Paulis. States 1 and 2, t ≤ 1.5 ms.
  - Bootstrap B = 100 over settings.
  - Report:
    - J^sym = (XX + YY)/2 vs distance with a power-law fit (J0, α), and K = (XX − YY)/2.
    - The q = 2 and q = 1 parts.
    - Γ projected to PSD (projected weight), γ_c* (Lemma B.1) with a bootstrap CI.
    - The ambiguous-setting count.
- [ ] **Step 2: Validation.** Simulate the central model to 10 ms for both states. Report the RMS model–data deviation on the validation window (t > 1.5 ms) against the mean SEM.
- [ ] **Step 3: 51-ion Néel diagnostics.** ⟨M⟩(t), Var M(t), staggered magnetization, P(M, t), with bootstrap CIs and linear-fit slopes.
- [ ] **Step 4: Error budget for a Néel quench in the learned 10-ion model.** Charged part (B3′, δH = bootstrap-scale charge-breaking terms) vs symmetric part (Thm A) vs BEQS Eqs. (11), (12), (14), up to 10 ms.
- [ ] **Step 5: Render, inspect, commit** `feat: E6 BEQS 10-ion re-learning and 51-ion symmetry diagnostics`.

### Task 7: Vector figures for the paper

**Files:** Modify `experiments/common.py` (add `save_figure(fig, stem) -> Path`, writing `results/<stem>.png` and `.pdf`) and `experiments/e1`–`e7` (use it).

- [ ] **Step 1: Implement and rerun** every experiment. Expected: all acceptance flags unchanged from `results/*.json` before the rerun, and a PDF exists for every PNG.
- [ ] **Step 2: Commit** `chore: PDF figures for all experiments`.

### Task 8: PRA manuscript draft

**Files:** Create `paper/main.tex`, `paper/refs.bib`, `paper/README.md` (build instructions).

- [ ] **Step 1: Write the manuscript.**
  - revtex4-2 `pra,twocolumn,superscriptaddress`.
  - Sections: Introduction · Setting · Heisenberg forgetting (A, A2) · Symmetry-resolved anatomy (B0–B4, B3′, B3-loc) · Numerics (E1–E5, E7) · Application to trapped-ion data (E6) · Discussion · Appendices (proofs, Prop. C with E4, numerical methods, data processing).
- [ ] **Step 2: Build** `latexmk -pdf main.tex` in `paper/`. Expected: exit 0, and `grep -c "undefined" main.log` = 0.
- [ ] **Step 3: Commit** `docs: PRA manuscript draft`.

### Task 9: Documentation and handoff

**Files:** Modify `docs/results/summary.md`, `docs/literature/novelty-check.md`, `docs/theory/error-anatomy-theory.md` (§6, §7). Create `README.md` and `docs/data/beqs-data-notes.md` (data structure, inferred conventions, and a confirmation e-mail draft to M. K. Joshi, not sent).

- [ ] **Step 1: Write the documents. Run `pytest -q`. Commit** `docs: phase-2 summary, data notes, README`.
- [ ] **Step 2: Push the branch.** Give the user the PR link.
