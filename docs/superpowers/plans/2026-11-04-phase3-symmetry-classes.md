# Phase 3: Weak/Strong Symmetry Classes, Certified γ_c, Validation Deviation, Hydrodynamic Forgetting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the symmetry-resolved error anatomy from Hamiltonian to dissipative imperfections (weak vs strong U(1) symmetry). Test it on the BEQS public data with an extended learning ansatz and a model-conditional certified collective rate. Explain the 10-ion validation deviation. Derive and check the √t law of Heisenberg forgetting under charge conservation.

**Architecture:** Library additions:
- superoperator charge parts in `qerr.charge`;
- single-site Kossakowski parameters, structured and PSD-constrained fits in `qerr.learning`.

Four experiment scripts consume them:
- E8 (N = 6 exact);
- E9 (10-ion data, extended ansatz, γ_c certification, shadow feasibility);
- E6c (validation variants);
- E10 (free fermions, N = 400).

The theory goes into a local-only note and into the git-ignored manuscript.

**Tech Stack:** Python 3.14, NumPy, SciPy, matplotlib, pytest (venv `C:\Users\mwill\.venvs\pra-qno`); TeX Live 2026.

**Spec:** `docs/superpowers/specs/2026-11-04-phase3-symmetry-classes-design.md` (phase-2 spec: `docs/superpowers/specs/2026-11-04-phase2-local-data-draft-design.md`)

## Global Constraints

- **Dependencies.** None new beyond numpy/scipy/matplotlib/pytest.
- **Units.** Theory experiments use J₀ = 1. Ion data use ms and rad/ms internally and are reported in rad/s.
- **Conventions.** `qerr.spins`: |0⟩ = up, site 0 leftmost, σ⁻ = |1⟩⟨0|. Charges: [S^z, σ⁺] = σ⁺. Collective dephasing damps charge q at 2γ_c q².
- **Git.**
  - `paper/` is never committed; this also covers the new theory notes, which live in `paper/notes/`.
  - Stage explicit paths only.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Manuscript.**
  - Compile the PDF after every `.tex` change.
  - No bullets: use enumerate numbers or `\BfPara`.
  - Keep the template authors.
  - Data wording: "measured by Kraft et al. [cite], analysed here from the public data".
- **Figures.** Palettes from `experiments/common.py`. Every figure is saved as PNG and PDF.
- **Claims.** "Certified" only with its stated condition (the structured model).

## Review Focus

1. **Superoperator charge sign.** S_Q must map B_p → B_{p+Q} in the Schrödinger picture. A sign slip would swap the Q = ±2 labels silently. *(Task 1 `test_superop_charge_maps_sectors`.)*
2. **Heisenberg action of the new dissipator parameters.** T-type (antisymmetric) terms carry a factor i and an anticommutator. *(Task 2 `test_site_noise_adjoint_matches_superop`, `test_site_noise_reproduces_emission`.)*
3. **Degenerate parameters.** The extended ansatz must not keep the old emission columns, which are collinear with S_XX, S_YY, T_XY. *(Task 2 `test_site_noise_ansatz_full_rank`.)*
4. **PSD projection metric.** Off-diagonal Γ parameters enter twice in ‖Γ‖_F. The projected-gradient fit must scale them by √2 or it is not a Euclidean projection. *(Task 2 `test_psd_fit_is_feasible_and_optimal`.)*
5. **Finite-size contamination of the E10 power laws.** Fit only where the kernel has not reached the chain ends. *(Task 6 acceptance flag `kernel edge weight < 1e-3 in fit window`.)*

---

### Task 1: Superoperator charge decomposition

**Files:** Modify `src/qerr/charge.py`; Test `tests/test_charge.py`

**Interfaces:**
- Produces:
  - `superop_charge_part(n, S, Q) -> ndarray`: S is a column-stacking (D², D²) superoperator, the `qerr.lindblad.superop` convention. It keeps the entries between |a⟩⟨b| and |c⟩⟨d| with (M_a − M_b) − (M_c − M_d) = Q.
  - `superop_charges(n, S, tol=1e-12) -> dict[int, float]`: Q ↦ Frobenius norm of S_Q, for the nonzero parts.

- [ ] **Step 1: Write the failing tests.**
  - `test_superop_charge_maps_sectors`: for random S at n = 2, S_Q(P_p X) ∈ B_{p+Q}, and Σ_Q S_Q = S.
  - `test_dissipator_charges`: D[σ⁻], D[σ⁺] and D[Z] have charges {0}; −i[X₀, ·] has {±1}; D[X] has {0, ±2}; and D[X]_0 = D[σ⁺] + D[σ⁻] to 1e-12.
- [ ] **Step 2: Run.** `pytest tests/test_charge.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** with vec index a + D·b ↔ |a⟩⟨b| and charge M_a − M_b from `spins.total_sz`.
- [ ] **Step 4: Run.** `pytest -q`. Expected: all pass (34 + 2).
- [ ] **Step 5: Commit** `feat: superoperator charge decomposition`.

### Task 2: Learning — single-site Kossakowski parameters, structured and PSD-constrained fits

**Files:** Modify `src/qerr/learning.py`; Test `tests/test_learning.py`

**Interfaces:**
- Consumes: `qerr.pauli.pmul`, `qerr.lindblad.superop`.
- Produces:
  - `Ansatz.noise_sites: list` (default `[]`). It appends 8 parameters per site, in the order `NOISE_KEYS = ("SXX", "SYY", "SXY", "TXY", "SXZ", "TXZ", "SYZ", "TYZ")`.
    - Single-site dissipator: D(ρ) = Σ_{A,B∈{X,Y,Z}} K_AB(AρB − ½{BA,ρ}), with K = S + iT (S real symmetric, T real antisymmetric).
    - K_ZZ stays in Γ.
    - Heisenberg action: S_AA → AOA − O; S_AB → BOA + AOB; T_AB → i(BOA − AOB − {BA,O}).
  - `noise_charges() -> dict`: key ↦ superoperator charge set.
    - Q = 0: SXX+SYY, TXY.
    - Q = ±2: SXX−SYY, SXY.
    - Q = ±1: SXZ, TXZ, SYZ, TYZ.
  - `site_kossakowski(theta, ans) -> (n, 3, 3) complex`.
  - `flip_rates(K) -> (emission, absorption)`: emission = SXX+SYY+2TXY, absorption = SXX+SYY−2TXY.
  - `to_openmodel` raises `NotImplementedError` for noise sites. Fits with noise sites pass a precomputed `defect`.
  - `structured_map(ans) -> P`: θ_full = P θ_s. In θ_s, the dephasing block (Γ_ii, Γ_ij) is replaced by (γ_c, g_1..g_n), with Γ = γ_c 11ᵀ + diag(g). Other parameters pass through.
  - `fit_bounded(A, y, lower) -> theta` via `scipy.optimize.lsq_linear`.
  - `fit_psd(A, y, ans, n_iter=20000, tol=1e-12) -> theta`: least squares subject to Γ(θ) ⪰ 0. Profile out the non-Γ parameters (Schur complement), run FISTA on the √2-scaled Γ coordinates with an eigenvalue-clipping projection, then back-substitute.

- [ ] **Step 1: Write the failing tests.**
  - `test_site_noise_adjoint_matches_superop`: n = 2, a random PSD 3×3 K per site (K_ZZ given to Γ) and a random state ρ. For every 1- and 2-site Pauli O, the adjoint expansion Σ c⟨S⟩ equals Tr[O L(ρ)] from `superop` with jumps from the eigen-decomposition of K, to 1e-10.
  - `test_site_noise_reproduces_emission`: (SXX, SYY, TXY) = (γ/4, γ/4, γ/4) gives the same Σ c⟨S⟩ as an emission site.
  - `test_site_noise_ansatz_full_rank`: the integral system of a 2-qubit model with noise sites, without emission sites, has full column rank.
  - `test_structured_fit_recovers_gamma_c`: exact series of a 3-qubit model with Γ = 0.3·11ᵀ + diag(0.1, 0.2, 0.05). `fit_bounded` on A·P recovers γ_c and g to 1e-6.
  - `test_psd_fit_is_feasible_and_optimal`: with a feasible truth and exact data it returns the truth. With data pushed infeasible, the output Γ is PSD (min eig ≥ −1e-9) and its objective is ≤ that of the clipped unconstrained solution.
- [ ] **Step 2: Run.** `pytest tests/test_learning.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: dissipator charge parameters and constrained fits for Lindbladian learning`.

### Task 3: E8 — three classes of imperfection (N = 6, exact)

**Files:** Create `experiments/e8_symmetry_classes.py` → `results/e8_symmetry_classes.{json,png,pdf}`

**Interfaces:**
- Consumes: `common.base_model(6, γ_c, 0.05, 0.0)`, `neel`, `OpenModel(jumps=...)`, `evolve`, `superop`, `superop_charge_part`.

**Setup.**
- κ ∈ {0.02, 0.04}; γ_c ∈ {0, 0.25, 0.5, 1, 2, 4}; t ∈ [0, 20], dt = 0.25.
- Imperfections:
  - (I) δH = κΣX_j;
  - (II) jumps √κ σ⁺_j and √κ σ⁻_j;
  - (III) jumps √κ X_j.
- Observables: staggered magnetization m_s = Σ(−1)^j Z_j / N, and the leakage P(M ≠ 0).
- Bound: the dissipative anatomy theorem (B6, global form) for class III,
  sym(t) = ∫‖V_0†(O_u)‖ du (dense superop on vec O_u) + ‖V_ch‖_{1→1}·Σ_Q ν_Q·osc(O)·φ(2γ_c Q², t).
  - V_ch = V_{+2} + V_{−2} with V_{±2}(ρ) = κΣ_j σ^±_j ρ σ^±_j, so ‖V_ch‖_{1→1} ≤ 2Nκ.
  - V_{±2}†(A) = κΣ_j σ^±_j A σ^±_j annihilates 1, so ν_{±2} ≤ Nκ.

**Acceptance flags (JSON).** Ratios at t = 10 for γ_c ≥ 0.5:
- class I: ratio in [3.5, 4.5];
- class II: ratio in [1.8, 2.2], and its γ_c spread < 1e-6;
- class III: ratio in [1.8, 2.2];
- remainder (III − II): ratio in [3.5, 4.5] and decreasing in γ_c;
- the B6 bound ≥ |Δ_III| at all grid times.

- [ ] **Step 1:** Write the script.
- [ ] **Step 2:** Run it and read the flags.
- [ ] **Step 3:** View the PNG.
- [ ] **Step 4:** Commit `feat: E8 weak/strong symmetry classes of imperfections`.

### Task 4: E9 — dissipator charges, certified γ_c and shadow feasibility on the BEQS data

**Files:** Create `experiments/e9_symmetry_data.py` → `results/e9_symmetry_data.{json,npz,png,pdf}`

**Interfaces:**
- Consumes: the E6a pipeline functions; `results/e6_ion_learning.npz` (θ, for the defect model); the Task-2 API.

**Steps of the script.**
- (a) **Extended ansatz.** E6a ansatz with `emission_sites=[]` and `noise_sites=range(10)`. Use the defect from the E6a θ, then fit and bootstrap 100 times. Report:
  - z-scores of the Q = ±1 and Q = ±2 dissipator combinations per site;
  - emission and absorption rates;
  - the summed absorption + emission rate with a 95% upper bound.
- (b) **Structured dephasing.** Fit with g ≥ 0 bounds (flip rates unconstrained) and bootstrap. Report:
  - γ_c, its sd and its 5th percentile (the 95% one-sided lower bound);
  - Zeno factors φ(2γ_c q², t)/(t²/2) at t = 10 ms, q = 1, 2, for the point estimate and the lower bound.
- (c) **PSD-constrained full Γ.** Use `fit_psd` and the bootstrap distribution of the Lemma B.1 γ_c*, showing its fragility.
- (d) **Shadow feasibility.** Within-setting Brydges purity estimator of subsystems k = 1…10 at t = 0 (both states). Report its mean and its sd over settings. Compare with the setting-randomness floor sd ≈ √((∏_j m_j − 1)/S), where S = 200 and m_j = 0.75 + 0.75 Σ_a n_{j,a}⁴ is the per-site second moment for a pure state with Bloch vector n_j (m = 1.5 on a basis axis).
- (e) **51-ion strong-symmetry numbers.** From `results/e6_symmetry.json`: the balanced sector-changing rate per ion, (varM slope − ⟨M⟩-drift)/51, in s⁻¹.

- [ ] **Step 1:** Write the script.
- [ ] **Step 2:** Run it in the background (log with `python -u`) and read the JSON.
- [ ] **Step 3:** View the PNG.
- [ ] **Step 4:** Commit `feat: E9 dissipator charges and certified collective rate from the BEQS data`.

### Task 5: E6c — validation deviation

**Files:** Create `experiments/e6c_validation.py` → `results/e6c_validation.{json,npz,png,pdf}`

**Variants.**
- W15, W30, W50: full ansatz with learning windows of 1.5, 3, 5 ms (`learn`, 3 defect rounds).
- S15, S30: structured dephasing with g ≥ 0 (fit_bounded with the defect of the matching W).
- Not run: uniform B_z. It is rejected by the E6a bootstrap (z up to 8.5), and the JSON records that.

**Validation.** Every variant uses `physical()` and is validated on t > 5 ms, the common window.

**Outputs.**
- Residuals split by charge-definite combinations:
  - single-site Z;
  - single-site X/Y;
  - pair ZZ;
  - pair flip-flop (XX+YY)/2 and (XY−YX)/2 (q = 0);
  - pair (XX−YY)/2 and (XY+YX)/2 (q = ±2);
  - pair XZ-type (q = ±1).
- The C0 and C2 trajectories of the (5,6) pair (state 2) for data and variants.
- The learned charge-0 nn-pair dephasing rates 2(Γ_ii + Γ_jj − 2Γ_ij) per variant.

- [ ] **Step 1:** Write the script.
- [ ] **Step 2:** Run it in the background and read the JSON.
- [ ] **Step 3:** View the PNG.
- [ ] **Step 4:** Commit `feat: E6c charge-resolved validation of the re-learned generator`.

### Task 6: E10 — hydrodynamic forgetting (√t law)

**Files:** Create `experiments/e10_hydro_forgetting.py` → `results/e10_hydro_forgetting.{json,png,pdf}`

**Setup.**
- NN hopping chain (−1), N = 400, local number dephasing g = 1, γ_c = 0.
- δH = ε Σ_k (c_k†c_{k+1} + h.c.), uniform miscalibration, with ε = 0.01.
- Observables at the centre c:
  - monopole n_c;
  - dipole n_c − n_{c+1};
  - quadrupole n_{c−1} − 2n_c + n_{c+1};
  - bond current.
- Integrand f_O(u) = ‖[δH, e^{uL†}O]‖ on a log grid u ∈ [0.05, 1000]. Fit power laws on the window where the kernel weight within 20 sites of either chain end is < 1e-3.
- Expected slopes: −1/2, −1, −3/2, and −1 for the current.
- True error |Δ(t)| for density-wave initial states (n_k = ½ + 0.4 cos(2πk/λ), λ ∈ {20, 40, 80, 160}) at the crest: the peak error is roughly λ-independent while the bound grows as √t.

**Acceptance flags.**
- Monopole slope in [−0.6, −0.4]; dipole in [−1.15, −0.85]; quadrupole ≤ −1.3.
- Bound ≥ true error at all times.
- Peak true error spread across λ within a factor 2.

- [ ] **Step 1:** Write the script.
- [ ] **Step 2:** Run it and read the flags.
- [ ] **Step 3:** View the PNG.
- [ ] **Step 4:** Commit `feat: E10 hydrodynamic forgetting under charge conservation`.

### Task 7: Theory note (local only)

**Files:** Create `paper/notes/phase3-theory.md` (git-ignored).

**Content.**
- Lemma C.0 (superoperator charge).
- Definitions: weak and strong symmetry; Proposition C.1 (strong ⇒ P(M) stationary; covariant flips break it at first order).
- Theorem B6 (dissipative anatomy), with its commutator form.
- Corollary B6′ (leakage diagnostic: γ_c dependence separates classes).
- Remark (concavity of γ_c* ⇒ plug-in biased low).
- Theorem E (hydrodynamic forgetting, multipole orders) and Remark E.1: true error is time-uniform for density waves, which leaves an open problem.
- Remark E.2: covariant flips restore forgetting.

### Task 8: Manuscript, docs, push

**Files:**
- Modify (local only): `paper/main.tex`, `paper/refs.bib`.
- Modify (committed): `docs/results/summary.md`, `README.md`.

**Steps.**
- [ ] **Step 1:** Add the theory and E8–E10/E6c sections to the manuscript. Fix the E6b wording (strong symmetry), and update the abstract, contributions and outlook. Compile after every edit, and check the log for undefined references.
- [ ] **Step 2:** Update the docs and commit them with explicit paths.
- [ ] **Step 3:** Run the full test suite.
- [ ] **Step 4:** Do the final self-review.
- [ ] **Step 5:** Push the branch `phase3-symmetry-classes`.
