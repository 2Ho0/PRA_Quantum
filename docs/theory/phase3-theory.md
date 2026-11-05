# Phase-3 theory note: weak vs strong symmetry, dissipative imperfections, hydrodynamic forgetting

Research note, phase 3 (2026-11-04). This note continues `docs/theory/error-anatomy-theory.md`, using the same notation:
- S^z = ½Σσ^z_i;
- sectors B_q = {X : [S^z, X] = qX};
- Ad_φ(X) = e^{iφS^z} X e^{−iφS^z};
- collective dephasing D_c(γ_c) = −2γ_c ad²_{S^z}, which acts as −2γ_c q² on B_q;
- φ(a, t) = t/a − (1 − e^{−at})/a².

Every statement is checked numerically by E8, E9, E6c or E10 (§6).

## C. Weak and strong symmetry; imperfections of any kind

### Definition C.0 (superoperator charge)

For a superoperator S on operators and Q ∈ ℤ,

S_Q := (1/2π) ∫₀^{2π} e^{−iQφ} Ad_φ ∘ S ∘ Ad_{−φ} dφ.

### Lemma C.0

1. S = Σ_{|Q|≤2n} S_Q, and S_Q(B_p) ⊆ B_{p+Q}.
2. The dual satisfies (S_Q)† = (S†)_Q, with Tr[A S(X)] = Tr[S†(A) X]. So S_Q† also raises the charge by Q.
3. If S is Hermiticity-preserving, then S_Q(X)† = S_{−Q}(X†). Hence S_Q + S_{−Q} is Hermiticity-preserving.
4. For A ∈ B_a and B ∈ B_b, the map X ↦ AXB† has charge a − b.
   - The commutator −i[H_q, ·] has charge q.
   - Write a dissipator in a jump basis of definite charges, D(ρ) = Σ_ab K_ab (L_a ρ L_b† − ½{L_b†L_a, ρ}). Its charge-Q part collects the terms with q_a − q_b = Q, since L_b†L_a ∈ B_{q_a−q_b}.

*Proof.*
1. Ad_φ acts on B_p as e^{ipφ}. For X ∈ B_p, Ad_φ S Ad_{−φ} X = Σ_r e^{i(r−p)φ}(SX)_r. The Fourier coefficient selects r = p + Q.
2. (Ad_φ)† = Ad_{−φ}, and the pairing Tr[A·] is bilinear.
3. Ad_φ commutes with the adjoint X ↦ X†, up to φ → −φ.
4. Ad_φ(AXB†) = e^{i(a−b)φ} A Ad_φ(X) B†.

□ (Checked: `tests/test_charge.py::test_superop_charge_maps_sectors`.)

### Definition C.1

- L is **weakly U(1)-symmetric** (covariant) if L = L_0, equivalently [L, Ad_φ] = 0 for all φ.
- L is **strongly symmetric** if it has a GKSL form whose H and jumps all commute with S^z (Buča and Prosen, NJP 14, 073007 (2012)).

### Proposition C.1

1. Strong ⇒ weak.
2. Under strong symmetry, L†(f(S^z)) = 0 for every f. So the magnetization distribution P_t(M) = Tr[Π_M ρ_t] is stationary for every initial state.
3. A dissipator whose jumps have definite nonzero charge, such as σ^±_j (emission, absorption), is covariant but changes P(M) at first order:
   - D[σ^−_j]†(σ^z_j) = −(1 + σ^z_j);
   - D[σ^+_j]†(σ^z_j) = 1 − σ^z_j.
4. X dephasing is not covariant: D[X_j] = D[σ^+_j] + D[σ^−_j] + V_{+2} + V_{−2}, with V_{±2}(ρ) = σ^±_j ρ σ^±_j.

*Proof.*
1. The jumps and H commute with every function of S^z, hence with Ad_φ.
2. i[H, f] = 0, and L_k†fL_k − ½{L_k†L_k, f} = L_k†L_k f − L_k†L_k f = 0.
3. Item 4 of Lemma C.0, and a direct computation.
4. Use X = σ^+ + σ^−, X² = 1 and σ^+σ^− + σ^−σ^+ = 1.

□ (Checked: `test_dissipator_charges`.)

**Single-site noise in the charge basis.** Take the jumps (σ^+, σ^−, σ^z) with charges (+1, −1, 0). In the Pauli basis write K = S + iT, with S real symmetric and T real antisymmetric. The charge-definite combinations are:

| Q | Kossakowski entries | Pauli combinations | Physical meaning |
|---|---|---|---|
| 0 | K_{++}, K_{−−}, K_{zz} | S_XX + S_YY, T_XY, S_ZZ | absorption and emission rates S_XX + S_YY ∓ 2T_XY; dephasing |
| ±1 | K_{±z}, K_{z∓} | S_XZ, T_XZ, S_YZ, T_YZ | |
| ±2 | K_{+−}, K_{−+} | S_XX − S_YY, S_XY | |

(Checked: `test_noise_charges_match_superop`.)

### Classification

Relative to a covariant reference L = L_0 + D_c(γ_c), write the imperfection as V = L′ − L = V_0 + V_ch.

- **Class I.** Charged Hamiltonian errors, V = −i[δH_q, ·] with q ≠ 0.
- **Class II.** Covariant imperfections, V = V_0.
  - Hamiltonian examples: δH_0.
  - Dissipative examples: σ^± flips, σ^z dephasing.
  - Dissipators with charged jumps break only the strong symmetry.
- **Class III.** The charged parts V_Q (Q ≠ 0) of a dissipator. Examples: the γ_X − γ_Y imbalance and Re Γ_XY of X/Y noise (Q = ±2), and XZ-type cross terms (Q = ±1).

### Theorem B6 (anatomy of general imperfections)

**Assumptions.**
- L = L_0 + D_c with L_0 covariant GKSL, and L′ = L + V is GKSL.
- The twirled implemented generator L′_0 := L + V_0 can be written as L̃_0 + D_c, with L̃_0 covariant GKSL. This holds whenever V is itself a Hamiltonian commutator or a GKSL dissipator. More generally, it holds whenever γ_c is a protective rate of L′_0 (Lemma B.1).
- ρ_0, O ∈ B_0.

**Statement.**

|Δ(t)| ≤ ∫₀^t ‖V_0†(O_u)‖ du + ‖V_ch‖_{1→1} osc(O) Σ_{Q≠0} ν_Q φ(2γ_cQ², t),

with O_u = e^{uL†}O and ν_Q := sup_{X=X†} ‖V_Q†(X)‖/osc(X). The supremum is finite because V_Q†(1) = 0.

**Commutator form.** The second term may be replaced by

Σ_{p>0} ∫∫_{a+b≤t} e^{−2γ_c p² b} ‖V_ch†(e^{bL̃_0†} V_{±p}†(O′_a))‖ da db,

with O′_a = e^{aL′_0†}O and V_{±p} = V_p + V_{−p}.

**Hamiltonian case.** For V = −i[δH, ·], ‖V_ch‖_{1→1} ≤ 2‖δH_ch‖ and ν_q ≤ 2‖δH_q‖, which is Theorem B4.

*Proof.*
1. L is covariant, so twirl(L′) = L + V_0 = L′_0. This is GKSL, as an average of unitarily conjugated GKSL generators.
2. Split Δ = (⟨O⟩_{L′_0} − ⟨O⟩_L) + (⟨O⟩_{L′} − ⟨O⟩_{L′_0}).
3. Bound the first difference by Theorem A (general-V form).
4. For the second difference, take base L′_0 and perturbation V_ch in (D2).
   - The linear term vanishes. e^{sL′_0}ρ_0 ∈ B_0; V_ch maps B_0 into ⊕_{Q≠0} B_Q (Lemma C.0); e^{(t−s)L′_0} preserves the sectors; and Tr[O·] annihilates B_{Q≠0}.
   - In Heisenberg form, the quadratic term is ∫∫ Tr[V_ch†(e^{(s−u)L′_0†} V_ch†(O′_{t−s})) ρ′_u].
   - By Lemma C.0, V_ch†(O′_a) = Σ_{Q≠0} A_Q with A_Q ∈ B_Q and ‖A_Q‖ ≤ ν_Q osc(O′_a) ≤ ν_Q osc(O).
   - By Lemma B.0, ‖e^{bL′_0†}A_Q‖ ≤ e^{−2γ_cQ²b}‖A_Q‖.
   - Finally |Tr[V_ch†(Y)ρ]| = |Tr[Y V_ch(ρ)]| ≤ ‖Y‖ ‖V_ch‖_{1→1}. Integrate over the simplex.

□

**Message.** Whether collective dephasing protects against an imperfection is decided by its *superoperator charge*, not by whether the imperfection is coherent or dissipative.
- The covariant part of any imperfection is first order and unprotected.
- Every charged part, Hamiltonian or dissipative, is second order and Zeno-suppressed.

### Corollary B6′ (sector-leakage diagnostic)

Let ρ_0 lie in one magnetization sector M_0 and take O = 1 − Π_{M_0}, with osc(O) = ½.

1. **Classes I and III** give leakage ≤ ½ ‖V_ch‖_{1→1} Σ_Q ν_Q φ(2γ_cQ², t) = O(κ² t/γ_c). This is second order and Zeno-suppressed.
2. **Class II, no γ_c dependence at all.** For covariant L′ and ρ_0 ∈ B_0, the whole trajectory stays in B_0, where D_c = 0. So every charge-0 expectation value, including the leakage, is exactly independent of γ_c.
3. **Class II, size.** Covariant dissipators with charged jumps (flips) give first-order leakage.

**Operational test.** Scan an engineered collective dephasing, for example global phase noise.
- Sector-changing rates independent of γ_c are covariant (class II).
- Rates falling as 1/γ_c come from charged imperfections (classes I and III).

(E8.)

### Remark C.2 (what z-basis snapshots can test)

z-basis data measure P_t(M) only. They test the strong symmetry (Proposition C.1) and are blind to the weak one.

51-ion data, measured by Kraft et al. and analysed from their public data:
- **Drift of ⟨M⟩.** It equals the metastable-level (D_{5/2}) decay: −0.0218 ± 0.0036 vs 0.0218 per ms. That decay is covariant emission, class II.
- **Variance of M.** It grows faster: 0.129 ± 0.011 per ms. The excess is a *balanced* sector-changing rate of 2.10 ± 0.22 s⁻¹ per ion.
- **What it does not show.** By Corollary B6′, the class of that rate is decided only by a γ_c scan. The data do not show a violation of the weak symmetry.

### Remark C.3 (certifying the collective rate)

1. **Concavity and bias.** γ_c*(Γ) = max{γ : Γ − γ11ᵀ ⪰ 0} is concave on the PSD cone, because the sum of feasible pairs is feasible. By Jensen, E γ_c*(Γ̂) ≤ γ_c*(Γ) for unbiased Γ̂. The plug-in estimator is therefore biased low, and it collapses on noisy estimates: E9 finds 0 in 100 of 100 PSD-constrained replicates.
2. **Structured model.** Take Γ = γ_c 11ᵀ + diag(g) with g ≥ 0. Lemma B.1 with Sherman–Morrison gives γ_c* = γ_c + (Σ_i 1/g_i)^{−1} ≥ γ_c, with equality if some g_i = 0.
3. **Consequence.** A one-sided lower confidence bound on γ_c certifies the Zeno factors, conditionally on the model.
4. **Data.** E9 gives a point estimate of 29.9 ± 4.1 rad/s. The bootstrap mean sits 2.1 rad/s higher, a bias of the errors-in-variables estimator (the design matrix is built from noisy measured integrals), so the percentile bound of 26.0 rad/s is not conservative. The basic (reverse-percentile) bound is 20.7 rad/s, and we certify γ_c ≥ 20.7 rad/s at 95%; the extended ansatz gives 22.1. The Zeno factors at 10 ms are then ≤ 0.88 (|q| = 1) and ≤ 0.62 (|q| = 2).

## E. Forgetting under charge conservation (time-uniform bounds)

**Setting.**
- The reference L is strongly U(1)-symmetric on a lattice.
- The local conserved densities are q_x, with Σ_x q_x = Q and L†(Q) = 0.
- δH = Σ_z h_z, with supp h_z ⊆ {z}_ℓ, ‖h_z‖ ≤ ε, and **[h_z, Q] = 0**. That is, each term conserves the charge, as hopping terms and density-dependent terms do.

### Lemma E.1 (summation by parts)

For X = Σ_x c_x q_x,

‖[δH, X]‖ ≤ 2ε ‖q‖ Σ_z Σ_{x∈{z}_ℓ} |c_x − c_z| ≤ C_ℓ ε ‖q‖ ‖∇c‖_{ℓ¹}.

In 1D, C_ℓ = 2ℓ(ℓ+1).

*Proof.* [h_z, c_z Q] = 0, and [h_z, q_x] = 0 for x ∉ supp h_z. So [h_z, X] = [h_z, Σ_{x∈{z}_ℓ}(c_x − c_z) q_x]. In 1D, |c_x − c_z| ≤ Σ_{bonds between} |∇c|. A bond lies between 2d ordered pairs (x, z) at distance d, so it is counted at most Σ_{d=1}^{ℓ} 2d = ℓ(ℓ+1) times. □

### Theorem E (multipole forgetting)

**Assumption (H).** Decompose O_u = e^{uL†}O = Σ_x c_u(x) q_x + Y_u, and take a multipole order m ≥ 0 such that, for u ≥ u_0,
- ‖∇c_u‖_{ℓ¹} ≤ C_m (Du)^{−(m+1)/2}, and
- ‖[δH, Y_u]‖ ≤ ε [C′_m (Du)^{−(m+1)/2} + C″ e^{−u/τ}].

These hold for diffusive transport with diffusion constant D. The order m is that of O: the moments Σ_y a_y y^k of its density content vanish for k < m. Y_u contains the currents slaved to density gradients (Fick's law), which inherit the same power.

**Statement.**

‖[δH, O_u]‖ ≤ ε C (Du)^{−(m+1)/2} + ε C″ e^{−u/τ}.

The Theorem-A bound ∫₀^t ‖[δH, O_u]‖ du therefore grows as:
- ε √(t/D) for m = 0 (monopole);
- (ε/D) log t for m = 1 (dipole);
- it stays bounded for m ≥ 2, i.e. it is **time-uniform**.

*Proof.* Lemma E.1 applied to the density part, plus the bound on Y_u. Integrate. □

**Remarks.**
1. **Tightness and the no-go (E10).** For the dephased XX chain with a uniform hopping miscalibration, the integrand exponents are:

   | Observable | Measured exponent (N = 400, u ∈ [11, 780–1500]) | Prediction |
   |---|---|---|
   | monopole n_c | −0.506 | −1/2 |
   | dipole | −1.022 | −1 |
   | quadrupole | −1.527 | −3/2 |
   | bond current | −1.029 (relaxes into a density dipole) | −1 |

   The bound for the monopole grows as √t: 0.031, 0.067, 0.176 at u = 10, 100, 1000. The dipole and current bounds grow logarithmically, and the quadrupole bound saturates at 0.081.

   So for conserved densities no time-uniform bound of Theorem-A type exists. This is the obstruction behind open item 1 of the main note.
2. **The true error can still be uniform.** For a uniform miscalibration the imperfection renormalises the diffusion constant, with δD/D = 2ε for J → J(1 − ε). For a density wave of wavenumber k, the first-order error of the density at the crest is A (δD/D) (Dk²t) e^{−Dk²t}. Its maximum A(δD/D)/e is independent of λ = 2π/k, while the Theorem-A bound at the peak time t = 1/(Dk²) grows ∝ λ (E10). In E10 the peak errors are 3.26, 3.03, 2.98 and 2.96 × 10⁻³ for λ = 20, 40, 80 and 160, against a prediction of 2.94 × 10⁻³. The bound at the peak time grows from 0.027 to 0.107. A time-uniform local bound for U(1)-conserving dynamics therefore exists in this class, but it needs the hydrodynamic (time-derivative) structure, not norm-based forgetting. *Open problem* (PRX direction).
3. **Weak symmetry restores forgetting.** Suppose covariant flips at rate γ_f break the conservation; for spins with σ^± at rate γ_f each, D†(σ^z) = −2γ_f σ^z.
   - Then c_u acquires the factor e^{−2γ_f u}, and the m = 0 bound saturates at ≈ ε C √(π/(2γ_f D)).
   - For quasi-free fermions with loss and gain at rates κ_l and κ_g, the factorisation e^{−(κ_l+κ_g)t} of the traceless quadratic part is exact.
   - Covariant flips thus cost first-order error (Corollary B6′) but restore the forgetting of the conserved charge.

## 6′. Numerical checks (phase 3)

| Statement | Experiment | What is checked |
|---|---|---|
| Lemma C.0, Prop. C.1, the charge table | tests | dense superoperator decomposition; the Kossakowski combinations |
| Thm B6, Cor. B6′ | E8 (N = 6) | κ-doubling ratios (2 vs 4); class II exactly γ_c-independent; class-III remainder Zeno-suppressed; B6 bound ≥ error |
| Remark C.2, C.3; weak symmetry of the dissipator | E9 | extended ansatz (dissipator charges); structured γ_c lower bound; strict γ_c* collapse; shadow infeasibility; 51-ion rates |
| Validation deviation | E6c | charge-resolved out-of-sample residuals vs learning window and dephasing model |
| Thm E, Remarks E | E10 (N = 400) | exponents −1/2, −1, −3/2; √t growth of the bound vs bounded true error |
