# Phase-4 theory note: collective noise with memory

Research note, phase 4 (2026-11-05). Notation as in `docs/theory/error-anatomy-theory.md` and `docs/theory/phase3-theory.md`:
- S^z = ½Σσ^z_i, sectors B_q = {X : [S^z, X] = qX}, Π_q the projection onto B_q, 𝒵 = [S^z, ·];
- D_c(γ_c) = −2γ_c 𝒵², which acts as −2γ_c q² on B_q;
- the reference L_0 is covariant: [L_0, 𝒵] = 0.

**Motivation.** The collective dephasing in the BEQS device is modeled by a master equation, that is, by white noise. Kraft et al. note that slow global magnetic-field fluctuations also act as collective dephasing, but are described by an average over static fields instead (their footnote 73). Their outlook also mentions learning colored noise. Theorem B6 used only the Markov kernel e^{−2γ_cQ²b}. This note asks what survives when the collective noise has memory. Numerical checks are in E12 (`experiments/e12_noise_memory.py`). The data test is in E11 (`experiments/e11_model_selection.py`).

## G. Classical collective noise of any spectrum

**Setting.** Each run sees the extra field H_ξ(t) = ξ(t) S^z. Here ξ is a real classical process, independent of the system. Per run, the generator is L_0 − iξ(t)𝒵 (+ an imperfection V). Write Φ(t, s) = ∫_s^t ξ(u) du and

χ_q(t) := E exp(−iqΦ(t, 0)).

### Lemma G.0 (exact sector damping for any classical collective noise)

The noise-averaged reference evolution is

E[T_ξ(t, 0)] = e^{tL_0} Σ_q χ_q(t) Π_q.

Equivalently, ρ̄(t) = Σ_q χ_q(t) Π_q e^{tL_0} ρ_0. No assumption on ξ is needed, and ‖Π_q ρ̄(t)‖₁ ≤ |χ_q(t)| ‖Π_q ρ_0‖₁.

*Proof.* 𝒵 commutes with L_0, so the time-ordered propagator factorizes: T_ξ(t, 0) = e^{tL_0} e^{−iΦ(t,0)𝒵}. On B_q, e^{−iΦ𝒵} = e^{−iqΦ}. Average, then use that e^{tL_0} is CPTP and commutes with Π_q. □

**Gaussian noise.** For stationary zero-mean Gaussian ξ with covariance C(τ), χ_q(t) = exp(−q² w(t)/2), where w(t) = Var Φ(t, 0) = 2∫_0^t (t − τ) C(τ) dτ. The averaged evolution then obeys the exact time-local master equation

dρ̄/dt = L_0 ρ̄ + D_c(γ_c(t)) ρ̄,   γ_c(t) = ẇ(t)/4 = ½ ∫_0^t C(τ) dτ.

This is collective dephasing with a time-dependent rate. Exact classical-noise master equations of this kind are known (Kiely, EPL 134, 10001 (2021), ref. [72] of BEQS); what is new below is their use in the error anatomy. The three standard cases are:

| Noise | w(t) | χ_q(t) | γ_c(t) |
|---|---|---|---|
| white, E ξξ = 4γ_c δ | 4γ_c t | e^{−2γ_c q² t} | γ_c |
| quasi-static, ξ ~ N(0, σ²) | σ²t² | e^{−q²σ²t²/2} | σ²t/2 |
| Ornstein–Uhlenbeck, σ² e^{−\|τ\|/τ_c} | 2σ²τ_c²(t/τ_c − 1 + e^{−t/τ_c}) | e^{−q²w/2} | (σ²τ_c/2)(1 − e^{−t/τ_c}) |

In the OU case, τ_c → 0 at fixed σ²τ_c = 2γ_c recovers the white case.

### Theorem G (anatomy under classical collective noise)

**Assumptions.** L_0 is covariant GKSL. ρ_0, O ∈ B_0. The imperfection V = V_0 + V_ch is time-independent (Lemma C.0 decomposition). Δ̄(t) denotes the noise-averaged error.

1. **Selection rule.** The first-order term of V_ch vanishes for every run, hence on average.
2. **Covariant part.** If V is covariant, Δ̄(t) is the same for every noise process, including no noise. The whole trajectory stays in B_0, where the noise acts trivially.
3. **Second order, any stationary process.** The exact second-order term of the averaged error is

   Δ̄^{(2)}(t) = Σ_{Q≠0} ∫_0^t ds ∫_0^s du χ_Q(s − u) K_Q(t, s, u),
   K_Q(t, s, u) = Tr[O e^{(t−s)L_0} V_{−Q} e^{(s−u)L_0} V_Q e^{uL_0} ρ_0].

   Hence |Δ̄^{(2)}(t)| ≤ Σ_Q ‖V_{−Q}‖_{1→1} ν_Q osc(O) t τ_Q(t), with the **protection time** τ_Q(t) := ∫_0^t |χ_Q(b)| db.
4. **All orders.**
   - If ξ has independent increments (white noise, Lévy noise), Theorem B6 holds with e^{−2γ_cQ²b} replaced by |χ_Q(b)|.
   - For stationary Gaussian noise, it holds with the innovation kernel e^{−Q² v(b)/2}. Here v(b) = Var[Φ(u + b, u) | ξ(τ), τ ≤ u] is the variance that remains once the past is known.

*Proof.*
- Item 1: Per run, the reference propagator is covariant and preserves sectors. Apply the argument of Theorem B6.
- Item 2: Corollary B6′ item 2 applies per run.
- Item 3: Expand the per-run evolution to second order in V. With ρ_0 and O in B_0, only the path B_0 → B_Q → B_0 survives. On that path, the noise contributes the phase e^{−iQΦ(s,u)} of the middle segment; the outer segments carry charge 0 and no phase. For a stationary process, E e^{−iQΦ(s,u)} = χ_Q(s − u). The bound follows as in Theorem B6.
- Item 4: In the D2 form of Theorem B6, the last factor ρ′_ξ(u) depends only on ξ on [0, u], and the middle phase is e^{iQΦ(s,u)}.
  - For independent increments the two are independent, so the average factorizes.
  - For Gaussian ξ, condition on the past: |E[e^{iQΦ(s,u)} | ξ_{≤u}]| = e^{−Q² v(s−u)/2}, which multiplies a bounded random variable.

□

**Protection times.** As t → ∞:
- white noise: τ_Q → 1/(2γ_cQ²);
- quasi-static noise: τ_Q → √(π/2)/(|Q|σ);
- OU noise: in between.

Take noise strengths matched by the decay of a single charge, χ_1(T_2) = e^{−1}. Then τ_Q^{white} = T_2/Q² and τ_Q^{static} = (√π/2) T_2/|Q|, so

τ_Q^{static} / τ_Q^{white} = (√π/2)|Q| ≈ 0.886|Q|.

*Telegraph noise* is the exception that proves the rule. For ±a with switching rate λ, the Laplace transform gives exactly ∫_0^∞ χ_Q = 2λ/(Q²a²). This is the 1/Q² law of white noise with the motional-narrowing rate a²/(2λ), even though the kernel oscillates and changes sign once |Q|a > λ.

The 1/|Q| law therefore belongs to a static *spread* of fields: the unbounded Gaussian tails make the decay Gaussian in time. A bounded, switching fluctuator returns to 1/Q² once its switching averages the phase.

**Message.** The selection rule and the γ_c independence of covariant errors are statements about charge alone, so they hold for collective noise of any spectrum. The Zeno suppression of charged errors survives memory, but its charge scaling depends on the noise:
- 1/Q² for white noise, and asymptotically for telegraph noise;
- 1/|Q| for quasi-static Gaussian noise.

Slow, Gaussian-distributed field noise therefore protects pair-creation errors (|Q| = 2) about 1.8 times less well than white noise of the same single-charge coherence time.

**Numerical check (E12).** N = 6, Néel state, E8 reference without collective dephasing. Four noise processes are matched by T_2 and averaged exactly with Kubo's stochastic Liouville equation (`qerr.noise`).

| Check | Result |
|---|---|
| Item 2: error of a covariant imperfection (Σ Z_jZ_{j+1}) | Identical for white, quasi-static, OU, telegraph and no noise: max deviation 9.2 × 10⁻¹³ against an error of 8.6 × 10⁻⁴ |
| Item 3: second-order kernel formula vs exact average | Max relative deviation 0.38%. Covers the four processes (telegraph is non-Gaussian), T_2 ∈ {0.5, 1, 2, 4}, \|Q\| = 1, 2 and t ∈ {2.5, 5, 10} |
| Order | κ-doubling ratio of the leakage: 3.998–4.000 |
| Protection, \|Q\| = 2, at T_2 = 0.5 | Leakage ratio to white noise: quasi-static 1.754 (limit 1.772), OU (τ_c = T_2) 1.643 (limit 1.649) |
| Protection, \|Q\| = 1, at T_2 = 0.5 | Quasi-static 0.99, against a limit of 0.886. At \|Q\| = 1 the protection time T_2 is not yet short compared with 1/J |
| Weak noise (T_2 = 4) | All ratios tend to 1.1–1.2: there is little protection to lose |
| Telegraph, \|Q\| = 2, at T_2 = 0.5 | Ratio 1.03: the oscillating kernel. Its t → ∞ protection-time ratio is 0.739 for both \|Q\| (the 1/Q² law) |

**Remarks.**
1. *Innovation kernel for OU.* v(b) = 2σ²τ_c [b − 2τ_c(1 − e^{−b/τ_c}) + (τ_c/2)(1 − e^{−2b/τ_c})].
   - For b ≪ τ_c, v ≈ (2/3)σ²b³/τ_c: the noise is predictable over short times, so there is no all-orders protection there.
   - For b ≫ τ_c, v ≈ 4γ_c b: Markov behaviour.
   - Quasi-static noise has v ≡ 0. Its protection is a property of the ensemble average at second order. A single run with a static field has no decoherence, only a detuning.
2. *Telegraph noise.* For telegraph noise ±a with switching rate λ, χ_Q(t) = e^{−λt}[cosh Ωt + (λ/Ω) sinh Ωt], with Ω = √(λ² − Q²a²). This noise is non-Gaussian, but item 3 holds as stated.
3. *Testing on data.* Within the Gaussian class, the data test is whether γ_c(t) is constant (white) or grows (memory). Under the exact time-local equation above, a time-dependent collective rate is linear in its parameters. The integral method therefore applies unchanged (E11).

**Data test on the BEQS 10-ion data (E11, E11b).** The data were measured by Kraft et al. and are analysed here from their public data.

- *Model.* The Gaussian class above is linear in its rate parameters, so the integral method takes the collective rate γ_c + βt directly: β enters through a t-weighted collective column. White noise means β = 0; a quasi-static component means β = σ²/2 > 0.
- *Out-of-sample prediction (5–10 ms).* Adding β to the minimal physical model lowers the rms error from 0.052 to 0.040. The charged classes improve most (X_i, Y_i: 0.082 → 0.028–0.053).
- *The estimator is biased.* It is an errors-in-variables estimator. A simulation-based calibration (E11b) puts the white-noise null of the fitted β at −19.5 ± 4 rad/s per ms rather than 0, and a pure quasi-static truth with β = 38.2 at 17–20 ± 4–6.
- *Result.* The measured β = 3.7 (1.5 ms window) and 10.3 (3 ms window) lie 4.9σ and 8.1σ above the white-noise null. **White collective noise is rejected.** Interpolating between the two calibrations gives a quasi-static slope of about 22–31 rad/s per ms: 60–80% of the static value, or an rms static field of about 210–250 rad/s.
- *Consequences.*
  - The certified collective rate of phase 3 (γ_c ≥ 20.7 rad/s) assumes white noise and must say so.
  - The Zeno factor of |Q| = 2 errors at 10 ms ranges from 0.40 to 0.78 depending on the noise model (E11). A noise-robust certificate takes the worst case over the spectra the data allow.

## F. Forgetting in both pictures: a time-uniform bound for conserved densities

**Setting.** As in Theorem E:
- the reference L is strongly U(1)-symmetric, with local densities q_x and Q = Σ_x q_x;
- δH = Σ_z h_z with [h_z, Q] = 0, ‖h_z‖ ≤ ε, and supp h_z ⊆ {z}_ℓ.

Theorem E showed that every bound which controls the integrand by the worst state grows as √t for a density (m = 0). Theorems A and E are of this kind. This section shows that the obstruction comes from ignoring the state, not from the problem.

### Lemma F.0 (two-sided Duhamel identity)

Δ(t) = ∫_0^t ⟨i[δH, O_u]⟩_{ρ′_{t−u}} du,

with O_u = e^{uL†}O (reference, Heisenberg picture) and ρ′_s = e^{sL′}ρ_0 (implemented dynamics, Schrödinger picture).

*Proof.* d/ds Tr[O_{t−s} ρ′_s] = Tr[O_{t−s}(L′ − L)ρ′_s]. Integrate over s ∈ [0, t] and substitute u = t − s. □

The density part of i[δH, O_u] is a sum of perturbation currents j_{z,x} = i[h_z, q_x]/ε, weighted by differences of the observable's density profile (Lemma E.1). Its operator norm decays only as u^{−1/2}. In the evolved state, however, the expectation of each current is itself slaved to a density gradient (Fick's law), and gradients of a bounded profile decay diffusively.

### Theorem F (two-sided forgetting)

Decompose O_u = Σ_x c_u(x) q_x + Y_u, and assume the following for some D > 0:
- **(H_O)** ‖∇c_u‖_{ℓ¹} ≤ C_O (1 + Du)^{−1/2}. In addition, |⟨i[δH, Y_u]⟩_ρ| ≤ ε C_Y (1 + Du)^{−1/2} 𝒢(ρ) + ε C_R e^{−u/τ}. Here 𝒢(ρ) is the largest local gradient-type expectation of ρ: a density difference, current or coherence.
- **(H_ρ)** max_{z,x} |⟨j_{z,x}⟩_{ρ′_s}| + 𝒢(ρ′_s) ≤ C_ρ (1 + Ds)^{−1/2}.

Then for all t,

|Δ(t)| ≤ ε [(C_ℓ C_O + C_Y) C_ρ B(t) + C_R τ],   B(t) = ∫_0^t (1 + Du)^{−1/2} (1 + D(t − u))^{−1/2} du ≤ π/D.

The bound is **time-uniform**: |Δ(t)| ≤ ε[(C_ℓ C_O + C_Y) C_ρ π/D + C_R τ].

*Proof.*
1. Start from Lemma F.0.
2. For the density part, use Lemma E.1 with the expectation in place of the norm: |⟨i[δH, Σ_x c_x q_x]⟩_ρ| ≤ Σ_z Σ_{x ∈ {z}_ℓ} |c_x − c_z| |⟨i[h_z, q_x]⟩_ρ| ≤ C_ℓ ε ‖∇c‖_{ℓ¹} max |⟨j⟩_ρ|.
3. Add the Y_u part from (H_O), and insert (H_ρ) at s = t − u.
4. Finally, B(t) ≤ ∫_0^t (Du)^{−1/2} (D(t − u))^{−1/2} du = π/D, a Beta function B(½, ½). □

### Corollary F-qf (quasi-free fermions; no assumptions)

**Setting.** Take number-conserving quadratic dynamics with number dephasing, an observable O = Σ_{xy} w_{xy} c_x†c_y, and a nearest-neighbour δh with bond amplitudes e_b. The dynamics keeps O_u quadratic, O_u = Σ w_{xy}(u) c_x†c_y. Let C′_{xy}(s) = ⟨c_x†c_y⟩_{ρ′_s} and M(u) = i[δh, w(u)]. Then

|Δ(t)| ≤ ∫_0^t [a(u) γ(t − u) + m(u) κ(t − u)] du,

with
- a(u) = Σ_b |e_b| |w_{x+1,x}(u) − w_{x,x+1}(u)|;
- m(u) = Σ_{x≠y} |M_{xy}(u)|;
- γ(s) = max_b |n′_x(s) − n′_{x+1}(s)|, the largest density gradient of the implemented state;
- κ(s) = max_{x≠y} |C′_{xy}(s)|, its largest coherence.

*Proof.*
1. Lemma F.0 and ⟨Σ M_{xy} c_x†c_y⟩ = Σ M_{xy} C′_{xy} give the integrand.
2. Split C′ into its diagonal and off-diagonal parts.
3. The diagonal part sums by parts: Σ_x M_{xx} n_x = i Σ_b e_b (w_{x+1,x} − w_{x,x+1})(n_x − n_{x+1}).
4. Hölder's inequality bounds both parts. □

(Checked: `tests/test_hydro.py`. The Duhamel identity is reproduced to 10⁻⁶ and the pointwise inequality holds.)

**Remarks.**
1. *What (H_ρ) asks.* It is parabolic smoothing for the state, plus Fick's law. It holds for every initial state with bounded density, domain walls included. Neither smoothness nor a small amplitude is needed. This is how the bound differs from linear response around equilibrium.
2. *Rapid mixing is not required.* The stability results of Cubitt et al. (CMP 2015) and Kashyap et al. (PRX 2025) need a unique fixed point. A conserved charge rules that out. Theorem F replaces rapid mixing by diffusive smoothing in both pictures. It is the operator analogue of the parabolic error estimates that keep perturbations of a diffusion constant from accumulating.
3. *Tightness.* For a density wave the true error is at most A(δD/D)/e for every wavelength (Remark E.2). Theorem F gives O(ε osc(n_0)/D), also independent of the wavelength, whereas the Theorem-A bound at the peak time grows ∝ λ.
4. *Combine with Theorem A.* Both bounds are rigorous, so min(Theorem A, Corollary F-qf) is too. Theorem A is the tighter one at short times, before the state has developed gradients. Theorem F wins at all later times.
5. *Locality.* Corollary F-qf pairs the observable factors with *global* maxima of the state factors. It therefore cannot use the distance between the observable and the region where the gradients sit. A pointwise pairing ∫ Σ_b |a_b(u)| |∇n_b(t − u)| du would recover that distance. This is the next refinement.

**Numerical check (E13).**
- *Setup.* Dephased XX chain with N = 400, g = 1, so D = 2. The imperfection is a uniform hopping miscalibration with ε = 0.01, and the observable is n_C. Eight initial product states run to t = 1500/J:
  - density waves with λ = 20, 40, 80, 160;
  - domain walls at distance d = 10 and 40;
  - the Néel state;
  - random occupations.
- *Engine.* Exact second moments from the N × N propagator `qerr.hydro.evolve_two_point`, which serves both pictures.

| Check | Result |
|---|---|
| Corollary F-qf ≥ \|true error\| at all t | Holds for every state |
| Time-uniformity | F(1500)/F(150) is 0.83 for the walls and 0.20–0.63 for the other states: F saturates or decreases. The Theorem-A bound grows by 2.72 (√10 = 3.16) |
| (H_O) observable factors, u ∈ [20, 800] | Exponents: a −0.505, m −0.533, ‖∇c_u‖₁ −0.502, Theorem-A integrand −0.504 |
| (H_ρ) state factors, s ∈ [20, 800] | Domain wall: γ −0.502, κ −0.505 (parabolic smoothing). Waves, Néel and random states decay faster |
| Tightness, sup_t F / sup_t \|error\| | Waves 4.9, 3.8, 2.6, 2.1 for λ = 20 → 160; Néel 4.2; random 6.0; walls 14 |
| Theorem A at t = 1500 / peak error | 36–87, still growing |
| Theorem A at the peak time / peak error | Waves 8.4 → 36, growing ∝ λ; walls 17–49; Néel 1.4; random 3.2 |

Two tightness criteria were set before the run, and both missed:
- a spread below 2× across λ: it is 2.3×, with F tighter for longer waves;
- a ratio within 10× for every state: the walls reach 14×, the price of the global maxima (Remark 5).
