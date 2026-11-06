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
   - For a stationary Markov process on finitely many noise values ξ_k, with rate matrix W (dp/dt = Wp), it holds with the kernel K_Q(b) = max_k |1ᵀ e^{b(W − iQΞ)} e_k|, Ξ = diag(ξ_k): the worst case over the noise value at the start of the middle segment. Telegraph noise and the Ehrenfest (discretized OU) chain are of this kind. Quasi-static noise (W = 0) gives K_Q ≡ 1: no all-orders protection, as for the Gaussian innovation kernel with v ≡ 0.

*Proof.*
- Item 1: Per run, the reference propagator is covariant and preserves sectors. Apply the argument of Theorem B6.
- Item 2: Corollary B6′ item 2 applies per run.
- Item 3: Expand the per-run evolution to second order in V. With ρ_0 and O in B_0, only the path B_0 → B_Q → B_0 survives. On that path, the noise contributes the phase e^{−iQΦ(s,u)} of the middle segment; the outer segments carry charge 0 and no phase. For a stationary process, E e^{−iQΦ(s,u)} = χ_Q(s − u). The bound follows as in Theorem B6.
- Item 4: In the D2 form of Theorem B6, the last factor ρ′_ξ(u) depends only on ξ on [0, u], and the middle phase is e^{iQΦ(s,u)}.
  - For independent increments the two are independent, so the average factorizes.
  - For Gaussian ξ, condition on the past: |E[e^{iQΦ(s,u)} | ξ_{≤u}]| = e^{−Q² v(s−u)/2}, which multiplies a bounded random variable.
  - For a Markov process, the future phase depends on the past only through ξ(u). Kubo's stochastic Liouville equation gives E[e^{−iQΦ(s,u)} | ξ(u) = ξ_k] = 1ᵀ e^{(s−u)(W − iQΞ)} e_k, whose modulus is at most K_Q(s − u) (the conjugate phase has the same modulus because W is real).

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
| Item 4, all-orders kernel bound (E12b) | Holds against the exact leakage for every process, T_2 and \|Q\|. Bound/exact at t = 10: white 7.5–26, OU (Ehrenfest chain) 9.8–61, telegraph 11–70, quasi-static 32–231 (no all-orders protection, K_Q ≡ 1) |

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
- *Result (Gaussian calibration, E11b).* The measured β = 3.7 (1.5 ms window) and 10.3 (3 ms window) lie 4.9σ and 8.1σ above the white-noise null. Interpolating between the two calibrations gives a quasi-static slope of about 22–31 rad/s per ms: 60–80% of the static value, or an rms static field of about 210–250 rad/s.
- *Shot-level calibration (E14), which supersedes E11b.* Synthetic data are sampled exactly like the BEQS data (the same 200 settings, 200 shots each) from truths that mix white and quasi-static collective noise, f = 0, ¼, ½, ¾, 1, with 20 realizations each (`qerr.shots`).
  - The white-noise null widens to −22.4 ± 9.1 (Xb15) and −22.9 ± 6.6 rad/s per ms (Xb30). The Gaussian calibration drew independent noise for every observable; shot-level samples keep the correlations between the strings read out in the same setting.
  - The measured β lies 2.9σ (1.5 ms) and 5.1σ (3 ms) above the null. The preregistered criterion, more than 3σ in both windows, is met only in the 3 ms window. With the f = ½ trajectory defect instead of the white one, the significances are 2.5σ and 4.6σ.
  - **White collective noise is rejected at 5.1σ by the 3 ms window.** The Neyman belt gives the quasi-static fraction f = 0.93 [0.62, 1] (3 ms) and 0.64 [0.12, 1] (1.5 ms). The fitted β increases monotonically with f in both windows, as required.
  - *Conservative check.* The bootstrap of the real data (settings resampled) has a β spread of 12.9 (Xb15) and 8.9 (Xb30), 1.35–1.4 times the shot-level null. Measured against that spread, z = 2.0 and 3.7. The rejection should therefore be quoted as 3.7–5.1σ in the 3 ms window. The bootstrap means (−8.5, −8.4) lie 12–19 below the point estimates: the errors-in-variables attenuation grows with the noise.
  - *Power.* The 1.5 ms window has four time points. Even a pure quasi-static truth gives an expected z of only 4.1, and f ≈ 0.64 gives 2.9, so a 3σ outcome there was a coin flip. The preregistered criterion was set without a power calculation.
  - The same simulations reproduce a decline of the learned J_nn with later windows although the truth is constant (607 rad/s): 581–588 on [0, 1.5], 539–551 on [0, 5], 491–501 on [1, 3] ms. The measured decline (620, 498, 373) is 10–25% steeper on [0, 5] and [1, 3]: part of the J drift is genuine.
- *Consequences.*
  - The certified collective rate of phase 3 (γ_c ≥ 20.7 rad/s) assumes white noise and must say so.
  - The Zeno factor of |Q| = 2 errors at 10 ms ranges from 0.40 to 0.78 depending on the noise model (E11). A noise-robust certificate takes the worst case over the spectra the data allow.
  - Re-certified with window-matched mixtures over the wider interval f ∈ [0.12, 1] (E14): at 10 ms the worst-case Zeno factor is 0.85 for |Q| = 1 and 0.57 for |Q| = 2 at the certified scale γ_c ≥ 20.7 rad/s (0.80 and 0.49 at the point estimate). Memory helps here: at matched early-time decay, the quasi-static truth (f = 1) protects more at 10 ms (0.71 and 0.44) than the white one (0.88 and 0.62).

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
   - *Checked (E13b).* Hölder's inequality entry by entry gives P(u, s) = Σ_b |e_b| |w_{x+1,x}(u) − w_{x,x+1}(u)| |n′_x(s) − n′_{x+1}(s)| + Σ_{x≠y} |M_xy(u)| |C′_xy(s)|, which is still rigorous, and F_loc(t) = ∫_0^t P(u, t − u) du.
   - sup_t F_loc / peak error: waves 1.37, 1.11, 1.05, 1.04 (λ = 20 → 160), Néel 2.0, random 1.6, walls 2.1 and 2.0. The global pairing gave 4.9–2.1, 4.2, 6.0 and 14. F_loc never exceeds it.
   - Both E13 tightness criteria are met by the local pairing: the spread across λ is 1.3× and every state is within 2.2×. The global maxima were therefore the main loss. The per-bond absolute values cost at most 37% at the peak (λ = 20).
   - At late times F_loc saturates (≈ 5 × 10⁻³ for the walls) while the true error decays (5 × 10⁻⁴ at t = 1500). The bound is time-uniform, not decaying: the convolution of two s^{−1/2} factors is constant.
   - The numerical evaluation falls below the true error only at the rounding level. At the first grid time it is 0.2% low (bilinear interpolation of P on the first cell); elsewhere the shortfall is below 10⁻²⁴ in absolute terms (float32 underflow). At most this is 1.2 × 10⁻⁶ of the peak error (Néel). The preregistered flag "holds at all t", with an absolute tolerance of 10⁻¹⁰, is reported as failed for λ = 20 and Néel on these numerical grounds.

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

### Lemma F.2 (exact density propagator of the dephased XX chain)

**Setting.** Infinite chain, hopping J, number dephasing g. Write the two-point matrix in centre-of-mass momentum k and relative distance r. Then

dC_r/dt = −a_k (C_{r+1} − C_{r−1}) − g (1 − δ_{r0}) C_r,   a_k = 2J sin(k/2).

For a product initial state only the density mode is nonzero at t = 0: C_r(k, 0) = δ_{r0} n̂_0(k). Write n̂(k, t) = G_k(t) n̂_0(k).

**Statement.**
1. G̃_k(p) = ∫_0^∞ e^{−pt} G_k(t) dt = 1/(√((p + g)² + 4a_k²) − g). The coherences follow the density: C̃_{±m}(k, p) = (±μ_k(p))^m C̃_0(k, p) for m ≥ 1, with μ_k(p) = [√((p + g)² + 4a_k²) − (p + g)]/(2a_k).
2. If 2|a_k| < g, G̃_k has exactly one pole, at p_+(k) = −g + √(g² − 4a_k²) = −Dk² + O(k⁴) with D = 2J²/g, and residue g/ν_k, ν_k = √(g² − 4a_k²). Every other singularity lies on Re p = −g (branch points −g ± 2ia_k). Hence G_k(t) = (g/ν_k) e^{p_+(k) t} + O(e^{−gt}).
3. If 2|a_k| ≥ g, every singularity lies on Re p = −g.
4. |G_k(t)| ≤ 1 for all t.

*Proof.*
1. Laplace-transform the chain. For r ≠ 0 the recursion has constant coefficients, and its square-summable solution is geometric with ratio ±μ_k (the root of a_k μ² + (p + g)μ − a_k = 0 with |μ| < 1). The r = 0 equation then reads p C̃_0 − 1 = −2a_k μ_k C̃_0, which gives item 1.
2. A pole needs √((p + g)² + 4a_k²) = g. The branch that behaves as p + g at infinity takes the value +g at p_+ and −g at p_− = −g − ν_k, so only p_+ is a pole. Its residue is the inverse derivative of the square root at p_+, which is g/ν_k. Expand p_+ in k for the diffusive limit.
3. For 2|a_k| ≥ g the two candidate poles have p + g = ±i√(4a_k² − g²).
4. The hopping term is anti-Hermitian in ℓ²(r), so d/dt Σ_r |C_r|² = −2g Σ_{r≠0} |C_r|² ≤ 0. □

**Check (E13c).** The crest density of the E13 density waves equals 1/2 + 0.4 G_k(t), with G_k from the truncated r-chain, to 2 × 10⁻⁶ for λ = 20, 40 and 80. For λ = 160 the agreement is 4 × 10⁻³, because the open boundary of the N = 400 chain has a node there. In the slow-mode window 10/g < t < 3/(Dk²), the fitted decay rate equals −p_+ to 6–7 digits: 0.2199671 vs 0.2199666 (λ = 20), 0.0505229 vs 0.0505229 (λ = 40), 0.01240764 vs 0.01240764 (λ = 80). The diffusive estimate Dk² is 11% off at λ = 20, so the exact pole, not Dk², is the relevant rate.

### Proposition F.1 (the hypotheses of Theorem F hold for the dephased XX chain)

**Setting.** As in Lemma F.2, with O = n_C, a uniform hopping miscalibration δh of size ε, and an initial product profile of bounded variation V(n_0) = Σ_x |n_0(x + 1) − n_0(x)| < ∞. A domain wall has V = 1.

**Statement.** There is a constant C that depends only on g/J such that, for all u, s ≥ 0,

γ(s) + κ(s) ≤ C V(n_0) (1 + Ds)^{−1/2},   a(u) + m(u) ≤ C ε (1 + Du)^{−1/2}.

Corollary F-qf then gives |Δ(t)| ≤ 2C² ε V(n_0) B(t) ≤ 2πC² ε V(n_0)/D for all t. This bound is time-uniform and assumes nothing.

*Proof sketch.*
1. *Fourier bounds.* On the slow region |a_k| ≤ g/4 the residue is at most 2/√3 and −p_+(k) ≥ 2a_k²/g ≥ c k², so Lemma F.2 gives |G_k(t)| ≤ C e^{−c k² t}. On the rest of the Brillouin zone every singularity has Re p ≤ −(1 − √3/2) g, which gives |G_k(t)| ≤ C e^{−c′ g t}. Hence ∫|G_k(t)| dk/2π ≤ C (1 + Dt)^{−1/2}. The second estimate must be uniform in k: the residue g/ν_k diverges where the pole meets a branch point (2|a_k| = g), and the cut contribution cancels the divergence. The section after this sketch proves the uniform estimate by hypocoercivity instead.
2. *State side.* n(s) = K_s ∗ n_0 with K̂_s(k) = G_k(s). Then ∇n(s) = K_s ∗ ∇n_0, so γ(s) ≤ ‖K_s‖_∞ V(n_0) ≤ V(n_0) ∫|G_k(s)| dk/2π.
   - In the slow-pole part, the m-th coherence is the density times μ_k(p_+)^m, and μ_k(p_+) = (g − ν_k)/(2a_k) = O((J/g)|sin(k/2)|). One factor of sin(k/2) turns n̂_0 into the transform of ∇n_0, so κ(s) obeys the same bound. The fast part is O(e^{−c′gs}).
3. *Observable side.* The Heisenberg dynamics of w(u) is the same chain with a_k → −a_k, and G_k is even in a_k. It starts from w(0) = e_C e_Cᵀ; measure positions from C.
   - a(u) = 2ε ‖Im w_1(u)‖_{ℓ¹} is the ℓ¹ norm of a sequence whose transform σ_u(k) has the slow part μ_k(p_+)(g/ν_k) e^{p_+(k)u}. Carlson's inequality, (Σ_x |f_x|)⁴ ≤ c Σ_x |f_x|² Σ_x (1 + x²)|f_x|², reads ‖f‖_{ℓ¹} ≤ c^{1/4} ‖f̂‖_2^{1/2} (‖f̂‖_2² + ‖f̂′‖_2²)^{1/4} in Fourier space.
   - Step 1 gives ‖σ_u‖_2 ≲ (J/g)(1 + Du)^{−3/4} and ‖σ_u′‖_2 ≲ (J/g)(1 + Du)^{−1/4}, so a(u) ≲ ε (J/g)(1 + Du)^{−1/2}.
   - M = i[δh, w(u)] is the hopping term of the Heisenberg chain applied to w(u), so m(u) needs the ℓ¹ norm over the coherence index r as well as over x. *Closing the ℓ¹ sum over r* below proves m(u) ≤ Cε(1 + Du)^{−1/2} with an exponential weight in r.
4. Insert into Corollary F-qf and use ∫_0^t (1 + Du)^{−1/2} (1 + D(t − u))^{−1/2} du ≤ π/D. □

**Closing the gap in step 1 (proof; checked by E13d).** The uniform estimate does not need the pole–cut split; hypocoercivity gives it with explicit constants.
- *Setting.* In time units 1/g and with α = a_k/g, the r-chain reads ċ = Mc with M = L − T, L = −(1 − Π) and T = α(S − S*). Here S is the shift r → r + 1, Π projects onto r = 0 and Q = 1 − Π. T is skew-adjoint and L is symmetric; ΠTΠ = 0, ‖TΠ‖ = ‖ΠT‖ = √2|α| and ‖T‖ ≤ 2|α|.
- *Modified energy.* Take the auxiliary operator of Dolbeault, Mouhot and Schmeiser (Trans. AMS 367, 3807 (2015)), A = (1 + (TΠ)*(TΠ))^{−1}(TΠ)* = ΠT*/(1 + 2α²). Write w = α²/(1 + 2α²), a = ‖Qc‖, b = |c_0| and H = ½‖c‖² + ε Re⟨Ac, c⟩. Since A maps into ΠH and LA = 0,
  dH/dt = −a² + ε Re[⟨ALc, c⟩ − ⟨ATΠc, c⟩ − ⟨ATQc, c⟩ + ⟨TAc, c⟩].
  - ATΠ = 2wΠ, so the second term is −2w b².
  - ⟨TAc, c⟩ = ‖ΠTQc‖²/(1 + 2α²) ≤ 2w a².
  - |⟨ALc, c⟩| + |⟨ATQc, c⟩| ≤ κ a b with κ = √2|α|(1 + 2|α|)/(1 + 2α²). Young's inequality gives κab ≤ w b² + (κ²/4w) a², and κ²/4w = (1 + 2|α|)²/(2(1 + 2α²)) ≤ 3/2.
  - Hence dH/dt ≤ −(1 − 5ε/2) a² − ε w b², and ε = 1/5 gives dH/dt ≤ −a²/2 − (w/5) b².
  - The cross term is O(|α|) because ‖AL‖ = ‖ΠTQ‖/(1 + 2α²) = O(|α|). That is what lets ε stay independent of α. With O(1) constants in the cross term, the general DMS rate would come out O(α⁴) instead of O(α²).
- *Decay.* ‖A‖ = √2|α|/(1 + 2α²) ≤ 1/2, so (2/5)‖c‖² ≤ H ≤ (3/5)‖c‖² and dH/dt ≤ −(w/5)‖c‖² ≤ −(w/3)H. Therefore
  ‖e^{tM}‖ ≤ √(3/2) e^{−λt},  λ = α²/(6(1 + 2α²)) ≥ min(α², 1)/18.
  In physical units λ_k ≥ min(a_k²/g, g)/18, uniformly in k, with explicit constants.
- *Step 1.* |G_k(t)| ≤ ‖e^{tM}‖, and a_k² ≥ 4J²k²/π² for |k| ≤ π. Splitting off the part of the zone where a_k > g gives
  ∫|G_k(t)| dk/2π ≤ √(3/2) [min(1, (3/2)√(π/(Dt))) + e^{−gt/18}] ≤ C(1 + Dt)^{−1/2},
  with C depending only on g/J.
- *Coherences carry sin(k/2) (step 2).* Qc obeys d(Qc)/dt = (−1 − QTQ)Qc − c_0 TΠe_0, and QTQ is skew. Hence
  ‖Q e^{tM} e_0‖ ≤ √2|α| ∫_0^t e^{−(t−s)} |c_0(s)| ds ≤ 2|α| e^{−λt}.
  - By duality (M^T is the chain at −α), ‖Π e^{tM} Q‖ ≤ 2|α| e^{−λt}, and then ‖Q e^{tM} Q‖ ≤ e^{−t} + 4α² e^{−λt}.
  - Since |α|·|n̂_0(k)| = (J/g)|(∇n_0)^(k)| ≤ (J/g)V(n_0), the bond coherence of the state obeys κ(s) ≤ (2J/g) V(n_0) ∫ e^{−λ_k s} dk/2π ≤ C V(n_0)(1 + Ds)^{−1/2}.
- *k-derivative (step 3).* d = ∂_α c obeys ḋ = Md − (S − S*)c with d(0) = 0. The source has |Π(S − S*)c| ≤ √2‖Qc‖ ≤ 2√2|α| e^{−λt} and ‖Q(S − S*)c‖ ≤ √6 e^{−λt}. Propagating each part with the block estimates above gives
  ‖Q ∂_α e^{tM} e_0‖ ≤ (3 + 16α²t) e^{−λt}.
  - The Duhamel bound without the split by sector, of order t e^{−λt}, does not decay in L²(dk). The outline in an earlier version of this note used it.
  - With the factor α²t, and since λ ≥ α²/18 for |α| ≤ 1, one gets |∂_kσ_u(k)| ≤ C(J/g) e^{−λ_k u/2}. Hence ‖σ_u′‖_2 ≤ C(J/g)(1 + Du)^{−1/4} and ‖σ_u‖_2 ≤ C(J/g)(1 + Du)^{−3/4}, as step 3 requires.
- *Numerical check (E13d).* The estimates hold for every truncation |r| ≤ R, so the truncated chain can be checked directly. Exact matrix exponentials were used for α from 0.01 to 10 and t up to 15/λ.
  - The largest ratios of measured to bound for the five operator estimates are 0.83, 0.71, 0.71, 0.9996 and 0.47. The value 0.9996 is reached as t → 0, where the e^{−t} term is exact.
  - The integrated estimate reaches 0.40–0.41 at J/g = 0.25 and 1.
  - λ is 3–8% of the slowest rate of the infinite chain: the constants are conservative, the scaling is right.

**Closing the ℓ¹ sum over r (Lemma F.3; proof, checked by E13e).** The estimates above control the coherences in ℓ²(r), but m(u) = Σ_{x≠y} |M_xy(u)| is an ℓ¹ norm over r as well as over x. An exponential weight in r closes it.
- *Weighted chain.* Let W = diag(e^{θ|r|}), s = sinh(θ/2) and α′ = α cosh(θ/2). Conjugation gives W^{1/2} M W^{−1/2} = −Q − α′K − αsB, with K = S − S* and B the symmetric matrix with +1 on the bonds (r − 1, r) for r ≥ 1 and −1 for r ≤ 0. The skew part is the old chain at α′. The weight adds the symmetric term P = αsB, with ‖P‖ ≤ 2|α|s, ‖PΠ‖ = √2|α|s and ΠPΠ = 0.
- *Lemma F.3 (weighted decay).* Let α_m = 2J/g and |s| ≤ 1/(40(1 + α_m)). For |α| ≤ α_m,
  ‖W^{1/2} e^{tM} W^{−1/2}‖ ≤ √(3/2) e^{−λ_θ t},  λ_θ = α²/(13(1 + 2α²)) ≥ min(α², 1)/39.
  *Proof.* Take the modified energy of the unweighted proof at α′: T′ = α′K, A = ΠT′*/(1 + 2α′²), ε = 1/5. The skew part gives dH/dt ≤ −a²/2 − (w′/5) b² with w′ = α′²/(1 + 2α′²). The symmetric term adds X = −⟨Pc, c⟩ − ε Re[⟨APc, c⟩ + ⟨Ac, Pc⟩], and
  - |⟨Pc, c⟩| ≤ 2|α|s a² + 2√2|α|s ab, because ΠPΠ = 0;
  - ⟨APc, c⟩ = αα′s (2c_0 + c_2 + c_{−2}) c̄_0 / (1 + 2α′²), so its modulus is at most s w′ (2b² + √2 ab);
  - Ac is a multiple of e_0 of norm at most √2|α′| a/(1 + 2α′²), so |⟨Ac, Pc⟩| ≤ 2s w′ a².
  Young's inequality on the ab terms, against half of the b² margin, costs at most 22.9 s²(1 + 2α′²)/(1 − 2s) a². With the stated s the a² coefficients sum to at most −2/5 and the b² coefficient is at most −0.095 w′. So dH/dt ≤ −0.095 w′ ‖c‖² ≤ −0.158 w′ H, and (2/5)‖c‖² ≤ H ≤ (3/5)‖c‖² gives the claim, since 0.079 w′ ≥ α²/(13(1 + 2α²)). Every step uses |s| only, so the bound also holds for W^{−1/2} e^{tM} W^{1/2}, which is the transpose of the weighted chain at −α. □
- *Weighted block estimates.* On QH the generator −1 − Q(T′ + P)Q has symmetric part at most −(1 − 2|α|s) ≤ −0.95, and ‖Q(T′ + P)Π‖ = √2|α| e^{θ/2}. The Duhamel arguments of the unweighted proof then give
  ‖W^{1/2} Q e^{tM} e_0‖ ≤ 2|α| e^{−λ_θ t},  ‖Π W^{1/2} e^{tM} W^{−1/2} Q‖ ≤ 2|α| e^{−λ_θ t},  ‖Q W^{1/2} e^{tM} W^{−1/2} Q‖ ≤ e^{−0.95t} + 3.2α² e^{−λ_θ t}.
  For the α-derivative the source is −K_θ c with K_θ = cosh(θ/2) K + sB, ‖K_θ‖ ≤ 2e^{θ/2} and ΠK_θΠ = 0, so
  ‖W^{1/2} Q ∂_α e^{tM} e_0‖ ≤ (3 + 14α²t) e^{−λ_θ t},  |Π ∂_α e^{tM} e_0| ≤ 8.6|α| t e^{−λ_θ t}.
- *The ℓ¹ sum.* Write r = x − y and R = (x + y)/2, measured from C. A uniform miscalibration δh = (ε/J) h gives M̂_r(k, u) = (ε/J) a_k (K ŵ(k, u))_r, the hopping term of the Heisenberg chain. Two Cauchy–Schwarz steps follow.
  - For each r, the weight λ + μR² with λ = ‖Rf‖², μ = ‖f‖² gives the two-sided Carlson inequality ‖f‖_{ℓ¹} ≤ √2 ‖f‖_2 + √(2π) ‖f‖_2^{1/2} ‖Rf‖_2^{1/2}. By Parseval the two norms are those of M̂_r and ∂_k M̂_r in L²(dk/2π).
  - Over r, the weight e^{θ|r|} gives
    m(u) ≤ S_θ^{1/2} [√2 𝒳 + √(2π) 𝒳^{1/2} 𝒴^{1/2}],  S_θ = Σ_{r≠0} e^{−θ|r|} = 2/(e^θ − 1),
    where 𝒳² = ∫ ‖W^{1/2} Q M̂(k, u)‖² dk/2π and 𝒴² is the same for ∂_k M̂.
  - Lemma F.3 and the derivative estimates give, with τ = gu,
    ‖W^{1/2} Q M̂‖ ≤ (ε/J) |a_k| 2e^{θ/2} √(3/2) e^{−λ_θ τ},
    ‖W^{1/2} Q ∂_k M̂‖ ≤ 2ε e^{θ/2} (|a_k′|/J) [√(3/2) + |α|(3 + 8.6|α|τ + 14α²τ)] e^{−λ_θ τ}.
    As in step 3, 𝒳 ≤ Cε (1 + Du)^{−3/4} and 𝒴 ≤ Cε (1 + Du)^{−1/4}.
  - Hence m(u) ≤ Cε (1 + Du)^{−1/2}, with C depending only on g/J. This was the last open step: Proposition F.1 holds as stated, with no hypothesis left.
- *Numerical check (E13e).* Exact matrix exponentials on truncated chains, α from 0.01 to α_m, t up to 15/λ_θ, J/g = 0.25 and 1 (θ = 0.033 and 0.017).
  - The largest ratios of measured to bound are 0.82 for the weighted decay (both signs of θ), 0.72 and 0.70 for the two off-diagonal blocks, 0.9992 for the QQ block (as t → 0), 0.48 and 0.45 for the two derivative estimates, and 0.17 for the ℓ¹ norm over r.
  - On the E13 chain (N = 400, J = g = 1, ε = 0.01) the measured m(u) stays below the Carlson step evaluated on the exact r-chain (largest ratio 0.72), and that stays below the closed bound (largest ratio 0.05). The Carlson step is nearly tight; the slack sits in the hypocoercive constants. The closed bound is 76, 2550 and 730 times m at u = 1, 100 and 1500.
  - The closed bound has a hump near u ≈ 10, made by the factors α²τ, so its slope on [20, 800] is −1.26. Its slope is −0.504 on [10⁵, 10⁶] and −0.501 on [10⁷, 10⁸], and m_B √u tends to 4.88. The measured m(u) has slope −0.53 on [20, 800].

**Remarks.**
1. The constants are not optimized. The exponents match E13: −0.50 to −0.53 for a, m and ‖∇c_u‖₁, and −0.50 for γ and κ of the domain wall.
2. With Proposition F.1, Corollary F-qf is a proven time-uniform bound for quasi-free fermions with dephasing. For interacting chains (H_O) and (H_ρ) remain hypotheses; E16 tests them at N = 10.
3. *Interacting check (E16).* The dephased XXZ chain at the isotropic point Δ = 1 (N = 10, g = 4, exact) is compared with the quasi-free control Δ = 0 at the same size.
   - Lemma F.0 holds to 4 × 10⁻⁴ of the peak error, which is the quadrature error.
   - The integrand G(u, t − u) has a fixed sign on every diagonal. Hence F_abs equals the envelope of the true error to 10⁻⁴ for all four initial states and both Δ, while Theorem A is 8–85 times looser.
   - On [1, 10] the exponents of the (H_O) factors differ from the control by 0.06, and those of the density gradient and the current by at most 0.07. The bond coherence is nonzero only at Δ = 1, where ZZ breaks the sublattice symmetry, and decays faster (exponents −1.1 to −1.9).
   - The exponents (about −0.9) are steeper than the infinite-chain −1/2 because the slowest mode of the open chain decays at D(π/N)² ≈ 0.05 and everything saturates after t ≈ 20. The comparison between Δ = 1 and Δ = 0 at the same size is the meaningful one.
   - The preregistered comparison of growth on [2, 10] (F_abs against A) fails for the lowest density mode, whose error itself grows until t ≈ 1/(Dk²) ≈ 20. That criterion did not test time uniformity.

## Data checks on the BEQS 10-ion data (E15, E11e, E17, E11f, E11g)

The data were measured by Kraft et al. and are analysed here from their public data.

**Out-of-sample bands (E15).** The two reference models of E11 are learned on t ≤ 1.5 ms and predict 1.5 < t ≤ 10 ms: X15 (U(1)-symmetric couplings, single-site fields, collective dephasing, fixed D5/2 emission) and the memory model Xb15 (collective rate γ_c + βt).
- *Certified band.* Theorem A in its general-V form, with the bootstrap deviations V_k = L_k − L_ref of X15 (30 refits) and the 95% quantile over k. It holds for every initial state.
  - Only mean Z stays informative to 10 ms: B = 0.13, 0.60 and 1.62 at 1.5, 5 and 10 ms, against osc(O) = 2. Its integrand at u = 0 is 0.063 per ms, against 0.29 for mean X and Y and 0.84–1.66 for the observables of the central pair. This is Remark A.3 on data: a conserved observable sees only the symmetry-breaking part of δH.
  - The other observables become vacuous at 1–4 ms. Corollary A.1, the BEQS form, is already 12.6 at 1.5 ms.
  - The preregistered coverage (at least 90% of the held-out points) is met with 100%, but mostly after the band has become vacuous. While it is informative, it covers every held-out point of mean Z, X and Y.
- *Statistical band.* Twelve bootstrap refits per model, simulated exactly; the band is 1.96 √(sd_boot² + SEM²). It is 30–550 times narrower than the certified band and informative throughout.
  - Calibration factors, rms deviation over predicted spread, for X15 and Xb15: 1.49–1.54 and 1.47–1.50 in the three neutral classes; 1.82–2.40 and 0.61–1.12 in the three charged classes; 1.77 and 1.08 over all 435 observables.
  - The memory model therefore changes only the charged classes, as Lemma G.0 requires: collective noise multiplies the charge-q blocks by χ_q and leaves the neutral block alone. The evidence for memory now includes out-of-sample prediction, not only the fitted β.
  - The neutral classes keep a structural error of about 1.5 times the predicted spread. For example, the measured mean Z lies 0.010–0.035 above both predictions for every held-out time and both initial states.
- *Consequence for certification.* On this device a worst-case certificate over all initial states is informative for 10 ms only for conserved observables. Elsewhere the usable statement is a probabilistic band. Its calibration is empirical: it holds as it stands in the charged classes of the memory model, and it needs a factor of about 1.5 in the neutral classes.

**The single-ion noise block (E11e).** Synthetic shot-level data from the E14 truths contain no single-ion noise; their total flip rate is the D5/2 emission, 0.86 rad/s. The E11c fit (minimal model plus the full Kossakowski block of every ion, t ≤ 1.5 ms) is applied to 20 realizations of each truth, f = 0, ½, 1.
- The fit returns non-PSD XY blocks on 9–10 of 10 ions and a total flip rate of about −50 rad/s (5–95%: −69 to −22). The measured −30.5 rad/s lies inside this range for f = 1. The negative total flip rate and the per-ion count of non-PSD blocks are therefore produced by the estimator.
- The rms of the charged site means is 23–26 rad/s on synthetic data (95th percentile 29.5–33.6) and 54.2 rad/s on the data. By the preregistered rule the charged structure is in the data.
- *Per combination (second run with the same seeds, post hoc).* The estimator has systematic charged offsets of up to 37 rad/s, and the data lie outside the synthetic range in all six charged combinations. The largest excess is in the charge-2 imbalance S_XX − S_YY: −94 rad/s, against a synthetic median of −18 to −22.
- *The corrected block is not PSD.* After subtracting the synthetic medians, the total flip rate becomes +9 to +13 rad/s, but the site-mean block still has S_XX = (S + K)/2 = −31 rad/s, with z = −3.4 against the data bootstrap. A GKSL single-ion noise requires S_XX ≥ 0.
- *Interpretation.* If the bias carries over from the synthetic truths to the data, the charge-2 component cannot come from Markovian single-ion noise. Either the single-ion noise has memory (a negative time-local rate, which breaks CP-divisibility), or an effect outside the model projects onto the S_XX − S_YY column. E11d excluded transients, SPAM, field drift, a readout tilt and charge-2 couplings as such effects. E11f below tests the first option and finds no support for it.

**Conserved quantities (E17).** The learned models conserve three neutral quantities exactly or nearly: the magnetisation (mean Z), its square (the mean ZZ over all pairs) and the energy of the symmetric Hamiltonian Y15. Collective dephasing, white or quasi-static, conserves all three, and a drift of the symmetric couplings conserves the first two.
- The neutral residual of E15 sits in these directions: the calibration factor is 2.5 (Σ Z_i), 2.9 (Σ Z_iZ_j) and 4.2 (J-weighted flip-flops) along them, and 1.34–1.55 in the rest.
- The data break all three conservation laws. Between t = 0 and the mean over t ≥ 2 ms, mean Z of the tilted state rises by 0.044 (z = 17), mean ZZ of the y-polarised state rises by 0.031 (z = 13), and the energy changes by +128 and −352 rad/s (z = 5 and −15). A drift of the symmetric Hamiltonian cannot produce the first two.
- *Mean-Z balance.* For any U(1)-symmetric remainder, d(Σ_i Z_i)/dt = Σ_i 2(h^x_i⟨Y_i⟩ − h^y_i⟨X_i⟩) minus the non-conserving dissipation. With the measured ⟨X_i⟩, ⟨Y_i⟩ as regressors, the mean-Z changes of both states need a coherent field (h^y ≈ −21 rad/s, acting while the transverse polarisation lasts, about 2 ms) and an incoherent flip rate S = S_XX + S_YY ≈ 5 rad/s, about twelve times the D5/2 decay: χ² = 63 for 36 degrees of freedom, against 291–895 for either mechanism alone. The flip rate agrees in order of magnitude with the bias-corrected flip rate of E11e. The per-ion X15 fields, evaluated on the measured polarisations, predict a fall of mean Z in the y-polarised state that the data do not show, so they are a compromise of the fit.
- Time-point jitter (second differences 0.84–2.5 times the bootstrap expectation), drift over the acquisition (two of six homogeneity tests at p ≈ 0.01) and collision-like shots (at most 0.2–0.3% of the shots, compared with a model in E11f) are small next to these violations.
- So the neutral structural error comes from conservation-breaking processes that the learned generator does not contain. Collective noise leaves the neutral block alone (Lemma G.0), which is why the memory model does not change it.

**Non-Markovian truths (E11f, E11g).** The shot-level SBC of E11e was repeated with truths that each add one process: quasi-static single-ion z detuning (rms 50 and 150 rad/s), a quasi-static common scale of the couplings (10%), the X15 transverse fields under a quasi-static collective detuning, and Markovian local dephasing (5 rad/s), which the E11c ansatz lacks.
- None of them reproduces the imbalance. The shifts of the median K relative to the control are +1.9, −9.6, +1.2, +1.7 and −12.8 rad/s, against the −74 rad/s excess of the data; the preregistered thresholds were the 5–95% range and a shift of −37 rad/s. Noise-free fits give |K| ≤ 8 rad/s, so the synthetic offset of about −20 rad/s is estimator bias from shot noise, not model misspecification.
- Removing one interior time point of the learning window (E11g, post hoc) changes K by about 2σ of its synthetic spread, but the excess over the synthetic median stays between −49 and −105 rad/s.
- The memory reading of the imbalance (CP-divisibility) is therefore not supported by covariant single-ion noise with memory. What remains is an effect outside the model class. Since E17 sees U(1)-breaking processes directly in the conserved quantities of the same data, the leading candidate is U(1)-breaking dynamics that change over the first few milliseconds, which a time-local ansatz cannot represent. E11d only removed the first 0.5 ms.
