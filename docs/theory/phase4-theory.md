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
2. With Proposition F.1, Corollary F-qf is a proven time-uniform bound for quasi-free fermions with dephasing. For interacting chains (H_O) and (H_ρ) remain hypotheses; E16 tests them at N = 10, and Proposition F.4 below reduces the strong-dephasing case to a single parabolic estimate.
3. *Interacting check (E16).* The dephased XXZ chain at the isotropic point Δ = 1 (N = 10, g = 4, exact) is compared with the quasi-free control Δ = 0 at the same size.
   - Lemma F.0 holds to 4 × 10⁻⁴ of the peak error, which is the quadrature error.
   - The integrand G(u, t − u) has a fixed sign on every diagonal. Hence F_abs equals the envelope of the true error to 10⁻⁴ for all four initial states and both Δ, while Theorem A is 8–85 times looser.
   - On [1, 10] the exponents of the (H_O) factors differ from the control by 0.06, and those of the density gradient and the current by at most 0.07. The bond coherence is nonzero only at Δ = 1, where ZZ breaks the sublattice symmetry, and decays faster (exponents −1.1 to −1.9).
   - The exponents (about −0.9) are steeper than the infinite-chain −1/2 because the slowest mode of the open chain decays at D(π/N)² ≈ 0.05 and everything saturates after t ≈ 20. The comparison between Δ = 1 and Δ = 0 at the same size is the meaningful one.
   - The preregistered comparison of growth on [2, 10] (F_abs against A) fails for the lowest density mode, whose error itself grows until t ≈ 1/(Dk²) ≈ 20. That criterion did not test time uniformity.

### Proposition F.4 (interacting chains in the strong-dephasing limit: time uniformity reduces to a parabolic estimate)

**Setting.** The dephased XXZ chain of E16: H = ½ Σ_b (X_bX_{b+1} + Y_bY_{b+1} + Δ Z_bZ_{b+1}) with hopping amplitude J = 1 on an open chain, number dephasing at rate g (a hop coherence decays at rate g), the uniform hopping miscalibration J → (1 + ε)J, and the observable n_C.

**Statement.**
1. *Reduction.* To leading order in J/g, adiabatic elimination of the bond coherences turns the diagonal of the density matrix into a classical exclusion process. A particle hops across bond b at rate w_b(η) = 2J²g/(g² + E_b(η)²), where E_b = Δ(Z_{b−1} − Z_{b+2}) is the ZZ energy cost of the hop (Z = 1 − 2n; a missing neighbour at an open end contributes 0). At Δ = 0 this is the symmetric simple exclusion process (SSEP) with D = 2J²/g; for Δ ≠ 0 it is a symmetric Katz–Lebowitz–Spohn (KLS) process. Both are reversible, because w_b does not change under the hop.
2. *Time rescaling.* Every rate is proportional to J², so the miscalibrated process is the reference process run at speed λ = (1 + ε)². For every initial distribution the error is
   Δ_cl(t) = n_C(λt) − n_C(t),  |Δ_cl(t)| ≤ (λ − 1) sup_{s∈[t,λt]} s |ṅ_C(s)|.
   Hence the error is time-uniform if and only if the parabolic estimate P = sup_{t>0} t |ṅ_C(t)| < ∞ holds, and then sup_t |Δ_cl| ≤ (λ − 1)P ≈ 2εP.
3. *SSEP (Δ = 0).* P ≤ (A/2) osc(n_0) with the universal constant A = sup_t t‖∂_t p_t‖_{ℓ¹} = 0.527 of the continuous-time simple random walk, for every chain length and every initial profile. So sup_t |Δ_cl| ≤ 0.264 (λ − 1) osc(n_0).
4. *KLS (Δ ≠ 0).* Duality fails, and P < ∞ uniformly in N is a microscopic parabolic-regularity estimate for a reversible exclusion process that is not of gradient type. We state it as a conjecture.
   - Its hydrodynamic version is proven (Lemma F.5), and so is its near-equilibrium version (Remark 4).
   - E16c supports it up to N = 128 (Remark 6).

*Proof.*
- Item 2: the generator of the miscalibrated process is λW, so e^{tλW} gives n_C(λt). The mean-value theorem gives n_C(λt) − n_C(t) = (λ − 1)t ṅ_C(ξ) with ξ ∈ [t, λt], and t ṅ_C(ξ) = (t/ξ) ξ ṅ_C(ξ) with t/ξ ≤ 1.
- Item 3: in SSEP the one-point function obeys n_C(t) = Σ_y p_t(C, y) n_0(y), with p_t the kernel of a single continuous-time random walk with the same rate (self-duality). Since Σ_y ∂_t p_t(C, y) = 0, subtracting the midpoint of n_0 gives |ṅ_C(t)| ≤ ‖∂_t p_t(C, ·)‖₁ osc(n_0)/2.
  - On ℤ the kernel is p_t(x) = e^{−2wt} I_x(2wt), and t‖∂_t p_t‖₁ depends on wt only. Its supremum is A = 0.527, reached at wt ≈ 0.40; the Gaussian limit is 2/√(2πe) = 0.484 (computed with modified Bessel functions).
  - On an open segment the reflecting kernel is a sum over image points that partition ℤ, so the same ℓ¹ bound holds. □

**Remarks.**
1. *Relation to Proposition F.1.* At Δ = 0 Proposition F.1 already proves time uniformity for every g, with constants that depend on g/J and a bound proportional to the variation V(n_0). At leading order in J/g the reduction gives an explicit constant that depends neither on V(n_0) nor on N.
2. *Check (E16b).* Exact master equation on the 2^N configurations, g = 4, ε = 0.01, N = 10, 14, 18, Δ = 0, 1, 2, the four initial states of E16.
   - The reduction is quantitatively relevant. At N = 10 the classical error Δ_cl reproduces the exact quantum error of E16 within ratios 1.01–1.36 for all four states at Δ = 0 and 1 (preregistered: within 1.5). The largest deviation is for the Néel state, the sharpest profile.
   - P = sup_t t |ṅ_C| does not depend on N for the domain wall (0.132, 0.127, 0.119 at Δ = 0, 1, 2) and the Néel state (0.184, 0.190, 0.203), and it decreases with N for the lowest mode.
   - The preregistered no-growth test fails for the random profile (a factor 1.6–1.7 from N = 10 to 18). The same growth appears at Δ = 0, where P is proven bounded: the random occupations near C change with N, so that criterion compared different profiles.
   - Post hoc, at fixed N, P(Δ)/P(0) = 0.90–1.10 for every state and size, and every P is below the SSEP bound 0.264 osc(n_0). The classical errors sup_t |Δ_cl| = 0.0024–0.0040 lie below the SSEP bound 0.264(λ − 1) = 0.0053.
3. *What is open, and what the literature gives (checked 2026-11-07).*
   - *The class.* The KLS process here is a speed-change exclusion process that is reversible under every Bernoulli product measure. The rate of bond b depends only on η_{b−1} and η_{b+2}, not on the bond's own occupations, and lies in [w_min, w_max] = [2J²g/(g² + 4Δ²), 2J²/g].
   - *Not gradient.* For Δ ≠ 0 the process is not of gradient type. Write w_b = α + β(η_{b−1} + η_{b+2} − 2η_{b−1}η_{b+2}). The current then contains the three-point term η_{b−1}η_{b+2}(η_b − η_{b+1}), and its sum over the bonds of a ring does not vanish.
   - *Known results for the class.*
     - The hydrodynamic limit ∂_s ρ = ∂_x(D(ρ)∂_x ρ): Varadhan's non-gradient method, and Funaki, Uchiyama and Yau (1996) for speed-change exclusion.
     - Smoothness of D in the density: Bernardin, Stoch. Proc. Appl. 101, 43 (2002).
     - Decay to equilibrium in the stationary measure: Gu and Yang (arXiv:2509.20797, 2025) prove Var_ρ[P_t u] = C_u t^{−d/2} + o(t^{−(d+δ)/2}) for every local u. Their hypotheses (local rates between two positive constants, independent of the bond's occupations) cover this process.
   - *What the known results do not give.* Every statement above is about the stationary Bernoulli measure in infinite volume. None gives an L^∞ bound, a result for non-equilibrium initial data, or a time-derivative estimate. Gu and Yang note that the triple-norm contraction |||P_t F||| ≤ |||F||| behind their SSEP estimates is verified only for symmetric exclusion. That contraction is the analogue of the ℓ¹ bound on ∂_t p_t used in item 3.
   - *So what is open is one microscopic statement:* P_N = sup_t t|ṅ_C| bounded uniformly in N for product initial states far from equilibrium. Lemma F.5 below proves its hydrodynamic version, Remark 4 its near-equilibrium version, and E16c tests it numerically up to N = 128 (Remark 6).
   - The corrections of order (J/g)² to the reduction are also not controlled uniformly in time here.

### Lemma F.5 (the parabolic estimate holds for the hydrodynamic equation)

**Statement.** Let D ∈ C^{1,α}([0, 1]) with 0 < d_− ≤ D ≤ d_+, and let ρ be a bounded weak solution of ∂_s ρ = ∂_x(D(ρ) ∂_x ρ) with 0 ≤ ρ ≤ 1, either on ℝ × (0, ∞) or on an interval with no-flux boundary conditions. Then
   s |∂_s ρ(x, s)| ≤ C_*  for every x and every s > 0.
The constant C_* depends only on d_±, α and ‖D‖_{C^{1,α}}. It does not depend on the initial data, the time or the length of the interval.

For the KLS process of Proposition F.4:
- w_min ≤ D ≤ w_max, because the variational formula for D is monotone in the rates and equals the rate for SSEP.
- D(0) = D(1) = w_max: an isolated particle or hole hops with E = 0.
- C^{1,α} up to ρ = 0 and 1 is assumed here. Bernardin's theorem gives smoothness in the density, but its endpoint form was not checked.

*Proof.*
- *Rescaling.* Fix (x₀, s₀), set R = √s₀ and define ρ̃(y, τ) = ρ(x₀ + Ry, R²τ). Then ∂_τ ρ̃ = R² ∂_s ρ = ∂_y(D(ρ̃) ∂_y ρ̃). Because D depends on ρ only, the equation is invariant under parabolic scaling. So ρ̃ solves it on Q = (−1, 1) × (1/2, 1] with 0 ≤ ρ̃ ≤ 1, and ∂_τ ρ̃(0, 1) = s₀ ∂_s ρ(x₀, s₀).
- *Interior regularity on Q.*
  1. View a = D(ρ̃) ∈ [d_−, d_+] as a measurable coefficient. De Giorgi–Nash–Moser gives ‖ρ̃‖_{C^β(Q_{3/4})} ≤ C(d_±).
  2. Then a ∈ C^β with norm at most ‖D′‖_∞ C(d_±). The Schauder estimate for divergence-form equations gives ∂_y ρ̃ ∈ C^β(Q_{1/2}).
  3. In non-divergence form, ∂_τ ρ̃ = D(ρ̃) ∂_y² ρ̃ + D′(ρ̃)(∂_y ρ̃)². The coefficient and the source are Hölder, so the parabolic Schauder estimate bounds ‖ρ̃‖_{C^{2+γ, 1+γ/2}(Q_{1/4})} by a constant C_* of the stated kind. In particular |∂_τ ρ̃(0, 1)| ≤ C_*.
- *Boundary.* With a no-flux boundary, even reflection across the ends extends ρ to a periodic solution of the same equation, and the argument applies unchanged.
- References: Ladyzhenskaya, Solonnikov and Ural'tseva (1968), chapters III–V; Lieberman (1996). □

4. *Near-equilibrium microscopic version.*
   - Let ν be a Bernoulli product measure, so W is self-adjoint in L²(ν) for every N, and let μ be an initial law with density h = dμ/dν. The spectral theorem gives
     t |ṅ_C(t)| = t |⟨h − 1, W e^{tW} n_C⟩_ν| ≤ (2/e) ‖h − 1‖_{L²(ν)} Var_ν(e^{(t/2)W} n_C)^{1/2} ≤ (1/e) χ²(μ‖ν)^{1/2}.
   - This bound is uniform in N whenever the chi-square divergence χ²(μ‖ν) is, for example for initial states that differ from equilibrium on a fixed finite set. On ℤ, Gu and Yang's variance decay makes it decay like t^{−1/4}.
   - For a domain wall, a Néel state or a random profile, χ²(μ‖ν) grows exponentially with N. The open case of Proposition F.4 is therefore exactly the far-from-equilibrium one.
5. *What Lemma F.5 means for Proposition F.4.* Suppose the KLS one-point function converged to the hydrodynamic solution together with its time derivative, uniformly at all times beyond O(1). Then P_N ≤ C_* + o(1). The missing link is this convergence for single-site time derivatives. The hydrodynamic limit gives only spatially averaged densities at fixed macroscopic times.
6. *Check (E16c, preregistered).*
   - *Setup.* Continuous-time Monte Carlo of the same process, 10⁵ runs per case, N = 18, 32, 64, 128, Δ = 0, 1, 2, observed site C = N/2 − 1. Initial states: domain wall, Néel, and random occupations centred on C (the same values around C for every N). This removes the defect of E16b (b).
   - *Estimator.* t ṅ_C comes from the generator at small t and from log-time differences at large t, whichever has the smaller standard error.
   - *All four preregistered criteria hold:*
     - (a) at N = 18 the Monte Carlo P agrees with the exact master equation in all nine cases;
     - (b) P does not change from N = 32 to N = 128 beyond the statistical error in any case (for example Néel, Δ = 2: 0.203 → 0.205; wall, Δ = 2: 0.122 → 0.119; random, Δ = 2: 0.065 → 0.066);
     - (c) every P lies below the SSEP constant 0.264 osc(n_0); the largest is 0.205 (Néel, Δ = 2);
     - (d) at N = 128 the values on 100 ≤ t ≤ 1000 stay at the noise level (≤ 0.025), with no late regrowth.
   - P(Δ)/P(0) = 0.90–1.12 at every size. The supremum is reached at t = 0.4–10 (units of 1/J), so it is set by local relaxation, not by the system size.

## Data checks on the BEQS 10-ion data (E15, E11e, E17, E11f, E11g, E22, E22b, E23, E23c, E23k, E23s, E23d, E24, E24b, E25, E25b, E26)

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

**What acts in the first milliseconds, and what the late record shows (E22, E22b, 2026-11-07).** Both initial states are transverse product states: state 1 has X ≈ 0.87 and Z ≈ −0.45, state 2 has Y ≈ 0.99 and Z ≈ 0.05.
- *Learning window (E22 (A), diagnostic).* Six candidate generators were added to Xb15uf and learned with all 435 observables on t ≤ 1.5 ms, with a bootstrap over the settings.
  - An isotropic flip burst (τ = 0.42 ms) gets S_b = 30 ± 9 rad/s, against 144 rad/s from the mean-Z balance alone (E11h), and lowers χ² by only 138. Bursts of local z dephasing or of a uniform field do little.
  - The data prefer transverse charge-2 terms: an anisotropic burst, a constant anisotropic single-ion block, or a collective transverse block with ion–ion correlations (qerr.learning.correlated_adjoint). These lower χ² by 1100–1600, but every such 2×2 block is non-PSD at 3σ or more (smallest eigenvalues −33 ± 10, −16 ± 3, −8 ± 3 rad/s).
  - So no physical Markovian generator of these families describes the first milliseconds.
- *Late record (E22 (B)).* Three features lie 10–25σ from every learned model and every E11f truth:
  - (i) M² grows. The mean ZZ over all pairs, (M² − N)/(N(N − 1)), rises in both states (3.2 ± 0.3 and 4.7 ± 0.2 × 10⁻³ per ms). Every U(1)-covariant generator conserves it and independent flips pull it towards zero, so the growth requires U(1)-breaking processes that are correlated between ions. This matches the excess growth of Var M in the 51-ion data (E6b).
  - (ii) State 1 has a charge-2 floor. Across the 45 pairs, (XX − YY)/2 ≈ −0.045 C_ij and (XY + YX)/2 ≈ +0.051 C_ij with one common coefficient, where C = (XX + YY)/2. State 2 has no floor. The pattern is that of a non-conformal X/Y readout map, but a contrast mismatch alone would need a Y contrast above 1, given the length of the measured state-1 Bloch vector at t = 0.
  - (iii) The mean-Z level. A burst model reproduces state 1, and a model without transverse fields reproduces state 2; the learned uniform h_x ≈ −4 rad/s pulls the state-2 magnetisation down where the data stay flat.
- *Collective flips (E22b, preregistered).* Jump operators √g Σ_k X_k and √g Σ_k Y_k give d⟨Z_iZ_j⟩/dt = 4g⟨X_iX_j + Y_iY_j⟩ − 8g⟨Z_iZ_j⟩.
  - The late M² slope of each state fixes g: 2.2 and 1.5 rad/s.
  - Each state was predicted with the rate of the other state. The overall calibration factor drops to 0.88, the best so far, and the charged classes improve. But the neutral classes stay at 1.26, 1.49 and 1.20, so the rule of E20 fails.
  - A constant collective rate makes M² grow too fast while the transverse correlations are large (first 2 ms). The correlated process must be weaker early, or only partially correlated between ions.
- *The charge-2 imbalance (E22 (C)).*
  - Fitted per initial state against the same fit on 20 control realisations (preregistered), the excess of K lies in state 2 (−152.5 rad/s, below the control's 5th percentile of −98.6). State 1 is inside its control range (−140.6; 5–95% −160 to −85).
  - Removing the state-1 floor with a fitted readout map makes K more negative (−151.8 rad/s).
  - Adding the ion–ion correlated block leaves K unchanged (−94.4).
  - The imbalance therefore remains unexplained. It comes from state 2 and is reproduced by none of the Markovian generators or simulated truths tested so far.

**Shape of the correlated process, the neutral error, and the sources of K (E23, E23c, E23k, E23s, 2026-11-09).**
- *Shape (E23, diagnostic, bootstrap over settings).*
  - On t ≥ 2 ms the mean ZZ over all pairs grows linearly in state 2 (curvature −1 ± 1 × 10⁻⁴ ms⁻²) and decelerates in state 1. A quasi-static global transverse field would make it accelerate (t²), so it is not supported.
  - The connected part grows at 4.2 and 4.8 × 10⁻³ per ms in the two states.
  - The per-pair connected slopes do not decay with the ion distance; the farthest pairs grow fastest. The process is global, with a correlation length of at least the chain.
  - The slopes are not proportional to the pair's transverse correlation (correlation −0.30 ± 0.09 and −0.25 ± 0.08), unlike Markovian collective transverse noise. Pair patterns are, however, also redistributed by the U(1) dynamics.
  - State 2 shows a single excursion of the connected part at 0.5 ms (+0.023 ± 0.003).
- *The neutral error (E23).*
  - The Z_iZ_j error lies in state 2 (factors 1.76–1.86 for Xb15, Xb15u, Xb15uf and Xb15uc) and in directions orthogonal to the conserved sum (factor of the rest 1.25–1.36). The largest residuals involve the edge ions. No change of the U(1)-breaking processes can bring that class below 1.2.
  - Models learned on longer windows predict the same out-of-sample times worse (neutral factors on t > 3 ms: Xb15 1.49–1.52, Xb30 1.85–2.31, X50 1.83–2.48). This points to misspecification, not to a lack of data.
  - The learned current terms (XY − YX)/2 (rms 30–58 rad/s) are not the cause. Dropping them raises the window χ² by 2494 for 45 parameters and leaves K at −93.4.
- *Coupling-scale noise (E23c, preregistered, chosen by a screening on the same data).*
  - Model: a shot-to-shot Gaussian scale of all couplings, with σ calibrated on the other state (0.0825 and 0.12, the latter at the grid edge).
  - Factors: Z_i 1.213, Z_iZ_j 1.188, flip-flop 1.130, all 0.847. The rule of E20 fails on Z_i.
  - A post hoc check rejects the mechanism. On t ≥ 4 ms the temporal variance of the neutral observables (shot noise removed) matches the undamped Xb15uf (ratios 0.58–1.13), while the noise model reduces it 3–12 times. The data's late oscillations are undamped; the improvement came from replacing wrong-phase oscillations by their mean.
  - The calibration factor rewards such smoothing, so model comparisons must also check the late oscillation amplitude.
- *Generators per initial state (E23s, preregistered).*
  - Xb15uf learned on each state alone: neutral factors 1.69, 1.61, 1.62.
  - One joint generator with a global detuning and a collective rate of its own for state 2: 1.75, 1.61, 1.63. The state-2 detuning comes out at −125.6 rad/s, matching the single-state difference.
  - Both fail. More flexible models fit the window better and extrapolate worse. The joint Xb15uf remains the best neutral extrapolator, and its remaining 1.2–1.4 is a limit on the frequency accuracy of this model class.
- *Sources of K (E23k).*
  - The state-2 single-state anomaly comes from its t = 0.5 ms point. Removing it puts state 2 inside its control range, both without memory (−62.3 against −77 to −38) and with memory (0.2).
  - The joint anomaly survives (−154.3 against −65 to −43), while each state alone is consistent with the control. The single-state generators differ: mean z field 191 ± 24 against 68 ± 8 rad/s (5.6σ), γ_c at 2.9σ.
  - A truth with a run-to-run detuning of state 1 reproduces the pattern at about half the size (joint median −101 for δ = 124 rad/s). The preregistered rule fails.
  - Run-specific nuisance parameters (post hoc) halve the excess, leaving about −50 rad/s.
  - K is therefore not evidence for a charge-2 dissipator.

**Estimator bias of the coupling scale and the chance spread of the factor (E23d, diagnostic on saved results, 2026-11-09).** This revises two readings above: part of the window dependence of E23 and about half of the remaining neutral error of E23s come from the estimator, not from the model class.
- *A fixed coupling scale.* Every two-body coupling of Xb15uf is multiplied by a fixed λ (E23c grid, no averaging). The late data prefer λ = 1.069 (state 1) and 1.104 (state 2); this oracle uses t > 1.5 ms and serves as a diagnostic only. Between λ = 1 and 1.04 the late temporal variance of the neutral observables changes by 3–6%, so the gain is a phase correction, not smoothing (contrast E23c).
- *The integral method is an errors-in-variables regression.* Its design matrix A is built from measured correlators. With A = A₀ + E and y = A₀θ + ε, least squares gives to first order E[θ̂] ≈ (A₀ᵀA₀ + Σ_AA)⁻¹(A₀ᵀA₀θ + Σ_Ay), with Σ_AA = E[EᵀE] and Σ_Ay = E[Eᵀε], and shrinks the couplings.
  - Three estimates that use the learning window only agree on the shrinkage of J_nn: the bootstrap of Xb15 (3.4%), the SBC of E11b with independent Gaussian noise (2.5–4.0%) and the shot-level SBC of E14 (3.1–4.3%).
  - The shrinkage scales with the noise variance (0.20% at a quarter of the noise amplitude, 2.48% at the full noise; E11b, white truth), the signature of errors in variables rather than misspecification.
  - At the a-priori scale λ = 1.040 (E14 mean) the neutral factors become 1.16, 1.23 and 1.06 (from 1.34, 1.41 and 1.25); the charged classes 0.97, 1.06 and 0.68; all 435 observables 0.95. This removes 43–75% of the neutral excess above 1. It approximates a per-parameter correction by one common scale.
  - The shrinkage grows with the window (E14: 3–4%, 6–7% and 9–11% on [0, 1.5], [0, 3] and [0, 5] ms), a little over a third of the measured drift of J relative to the 1.5 ms window (−8% and −20%).
  - The collective parameters carry the same bias: γ_c is 30 rad/s at the centre and 40 ± 5 in the bootstrap mean, β is 7.6 and −8.5 ± 12.8 rad/s per ms. The parameter-level corrections, γ_c ≈ 20 rad/s and β ≈ 24 rad/s per ms, agree with the SBC estimate of β in E11b (22–31). These parameters do not act on the neutral block (Lemma G.0).
  - A prediction-level bootstrap correction (twice the central prediction minus the mean of the replicates) makes Xb15 worse (neutral factors 1.47–1.50 → 1.70–1.82): averaging oscillations of slightly different frequencies damps them. Corrections belong to the parameters.
- *Chance spread of the factor.* If the Xb15 band were exactly right (truth = central prediction + a draw from the covariance of its 12 replicates + shot noise; 4000 draws), the neutral factors have 90% quantiles 1.28, 1.21 and 1.24 and 95% quantiles 1.39, 1.29 and 1.32. All three neutral classes stay at or below 1.2 with probability 0.81, and the full rule of E20 passes with probability 0.71. The observed Xb15uf factors have p = 0.067, 0.017 and 0.088; at λ = 1.040, p = 0.20, 0.088 and 0.31. The jackknife error of each factor over the 12 replicates is 0.05–0.07.
- *Pure-gauge currents.* Phases arg(J_ij + iD_ij) = φ_i − φ_j on the 24 pairs up to distance 3 give χ² = 24.6 for 15 degrees of freedom: weak support, not used.
- *Consequences.*
  - (i) The learning step needs an estimator that is consistent under errors in variables. Candidates are corrected least squares, (AᵀA − Σ_AA)θ = Aᵀy − Σ_Ay, with the covariances computed from the raw shots including the correlations of strings from one setting; SIMEX; and regressors built from the model trajectory (instrumental variables).
  - (ii) A predictive band is √(variance + bias²); the bootstrap spread alone omits the bias.
  - (iii) Decision rules need null quantiles with familywise control (as E18b), not fixed thresholds.
  - (iv) If a slip at the start remains, the comparison bound for CPTP semigroups and a channel S is ‖e^{tL}S − e^{tL̂}‖⋄ ≤ ‖S − id‖⋄ + t‖L − L̂‖⋄; the slip adds a constant, not a growing term.
  - (v) The symmetric error bounds (Theorems A, B, E, F, G) are unaffected; the defects found here lie in the learning and the decision rules.
  - The remaining half of the scale shift that the late data prefer is open.

**Errors-in-variables correction of the learning (E24 preregistered, E24b post hoc, 2026-11-09).** The rules were committed before the run (94a2516).
- *Estimator.* SIMEX on the Xb15uf least-squares fit (E20 structure, window, defect and fixed flips). The setting bootstrap is not a clean noise doubling for this design: a resample of the 200 settings leaves some weight-3 strings unmeasured and drops about 30% of the rows. The noise is therefore added by resampling the 200 shots of every (setting, time), which keeps the design and the correlations between strings of one setting (qerr.shots.resample_shots). Level ζ = m resamples m times in a chain; the linear extrapolation is 2φ₀ − φ̄₁ (qerr.learning.simex_extrapolate).
- *Synthetic data with a known truth (preregistered: pass).* E11f control truth, 12 realisations, 40 chains each, oracle defect. The relative error of J_nn is −3.32 ± 0.46% for least squares, +0.92 ± 0.52% for linear SIMEX (criterion 1%) and +1.15 ± 0.53% for quadratic SIMEX. The coupling scale over the 45 pairs is 0.961, 1.000 and 1.002. Linear SIMEX also restores γ_c (12.3 → −0.4 rad/s against 0) and β (16.0 → 36.3 against 38.2 rad/s per ms).
- *Data (preregistered: pass by rule B; rule A fails).* With 100 chains the couplings rise by 4.5% (J_nn 599 → 627 rad/s), γ_c falls from 30 to 15 rad/s, and β rises from 5.1 to 31.4 rad/s per ms, inside the SBC-calibrated estimate of E11b (22–31).
  - The neutral factors become 1.27, 1.39 and 1.23 (from 1.34, 1.41 and 1.25). The charged classes become 0.51, 0.67 and 0.54 (from 0.90, 1.00 and 0.65), and all 435 observables 0.83 (from 0.99).
  - Rule B passes (null p = 0.103, 0.021 and 0.109 against the Holm threshold 0.0167), and the no-smoothing guard holds (1.04). The pass is narrow: as learned the smallest p was 0.0165.
- *Parts of the correction (E24b, post hoc).* Per-parameter SIMEX raises the couplings at distance 1–2 by 4–5% but the currents (XY − YX) by 46%.
  - On the synthetic data the currents are hardly identifiable in the window: slope 0.26 against the truth (truth rms 87 rad/s). Per-parameter SIMEX raises their error from 101 to 122 rad/s and the error of the couplings at distance ≥ 3 from 35.0 to 37.2.
  - One common SIMEX scale of all two-body terms (1.040 ± 0.002 per realisation) together with SIMEX for the collective rates gives the smallest coupling error (34.5 against 35.6 and 35.7 rad/s), an unbiased scale (1.000), currents as accurate as least squares (102.9) and γ_c error 3.4 (from 12.8).
  - On the data (variants built from the E24 parameters, evaluated on the same late data):
    - currents as learned: 1.15, 1.34, 1.13; all 0.80;
    - SIMEX only for J (d ≤ 2) and the rates: 1.20, 1.29, 1.14; all 0.80;
    - common scale 1.045 with SIMEX rates: 1.14, 1.21, 1.05; charged 0.54, 0.66, 0.54; all 0.76; null p = 0.23, 0.10, 0.33.
  - The common coupling scale corrects the neutral (phase) sector and the corrected collective rates the charged (damping) sector.
- *Consequences.*
  - The errors-in-variables bias of the integral method is real and removable.
  - A per-parameter correction amplifies noise along directions the window does not identify. The standard measurement-error remedy, a common reliability ratio estimated once and applied to all couplings, is the better estimator here.
  - Because the structured variant was chosen after E24, its numbers on the data are exploratory. A confirmation needs a preregistered coverage test of the whole procedure on synthetic data, or new data.
  - The band must be rebuilt for the corrected estimator; with the Xb15 band the charged factors are 0.5–0.7, i.e. the band is now too wide.

**The structured estimator as the learning method (E25 preregistered, E25b post hoc) and a detuning drift across the settings (E26 preregistered), 2026-11-09.** The rules were committed before the runs (11a42c9, e2a7f9e).
- *Estimator S.* One common SIMEX scale λ for the flip-flop couplings, the currents set to zero (step 0: rms error 87.5 against 102.9 rad/s for the λ-scaled currents on the 12 realisations of E24), SIMEX for γ_c and β, fields as learned. Band: 12 shot-level bootstrap replicates of the learning window, each put through the whole procedure (20 chains) and predicted for both states.
- *Synthetic data, end to end (preregistered rule S: fail).* Eight realisations of the E11f control truth, 40 chains each: λ = 1.041 ± 0.001 and β = 35.2 (least squares 15.6, truth 38.2 rad/s per ms). With the step-1 band the pooled neutral factors are 2.01, 1.77 and 2.40, above the 98.3% null quantiles 1.24, 1.23 and 1.26.
  - The spread across realisations is 1.05, 1.04 and 1.02 times the band; the bias of the mean prediction is 1.75, 1.47 and 2.20 times the band. The pooled factor obeys F² ≈ b² + s², where b and s are the bias and the spread in band units: a variance-only band gives F = √(1 + b²) (consequence (ii) of E23d, now measured).
- *Data (preregistered rule B′: fail).* S has λ = 1.045 and β = 31.4 rad/s per ms. The new band is narrower than the Xb15 band (rms ratio 0.45, 0.53 and 0.38 in the neutral classes; 0.67–0.77 in the charged ones). The neutral factors are 2.11, 1.65 and 1.86 (Holm rejects all three, p ≤ 0.00025), the charged 0.79, 0.84 and 0.68, all 435 observables 1.03. The guard holds (1.08).
  - The bootstrap replicates have β = 7.8 ± 14.0 rad/s per ms (mean ± sd), far below the centre: in the bootstrap world the data play the truth, so SIMEX inside a replicate targets the uncorrected data estimate (5.1). This shifts the centre of the replicates, not their spread.
- *Where the bias comes from (E25b, post hoc).* The truth's parameters are predicted with one group replaced by its mean estimate over the eight fits (late neutral bias in band units, Z_i, Z_iZ_j, flip-flop):
  - truth (route check): 0.000 in every class;
  - currents set to zero: 1.62, 1.28, 2.01;
  - couplings of S: 0.56, 0.65, 0.82 (least squares: 1.27, 1.28, 1.41);
  - z fields: 0.21, 0.09, 0.11;
  - all of S: 1.81, 1.55, 2.28, against the measured 1.75, 1.47, 2.20. The single groups add linearly (rms ratio 0.99–1.01).
  - So the common scale removes most of the coupling bias, and fixing the currents, which the window does not identify (E24b: slope 0.26), creates a larger one. A shot bootstrap cannot see it, because every replicate fixes the currents to the same value. The choice in step 0 minimised the parameter error; the late neutral predictions are sensitive to directions that the parameter error weighs little.
  - The near calibration of the Xb15 band in the neutral classes (pooled factors of least squares 0.93–1.06 on the same synthetic data) is a cancellation: the setting bootstrap relearns the noisy currents and is wider than the spread of the estimator (spread 0.71–0.86 band), and the bias (0.65–0.79 band) adds to it.
- *Detuning drift across the settings (E26).* Part A (data, learning window only): the between-setting variance of the per-setting means over the binomial shot variance is 1.26, 2.04, 2.33 and 2.16 for weight-1 strings at t = 0, 0.5, 1.0 and 1.5 ms (weight 2: 1.15–1.35; weight 3: 1.08–1.17). A global rotation angle fitted per setting has a spread that grows linearly in t (state 1: 0.06, 0.15, 0.29 rad), agrees between 0.5 and 1.0 ms setting by setting (Spearman 0.87 and 0.76), and is correlated along the acquisition order (lag-1 Spearman 0.80 and 0.58): a slow global detuning of rms 220–270 rad/s, the size of the quasi-static collective noise (√(2β_s) = 276 rad/s). The excess at t = 0 is not a drift and points to a separate per-setting preparation or readout fluctuation.
  - Part B (synthetic, preregistered): with a drift of rms 276 rad/s (AR(1), correlation 0.8) on the control truth the relative error of J_nn is −7.5 ± 1.2% for least squares (−4.0 ± 0.6% without drift). S with shot SIMEX gives −3.9 ± 1.2%, with a setting-level wild bootstrap −3.2 ± 1.3% (i.i.d. or AR(1) weights); without drift every noise model is within 0.5%. No noise model meets the 1% criterion in both worlds, so part C (data) was not run.
  - Writing the shrinkage as 1/(1 + r) and solving the linear extrapolation, the wild bootstrap adds about 30% of the drift's effective noise. A drift cannot be added to measured data: a rotation mixes X and Y of one ion, while a setting measures each ion in one basis. Removing the drift's share of the shrinkage therefore needs a model-based simulation (indirect inference or a parametric bootstrap).
  - The data's drift variance is 0.45–0.66 times that of the synthetic drift world (weight-1 excess at 1.0–1.5 ms, with and without the t = 0 excess). To first order in the noise variance the shrinkage of least squares on the data is then about 6%, and the scale needed is about 1.06–1.07, near the value the late data prefer for state 1 (1.069, E23d). This is an estimate under that assumption.
- *Consequences.*
  - A predictive band must carry the bias: the bias from fixing a non-identified direction (here the currents) and the bias from noise that resampling cannot reproduce (here the drift). Marginalising the non-identified directions over a hierarchical prior, or giving them a physical structure (pure-gauge phases J_ij e^{i(φ_i − φ_j)}, nine parameters; weak support in E23d), puts the first into the band. A parametric simulation with the measured drift puts the second into the estimator and the band.
  - The choice between estimators has to be made on the prediction error of the observables of interest, not on the parameter error.
  - Step 3 of E25 (an initial slip, run parameters with priors) is deferred: a failure of the same size occurs in a synthetic world without slip or runs, so a slip would absorb estimator bias.
  - The symmetric error bounds (Theorems A, B, E, F, G) are unaffected.

**What E25 and E26 change for the theorems (2026-11-09, Q30; no new run, one check on cached predictions).**
- Theorems A, B1–B4, B3′, B6, E, F and G bound the effect of a given model error. E25 and E26 concern how that error is estimated from data, so they leave these statements unchanged.
- Theorems A2 and A3, and the first-order budget B1′ built on them, assume a zero-mean Gaussian model error. The comparison with Eq. (12) of BEQS is made under the same hypothesis and stands. The learned models violate it: in the synthetic control world, which contains no drift and no slip, the late neutral predictions of S are biased by 1.5–2.2 times the rms of its band (E25). Applied to data, these statements need a separate bias term. Theorem A applied to the mean error bounds that term, and the bound saturates when the observable forgets.
- Conserved observables are insensitive to the symmetric part of the bias (Remark A.3). On the cached E25b predictions (U(1)-symmetric truth), replacing one symmetric group (currents set to zero, the couplings of S or of least squares, the z fields) changes mean Z by at most 2×10⁻¹⁶ at every time, while the single-site Z_i move by 0.002–0.024 (rms, t > 1.5 ms). Only the full mean estimate of S moves mean Z (at most 4×10⁻³), through the transverse fields it keeps from learning (rms 2.5 rad/s, zero in the truth). The certified band of mean Z (E15) is therefore the data statement least affected by E25.
- Statements to rescope on the data side: the bootstrap band captures the spread of the estimator, not its mean, and its earlier neutral calibration was a cancellation (E25b). The window drift of J (617, 573, 504 rad/s for 1.5, 3 and 5 ms) is partly estimator bias (shrinkage 3–4, 6–7 and 9–11% in-model), so it cannot by itself show that no static Markovian generator fits all times; the rejection of white collective noise (E14), the setting-level drift (E26) and the conservation-law violations (E17) can. The certified γ_c ≥ 20.7 rad/s is an effective white rate.

**What E27 and E28 change (2026-11-10, Q31; E27 and E28 preregistered in 9679fa0, E27b and E27c post hoc).**
- *Marginalising the non-identified currents (E27, E27b).*
  - The currents (XY − YX) were given an independent prior N(0, τ_g²) per pair, with τ_g the rms of the least-squares currents in each distance group. The prior was carried into a parametric-bootstrap correction (S_D) and into the band. On the data the band is 1.1–2.4 times the Xb15 band in the neutral classes, and the data fall inside it (factors 0.47–1.08, rule B′ passes).
  - In the synthetic drift world the band fails rule S″ in Z_iZ_j (pooled factor 1.31 against 1.17) and is too wide in Z_i and flip-flop (0.79, 0.54).
  - The band is almost entirely the current prior: currents alone spread the neutral predictions by 1.0–1.2 band widths. The truth's currents, which were learned from the data, shift the neutral classes by only 0.27–0.63 of that spread when set to zero, below all 200 prior draws of the same rms.
  - A plausible reading: learned currents are largest along the directions the window does not see, which are also the directions the observables barely feel. An i.i.d. prior matched to their rms therefore overweights the directions that matter. A prior shaped like the window's posterior (information matrix of the least-squares system plus the hierarchical τ) would not.
  - A coverage test for such a band must draw the truth's currents anew in every realisation. Realisations that share one truth do not average the common bias, so a null of independent band draws is too narrow on both sides.
- *Drift-induced shrinkage (E27, E27c).* Over 16 realisations per level (E27c), the drift shrinks the couplings of S by 2.3 ± 0.9% at the data's drift size (217 rad/s) and by 2.4 ± 0.8% at 276 rad/s (−1.7 ± 0.4% at 217 rad/s for a fit proportional to σ_d²). Least squares shrinks by the same amount, about 7% in total at the data's drift, as E26 estimated. The larger effect on a single data set is the scatter: under the drift the coupling scale of S varies with sd 3.7% across realisations, against 0.8% without it. A shot bootstrap cannot see this scatter; a parametric simulation with the drift can (E27's band carries it, replicate sd 3.1% on the data). The eight realisations of E27 step B were atypically high (Welch p = 0.012 against the 16 new ones; same code, other seeds), so the apparent absence of shrinkage there, and part of the Z_iZ_j bias of S_D, are properties of that sample. The parametric correction of S_D has the right sign and a smaller size (−0.9 to −1.3%) than the direct measurement.
- *Checks of Q30 (E28).*
  - C1: E9's lower bound on the effective white rate is not conservative under the data-matched drift (6 of 8 realisations below the target, 15.3 rad/s). The finite-shot bias of the estimate (+3.5 rad/s) is only partly seen by the setting bootstrap (+2.1–2.5). The statement "γ_c ≥ 20.7 rad/s certified" is withdrawn. The Zeno factors were already re-certified with the worst case of the mixed model (§9.1 of the results summary), so Theorem B6 and its use are unaffected.
  - C2: U(1)-breaking terms of twice the E6a bootstrap sd (K = (XX − YY)/2, L = (XY + YX)/2, transverse fields) are recovered in the learning window with slopes 0.92, 0.83 and 0.97. The E6a null result therefore has power (shot-only world).
  - C3: in-model estimator bias plus the data-matched drift explains 37–41% of the window dependence of J_nn (data −7.9% and −19.7% for the 3 and 5 ms windows; synthetic −3.2 ± 0.8% and −7.3 ± 0.8%). The remainder, about −12 percentage points at 5 ms, is evidence against a static Markovian generator once the bias share is subtracted.
- The theorems are unchanged. What changes is the data side: no validated band yet for the late neutral classes, no certified lower bound on γ_c, and a firmer U(1) null result and window drift of J.
