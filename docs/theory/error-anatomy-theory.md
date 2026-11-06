# Error anatomy of open quantum simulation under imperfect Hamiltonians — theory note (v0, 2026-11-04)

Status: statements and proofs for the PRA draft. Every statement is checked numerically by the experiments listed in §6. Notation follows the spec `docs/superpowers/specs/2026-11-04-dissipative-error-anatomy-design.md`.

## 0. Results at a glance

| Result | Assumption | Statement (schematic) |
|---|---|---|
| Thm A (Heisenberg forgetting) | $\mathcal L,\mathcal L'$ valid GKSL | $\lvert\Delta(t)\rvert\le\int_0^t\lVert[\delta H,O_u]\rVert du\le 2t\lVert\delta H\rVert\lVert O\rVert$ (the right-hand side is BEQS Eq. 14) |
| Cor A.2 | $\mathrm{osc}(O_u)\le K e^{-u/\tau}\mathrm{osc}(O)$ | $\lvert\Delta(t)\rvert\le 2K\lVert\delta H\rVert\mathrm{osc}(O)\,\tau(1-e^{-t/\tau})$: saturates |
| Thm A2 (expected error) | $\mathbb E g=0$, covariance $\Sigma$ | leading term $4\,\mathrm{osc}(O)\sum_m\sigma_m^2\lVert h_m\rVert^2 K^2\tau^2 f_2(t/\tau)$, which reproduces BEQS $2t^2\sigma^2M$ at short times |
| Thm A3 (concentration) | uniform forgetting on a convex set (Lemma A3.1: a ball, for primitive $\mathcal L$) | width $\sigma\,\mathrm{osc}(O)\sqrt{M}\,c_\delta\,T_{\rm eff}(t)+\mathrm{osc}(O)\delta'$, with $T_{\rm eff}\le\min(t,2K\tau)$ |
| Lemma B.0 | U(1)-covariant $\mathcal L_0$ + collective dephasing $\gamma_c$ | charge-$q$ sector decays exactly as $e^{-2\gamma_c q^2 t}$, with $K=1$ |
| Lemma B.1 | learned dephasing matrix $\Gamma$ | protective collective rate $\gamma_c^*=1/(\mathbf 1^T\Gamma^+\mathbf 1)$ |
| Thm B1 (selection rule) | covariant $\mathcal L$, charge-0 $O,\rho_0$ | charged imperfections give zero first-order response |
| Thm B2 (charged observables) | covariant $\mathcal L$ + $\gamma_c$ | $\lvert\Delta(t)\rvert\le\lVert\delta H\rVert\sum_{q\ne0}\lVert O_q\rVert(1-e^{-2\gamma_cq^2t})/(\gamma_cq^2)$: time-uniform |
| Thm B3 (Zeno, all orders) | covariant $\mathcal L$ + $\gamma_c$, charge-0 $O,\rho_0$, purely charged $\delta H$ | $\lvert\Delta(t)\rvert\le4\lVert\delta H\rVert\mathrm{osc}(O)\sum_{q\neq0}\lVert\delta H_q\rVert\,\varphi(2\gamma_cq^2,t)\approx 2\lVert\delta H\rVert\mathrm{osc}(O)\,t\sum_{q\ne0}\lVert\delta H_q\rVert/(\gamma_cq^2)$ |
| Thm B4 (anatomy) | as B3, general $\delta H=\delta H_0+\delta H_{\rm ch}$ | error = symmetric part (Thm A, not suppressed) + breaking part (B3, Zeno-suppressed) |
| Lemma B.2 (support) | any $\sigma^z$-dephasing matrix $\Gamma$, on-site terms | charge projections, all $\mathcal D_\Gamma$ (collective included) and on-site terms preserve operator support, so they leave Lieb–Robinson constants unchanged |
| Thm B3′ (commutator form) | as B3 | $\lvert\Delta(t)\rvert\le\sum_{p>0}\int_{a+b\le t}e^{-2\gamma_cp^2b}\lVert[\delta H,e^{b\mathcal L_0^\dagger}([\delta H_{\pm p},O_a])]\rVert$: reference dynamics only, light-cone limited, exactly computable for free fermions |
| Cor B3-loc | (LR) bound, local $\delta H=\sum_xh_x$ | $N$-independent: $4\,\mathrm{osc}(O)\,\varepsilon\sum_p\varepsilon_p\int\!\!\int e^{-2\gamma_cp^2b}\,\Xi_A(a,b)$; in 1D this is $\lesssim(\lvert A\rvert+vt)^2\varphi(2\gamma_cp^2,t)$ |
| Prop C | Gaussian rate uncertainty | $\mathbb E[P_{\rm exc}(t)]=e^{-\mu t+s^2t^2/2}$, which exceeds 1 for $t>2\mu/s^2$ |

## 1. Setting and notation

- $\mathcal H\cong\mathbb C^D$. A GKSL generator is $\mathcal L(\rho)=-i[H,\rho]+\sum_k\big(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\}\big)$.
- The Heisenberg dual $\mathcal L^\dagger$ is defined by $\mathrm{Tr}[A\,\mathcal L(X)]=\mathrm{Tr}[\mathcal L^\dagger(A)X]$.
- $e^{t\mathcal L}$ is CPTP, so $\lVert e^{t\mathcal L}(X)\rVert_1\le\lVert X\rVert_1$ for **all** $X$, because the induced 1→1 norm is at most the diamond norm, which is 1.
- $e^{t\mathcal L^\dagger}$ is unital and CP, so by Russo–Dye $\lVert e^{t\mathcal L^\dagger}(A)\rVert\le\lVert A\rVert$ for all $A$.
- **Oscillation.** $\mathrm{osc}(A):=\min_{c\in\mathbb C}\lVert A-c\mathbb 1\rVert$. For Hermitian $A$, $\mathrm{osc}(A)=(\lambda_{\max}-\lambda_{\min})/2$. We always have $[B,A]=[B,A-c\mathbb 1]$, hence $\lVert[B,A]\rVert\le2\lVert B\rVert\,\mathrm{osc}(A)$. Unital positive maps do not increase $\mathrm{osc}$ of Hermitian operators.
- **Forgetting function.** $\kappa_{\mathcal L}(u):=\sup_{X=X^\dagger}\mathrm{osc}(e^{u\mathcal L^\dagger}X)/\mathrm{osc}(X)\le1$. One has $\kappa_{\mathcal L}(u)\le\eta(e^{u\mathcal L})$, the trace-distance contraction coefficient. Proof: for states $\rho,\sigma$ write $\Phi(\rho-\sigma)=\tfrac12\lVert\Phi(\rho-\sigma)\rVert_1(\tilde\rho-\tilde\sigma)$. Then $\mathrm{osc}(\Phi^\dagger X)=\tfrac12\sup_{\rho,\sigma}\mathrm{Tr}[X\Phi(\rho-\sigma)]\le\eta(\Phi)\,\mathrm{osc}(X)$.
- **Imperfection.** The implemented generator is $\mathcal L'=\mathcal L+\mathcal V$ with $\mathcal V=-i[\delta H,\cdot]$ and $\delta H=\delta H^\dagger$. $\mathcal L'$ is again a valid generator. Write $\mathcal V^\dagger(A)=i[\delta H,A]$.
  - Deterministic $\delta H$ models miscalibration (a bias $k$ in BEQS language).
  - Random $\delta H(g)=\sum_{m=1}^M g_m h_m$ models learning uncertainty, with $\mathbb E g=0$ and $\mathbb E g g^T=\Sigma$.
- **Quantities.** $O_u:=e^{u\mathcal L^\dagger}(O)$, $\rho_t=e^{t\mathcal L}\rho_0$, $\rho'_t=e^{t\mathcal L'}\rho_0$, and $\Delta(t):=\mathrm{Tr}[O\rho'_t]-\mathrm{Tr}[O\rho_t]$.

## 2. Duhamel identities

**Lemma 1.** For any generators $\mathcal L$ and $\mathcal L'=\mathcal L+\mathcal V$,
$$e^{t\mathcal L'}-e^{t\mathcal L}=\int_0^t e^{(t-s)\mathcal L}\,\mathcal V\,e^{s\mathcal L'}\,ds \tag{D1}$$
$$\phantom{e^{t\mathcal L'}-e^{t\mathcal L}}=\int_0^t e^{(t-s)\mathcal L}\mathcal V e^{s\mathcal L}\,ds+\int_0^t\!ds\!\int_0^s\!du\; e^{(t-s)\mathcal L}\mathcal V e^{(s-u)\mathcal L}\mathcal V e^{u\mathcal L'}. \tag{D2}$$

*Proof.* Let $F(s)=e^{(t-s)\mathcal L}e^{s\mathcal L'}$. Then $F'(s)=e^{(t-s)\mathcal L}\mathcal V e^{s\mathcal L'}$; integrating gives (D1). For (D2), insert (D1) for $e^{s\mathcal L'}$ into (D1). $\square$

(D2) is exact: the last factor $e^{u\mathcal L'}$ is the full perturbed evolution. Only $\lVert e^{u\mathcal L'}\rho_0\rVert_1=1$ is ever used about it. This is why every bound below holds to all orders.

## 3. General results

### Theorem A (Heisenberg forgetting bound)

For valid $\mathcal L,\mathcal L'$, any state $\rho_0$ and any operator $O$:
$$\lvert\Delta(t)\rvert\le\int_0^t\big\lVert[\delta H,O_u]\big\rVert\,du .$$
More generally, for any perturbation $\mathcal V$ keeping $\mathcal L'$ valid, $\lvert\Delta(t)\rvert\le\int_0^t\lVert\mathcal V^\dagger(O_u)\rVert\,du$.

*Proof.* By (D1), $\Delta(t)=\int_0^t\mathrm{Tr}[O_{t-s}\,\mathcal V(\rho'_s)]ds=\int_0^t\mathrm{Tr}[\mathcal V^\dagger(O_{t-s})\rho'_s]ds$. Since $\rho'_s$ is a state, $\lvert\mathrm{Tr}[A\rho'_s]\rvert\le\lVert A\rVert$. Here $\mathcal V^\dagger(A)=i[\delta H,A]$; substitute $u=t-s$. $\square$

**Corollary A.1 (BEQS is recovered).** Because $\lVert[\delta H,O_u]\rVert\le2\lVert\delta H\rVert\,\mathrm{osc}(O_u)\le2\lVert\delta H\rVert\,\mathrm{osc}(O)$, Theorem A implies $\lvert\Delta(t)\rvert\le2t\lVert\delta H\rVert\,\mathrm{osc}(O)\le 2t\lVert k\rVert_1\lVert O\rVert$ for $\delta H=\sum_m k_m h_m$ with $\lVert h_m\rVert=1$. This is BEQS Eq. (14)/(A26). Theorem A is never worse, and it is strictly better whenever the Heisenberg-evolved observable forgets.

**Corollary A.2 (saturation).** If $\mathrm{osc}(O_u)\le K e^{-u/\tau}\mathrm{osc}(O)$, for example when $\kappa_{\mathcal L}(u)\le Ke^{-u/\tau}$, then
$$\lvert\Delta(t)\rvert\le 2K\lVert\delta H\rVert\,\mathrm{osc}(O)\,\tau(1-e^{-t/\tau})\le 2K\lVert\delta H\rVert\,\mathrm{osc}(O)\min(t,\tau).$$
For a primitive $\mathcal L$ in finite dimension, such $(K,\tau)$ always exist, with $1/\tau$ up to the spectral gap.

**Remark A.3 (conserved quantities).** Suppose $\mathcal L$ has a nontrivial Heisenberg fixed-point algebra and no purely imaginary eigenvalues. Then $O_u\to O_\infty$, the projection of $O$ onto the conserved observables. The integrand of Theorem A tends to $\lVert[\delta H,O_\infty]\rVert$, so the long-time accumulation rate of error is set by the commutator of the imperfection with the *conserved part* of the observable. Section 4 makes this precise for U(1).

### Theorem A2 (expected error of a zero-mean ensemble)

Let $\delta H(g)=\sum_m g_m h_m$ with $\mathbb Eg=0$, covariance $\Sigma$, and finite third moments. Then
$$\big\lvert\mathbb E\,\Delta(t)\big\rvert\le\sum_{m,n}\lvert\Sigma_{mn}\rvert\,I_{mn}(t)\;+\;\mathbb E\,R_3(t;g),$$
where
$$I_{mn}(t)=\int_{a,b\ge0,\;a+b\le t}\big\lVert[h_m,e^{b\mathcal L^\dagger}([h_n,O_a])]\big\rVert\,da\,db,$$
$$R_3(t;g)=\int_{a,b,c\ge0,\;a+b+c\le t}\big\lVert[\delta H,e^{c\mathcal L^\dagger}[\delta H,e^{b\mathcal L^\dagger}[\delta H,O_a]]]\big\rVert .$$
Under $\kappa_{\mathcal L}(u)\le Ke^{-u/\tau}$:
$$\big\lvert\mathbb E\,\Delta(t)\big\rvert\le 4\,\mathrm{osc}(O)\Big(\sum_{m,n}\lvert\Sigma_{mn}\rvert\lVert h_m\rVert\lVert h_n\rVert\Big)K^2\tau^2f_2(t/\tau)+8\,\mathrm{osc}(O)\,\mathbb E\lVert\delta H(g)\rVert^3K^3\tau^3f_3(t/\tau),$$
with
- $f_2(x)=1-(1+x)e^{-x}\le\min(1,x^2/2)$
- $f_3(x)=1-(1+x+x^2/2)e^{-x}\le\min(1,x^3/6)$

*Proof.* In (D2), the first term is linear in $g$ with $g$-independent propagators, so its mean vanishes exactly. In the second term, expand $e^{u\mathcal L'}=e^{u\mathcal L}+\int_0^u e^{(u-w)\mathcal L}\mathcal V e^{w\mathcal L'}dw$ once more.

- The $e^{u\mathcal L}$ part is bilinear in $g$. In the Heisenberg form it equals $-\sum_{mn}g_mg_n\int\!\!\int\mathrm{Tr}\big[[h_m,e^{(s-u)\mathcal L^\dagger}([h_n,O_{t-s}])]\rho_u\big]$. Take the expectation and bound $\lvert\mathrm{Tr}[A\rho_u]\rvert\le\lVert A\rVert$, with $a=t-s$ and $b=s-u$.
- The remainder is the triple integral, bounded the same way using only that $e^{w\mathcal L'}\rho_0$ is a state.
- For the envelope form, note that $i[h,A]$ is Hermitian for Hermitian $A$. Use $\lVert[h,A]\rVert\le2\lVert h\rVert\mathrm{osc}(A)$, $\mathrm{osc}(e^{b\mathcal L^\dagger}B)\le\kappa(b)\,\mathrm{osc}(B)$ and $\mathrm{osc}(B)\le\lVert B\rVert$. Then integrate $K^2e^{-(a+b)/\tau}$ over the simplex. $\square$

**Remark.** For diagonal $\Sigma=\sigma^2\mathbb 1$, unit-norm $h_m$ and $K=1$, the leading term is $4\sigma^2M\,\mathrm{osc}(O)\tau^2f_2(t/\tau)$. As $t\to0$ this is $2\sigma^2Mt^2\,\mathrm{osc}(O)$, the leading order of BEQS Eq. (12) $(e^{2t^2\sigma^2M}-1)\lVert O\rVert$. As $t\to\infty$ it saturates at $4\sigma^2M\tau^2\mathrm{osc}(O)$ instead of growing as $e^{2t^2\sigma^2M}$. The commutators in $I_{mn}$ vanish unless $h_m,h_n$ overlap the support of the evolving observable, so the exact form is automatically light-cone restricted.

### Theorem A3 (concentration)

Let $g=\Sigma^{1/2}z$ with $z\sim\mathcal N(0,\mathbb 1_M)$, $\sigma_{\max}^2=\lVert\Sigma\rVert$, $\delta H(g)=\sum_mg_mh_m$, $\mathcal L'(g)=\mathcal L-i[\delta H(g),\cdot\,]$ and $f(g)=\mathrm{Tr}[Oe^{t\mathcal L'(g)}\rho_0]$. Let $\kappa_{\mathcal L}(u)=\sup_{\rho\ne\sigma}\lVert e^{u\mathcal L}(\rho-\sigma)\rVert_1/\lVert\rho-\sigma\rVert_1\le1$ be the trace-norm contraction coefficient, i.e. the norm of $e^{u\mathcal L}$ on the real space of traceless Hermitian operators with the trace norm. Let $S\subset\mathbb R^M$ be convex with $\int_0^t\kappa_{\mathcal L'(g)}(u)\,du\le T_{\rm eff}(t)$ for every $g\in S$, and $\Pr[g\notin S]\le\delta'$. Then for every $\delta\in(0,1)$, with probability at least $1-\delta-\delta'$,
$$\lvert f(g)-\mathbb Ef\rvert\le\sigma_{\max}\,\mathrm{osc}(O)\Big(\sum_m\lVert h_m\rVert^2\Big)^{1/2}T_{\rm eff}(t)\sqrt{2\ln(2/\delta)}\;+\;\mathrm{osc}(O)\,\delta'.$$

**Lemma A3.1 (uniform forgetting survives a small Hamiltonian perturbation).** Let $\lVert\cdot\rVert_\star$ be a norm on the traceless Hermitian operators with $\lVert[\delta H,X]\rVert_\star\le2\lVert\delta H\rVert\,\lVert X\rVert_\star$ (true for the trace and the Hilbert–Schmidt norm). If $\lVert e^{u\mathcal L}X\rVert_\star\le Ke^{-u/\tau}\lVert X\rVert_\star$ for all traceless Hermitian $X$ and $\lVert\delta H\rVert\le1/(4K\tau)$, then $\lVert e^{u\mathcal L'}X\rVert_\star\le Ke^{-u/(2\tau)}\lVert X\rVert_\star$.

*Proof.* $\mathcal L$ and $\mathcal V=-i[\delta H,\cdot\,]$ both preserve the traceless Hermitian operators. By (D1), $\phi(u)=e^{u/\tau}\lVert e^{u\mathcal L'}\rVert_\star$ (operator norm on that space) obeys $\phi(u)\le K+2K\lVert\delta H\rVert\int_0^u\phi(s)\,ds$. Grönwall's inequality gives $\phi(u)\le Ke^{2K\lVert\delta H\rVert u}\le Ke^{u/(2\tau)}$. $\square$

- In the trace norm this is $\kappa_{\mathcal L'(g)}(u)\le Ke^{-u/(2\tau)}$ on the convex ball $S=\{g:\lVert\delta H(g)\rVert\le1/(4K\tau)\}$, so $T_{\rm eff}(t)=\int_0^t\min(1,Ke^{-u/(2\tau)})\,du\le\min(t,\,2K\tau)$.
- In the Hilbert–Schmidt norm, $\kappa\le\sqrt D\,\lVert e^{u\mathcal L'}\rVert_{2\to2}$ gives the same with $K\to\sqrt D\,K_2$ in $T_{\rm eff}$ but the larger radius $1/(4K_2\tau)$; $K_2$ and $\tau$ are computable from the spectrum of $\mathcal L$.
- A valid $\delta'$: by the matrix Gaussian series bound (Tropp, Found. Comput. Math. 12, 389 (2012)), $\Pr[\lVert\delta H(g)\rVert>r]\le2D\exp(-r^2/2v^2)$ with $v^2=\lVert\sum_kA_k^2\rVert$ and $A_k=\sum_m(\Sigma^{1/2})_{mk}h_m$.

*Proof of Theorem A3.*
1. *Gradient on $S$.* $\partial_mf=\int_0^t\mathrm{Tr}[e^{(t-s)\mathcal L'^\dagger}(O)\,\mathcal V_m(e^{s\mathcal L'}\rho_0)]\,ds$ with $\mathcal V_m=-i[h_m,\cdot\,]$. For a state $\rho$ and Hermitian $A$, $\lvert\mathrm{Tr}[A\,\mathcal V_m(\rho)]\rvert=\lvert\mathrm{Tr}[i[h_m,A]\rho]\rvert\le\lVert[h_m,A]\rVert\le\lVert h_m\rVert\,\mathrm{osc}(A)$; the last step subtracts the midpoint of the spectrum of $A$, which leaves the commutator unchanged. By duality, $\mathrm{osc}(e^{u\mathcal L'^\dagger}O)=\sup_{\rho,\sigma}\mathrm{Tr}[O\,e^{u\mathcal L'}(\rho-\sigma)]\le\kappa_{\mathcal L'}(u)\,\mathrm{osc}(O)$. Hence $\lvert\partial_mf\rvert\le\lVert h_m\rVert\,\mathrm{osc}(O)\,T_{\rm eff}(t)$ on $S$, and $\lVert\nabla_zf\rVert\le\sigma_{\max}\lVert\nabla_gf\rVert\le L_S:=\sigma_{\max}\,\mathrm{osc}(O)(\sum_m\lVert h_m\rVert^2)^{1/2}T_{\rm eff}(t)$.
2. *Lipschitz on $S$.* $S_z=\{z:\Sigma^{1/2}z\in S\}$ is convex, so the mean-value theorem along segments inside $S_z$ shows that $f$ is $L_S$-Lipschitz on $S_z$.
3. *Extension.* McShane's extension $\tilde f(z)=\inf_{w\in S_z}[f(w)+L_S\lVert z-w\rVert]$ is $L_S$-Lipschitz on $\mathbb R^M$ and equals $f$ on $S_z$. Clamping it to $[\lambda_{\min}(O),\lambda_{\max}(O)]$ keeps both properties; call the result $\hat f$. Since $f$ takes values in the same interval, $\lvert f-\hat f\rvert\le\mathrm{osc}(O)$ everywhere and $f=\hat f$ on $S_z$, so $\lvert\mathbb Ef-\mathbb E\hat f\rvert\le\mathrm{osc}(O)\,\delta'$.
4. *Gaussian concentration.* By the Tsirelson–Ibragimov–Sudakov inequality, $\Pr[\lvert\hat f-\mathbb E\hat f\rvert>L_S\sqrt{2\ln(2/\delta)}]\le\delta$. On the intersection of this event with $\{g\in S\}$, $f=\hat f$, and step 3 gives the claim. $\square$

**Remarks.**
- The earlier sketch had a factor 2 from $\lVert[h,A]\rVert\le2\lVert h\rVert\,\mathrm{osc}(A)$; subtracting the spectral midpoint removes it.
- Without dissipation, $\kappa\le1$, $S=\mathbb R^M$, $\delta'=0$ and $T_{\rm eff}=t$: for unit-norm $h_m$ the bound is $\sigma_{\max}\,\mathrm{osc}(O)\sqrt M\,t\sqrt{2\ln(2/\delta)}$, the form of BEQS Eq. (11).
- With a primitive $\mathcal L$, Lemma A3.1 makes $T_{\rm eff}\le2K\tau$ time-uniform. With a conserved charge, $\kappa_{\mathcal L}(u)$ does not decay and $T_{\rm eff}=t$; for U(1) dynamics the gains come from Corollary B1′ and Theorem F instead.
- *Check (E21).* N = 4, nearest-neighbour XY chain with local dephasing 0.5 and emission 0.2 (primitive; gap 0.2 on the traceless subspace), the 21 one- and two-local Paulis as $h_m$, $O=Z_0$, Néel initial state, 4000 samples. In the Hilbert–Schmidt norm $K_2=4.01$, $\tau=5.1$, radius $1/(4K_2\tau)=0.0122$; $\sigma$ is set so that the matrix Gaussian bound gives $\delta'=0.01$ (no sample left the ball).
  - Lemma A3.1 holds with a largest ratio of 0.34 over the samples inside $S$.
  - No sample exceeded the concentration bound at $t=1,5,20,60$ and $\delta=0.1,0.05,0.01$ (preregistered: at most $\delta+\delta'$ plus three binomial errors).
  - The bound saturates at $T_{\rm eff}=38.1$ (the factor $\sqrt D\,K_2=16$ delays the decay), whereas the bound without dissipation grows as $t$: 0.59 against 0.89 at $t=60$, $\delta=0.1$. It is loose by orders of magnitude here, mostly through $\sqrt D$ and the additive $\mathrm{osc}(O)\delta'$.

## 4. U(1)-symmetric generators and collective noise

**Setup.**
- $S^z=\tfrac12\sum_i\sigma_i^z$. The charge sectors are $\mathcal B_q=\{X:[S^z,X]=qX\}$, $q\in\{-n,\dots,n\}$. These are mutually Hilbert–Schmidt orthogonal, and $\mathrm{Tr}[AB]=0$ for $A\in\mathcal B_a$, $B\in\mathcal B_b$ unless $a+b=0$.
- $P_q=\frac1{2\pi}\int_0^{2\pi}e^{-iq\phi}\mathrm{Ad}_\phi\,d\phi$, with $\mathrm{Ad}_\phi(X)=e^{i\phi S^z}Xe^{-i\phi S^z}$. Hence $\lVert P_qX\rVert_1\le\lVert X\rVert_1$ and $\lVert P_qA\rVert\le\lVert A\rVert$.
- $\mathcal L$ is *covariant* if it commutes with every $\mathrm{Ad}_\phi$, equivalently $\mathcal L(\mathcal B_q)\subseteq\mathcal B_q$.
  - XY couplings, $\sigma^z$ fields, any $\sigma^z$-dephasing matrix $\Gamma$, and $\sigma^-$ emission are all covariant.
  - The BEQS learned model therefore is covariant up to its small XX/YY asymmetry, which we treat as part of $\delta H$.
- Collective dephasing is $\mathcal D_c(X)=\gamma_c(ZXZ-\tfrac12\{Z^2,X\})$ with $Z=\sum_i\sigma^z_i=2S^z$. This is the $\Gamma=\gamma_c\mathbf 1\mathbf 1^T$ member of the dephasing family $\mathcal D_\Gamma(X)=\sum_{ij}\Gamma_{ij}(\sigma^z_iX\sigma^z_j-\tfrac12\{\sigma^z_j\sigma^z_i,X\})$.

### Lemma B.0 (exact collective damping, $K=1$)

Let $\mathcal L=\mathcal L_0+\mathcal D_c$ with $\mathcal L_0$ a covariant GKSL generator. Then $\mathcal D_c|_{\mathcal B_q}=-2\gamma_cq^2$ and $e^{t\mathcal L}|_{\mathcal B_q}=e^{-2\gamma_cq^2t}e^{t\mathcal L_0}|_{\mathcal B_q}$. Consequently, for $X,A\in\mathcal B_q$,
$$\lVert e^{t\mathcal L}X\rVert_1\le e^{-2\gamma_cq^2t}\lVert X\rVert_1,\qquad \lVert e^{t\mathcal L^\dagger}A\rVert\le e^{-2\gamma_cq^2t}\lVert A\rVert .$$

*Proof.* For Hermitian jumps, $\mathcal D_c(X)=-\tfrac{\gamma_c}2[Z,[Z,X]]=-2\gamma_c\,\mathrm{ad}_{S^z}^2X$. This is a polynomial in $\mathrm{ad}_{S^z}$, which commutes with every covariant $\mathcal L_0$ (covariance means commuting with $\mathrm{Ad}_\phi=e^{i\phi\,\mathrm{ad}_{S^z}}$). Hence the exponentials factorize. $e^{t\mathcal L_0}$ is CPTP, so it is trace-norm contractive on all operators. The dual statement follows from Russo–Dye, using $\mathcal D_c^\dagger=\mathcal D_c$. $\square$

### Lemma B.1 (protective collective rate)

For a real symmetric $\Gamma\succeq0$, $\Gamma-\gamma\mathbf 1\mathbf 1^T\succeq0$ holds iff $\gamma\le\gamma_c^*$, where
- $\gamma_c^*:=1/(\mathbf 1^T\Gamma^+\mathbf 1)$ if $\mathbf 1\in\mathrm{range}\,\Gamma$
- $\gamma_c^*:=0$ otherwise.

Then $\mathcal D_\Gamma=\mathcal D_c(\gamma_c^*)+\mathcal D_{\Gamma-\gamma_c^*\mathbf 1\mathbf 1^T}$, where the second dissipator is valid and covariant. So Lemma B.0 and Theorems B2–B4 hold with $\gamma_c=\gamma_c^*$ for any learned dephasing matrix.

*Proof.* This is the standard criterion for $A-cvv^T\succeq0$ with $A\succeq0$: it requires $v\in\mathrm{range}A$ and $c\,v^TA^+v\le1$. Linearity of $\mathcal D_\Gamma$ in $\Gamma$ gives the split. $\square$

*Robustness (learned from data, E6a).* $\gamma_c^*$ is set by the *smallest* eigenvalues of $\Gamma$ that overlap with $\mathbf 1$. Statistical noise makes some of those eigenvalues zero or negative, and then $\gamma_c^*$ collapses to 0.
- **Observed.** On the re-learned 10-ion $\Gamma$, the negative eigenvalues carry 18% of the Frobenius weight, and $\gamma_c^*=0$ after PSD projection.
- **Robust alternative.** Use the structured estimate $\Gamma\approx\gamma_c\mathbf 1\mathbf 1^T+\mathrm{diag}(g)$, with $\gamma_c$ the mean off-diagonal entry. It is valid for Theorems B2–B4 whenever $g\ge0$ within errors.
- **Data value.** $30.5\pm5.2$ rad/s. The off-diagonal residuals are consistent with noise (rms $z=0.59$), and the most negative $g_i$ is $-2.6\sigma$.
- **Open item.** A certified version would need a PSD-constrained fit with confidence regions.

*Example.* If the learned $\Gamma$ is dominated by one eigenvector $a\approx\mathbf 1/\sqrt n$ with rate $\vartheta_0$, then $\gamma_c^*\approx\vartheta_0/n$. For the BEQS 10-ion data ($\vartheta_0=460$ rad/s) this gives $\gamma_c^*\approx46$ rad/s, assuming their rate convention matches ours. That is $\approx0.08\,J_0$ with $J_0=576$ rad/s.

### Theorem B1 (first-order selection rule)

Let $\mathcal L$ be covariant (no collective noise is needed), $\rho_0\in\mathcal B_0$, $O\in\mathcal B_0$, and let $\delta H=\sum_{q\ne0}\delta H_q$ have no charge-0 component. Then the linear response vanishes identically:
$$\int_0^t\mathrm{Tr}[O\,e^{(t-s)\mathcal L}\mathcal V e^{s\mathcal L}\rho_0]\,ds=0\quad\forall t,$$
so $\Delta(t)=O(\lVert\delta H\rVert^2)$.

*Proof.* $e^{s\mathcal L}\rho_0\in\mathcal B_0$. Next, $\mathcal V$ maps it into $\bigoplus_{q\ne0}\mathcal B_q$, and $e^{(t-s)\mathcal L}$ preserves sectors. Finally $\mathrm{Tr}[O\,\cdot\,]$ annihilates every $\mathcal B_{q\neq0}$. $\square$

**Corollary B1′ (effective parameter count).** For an ensemble $\delta H(g)=\sum_mg_mh_m$ in a charge-definite basis (each $h_m$ has charges $\pm q_m$ only), $\partial_m\langle O_t\rangle|_{g=0}=0$ for every $m$ with $q_m\ne0$. First-order fluctuations, which dominate BEQS-type concentration bounds, therefore involve only the $M_{\rm sym}$ symmetric parameters. For 2-local Pauli ansätze such as BEQS Eq. (22), $M_{\rm sym}\approx M/3$: 1 of 3 single-site directions and 3 of 9 two-site directions are symmetric.

### Theorem B2 (charged observables are bounded at all times)

Let $\mathcal L=\mathcal L_0+\mathcal D_c$ as in Lemma B.0 and $\rho_0\in\mathcal B_0$. Let $O=\sum_{q\neq0}O_q$ have no charge-0 component, so that $\mathrm{Tr}[O\rho_t]=0$. Then for **arbitrary** Hermitian $\delta H$:
$$\lvert\Delta(t)\rvert=\lvert\mathrm{Tr}[O\rho'_t]\rvert\le\lVert\delta H\rVert\sum_{q\neq0}\lVert O_q\rVert\,\frac{1-e^{-2\gamma_cq^2t}}{\gamma_cq^2}.$$

*Proof.* Apply Theorem A sector by sector. $O_{q,u}=e^{u\mathcal L^\dagger}O_q\in\mathcal B_q$ has $\lVert O_{q,u}\rVert\le e^{-2\gamma_cq^2u}\lVert O_q\rVert$ (Lemma B.0), and $\lVert[\delta H,O_{q,u}]\rVert\le2\lVert\delta H\rVert\lVert O_{q,u}\rVert$. $\square$

As $\gamma_c\to0$ the bound becomes $2t\lVert\delta H\rVert\sum\lVert O_q\rVert$.

### Theorem B3 (Zeno suppression of symmetry-breaking imperfections, all orders)

Let $\mathcal L=\mathcal L_0+\mathcal D_c$ as in Lemma B.0, $\rho_0\in\mathcal B_0$, $O\in\mathcal B_0$, and $\delta H=\sum_{q\neq0}\delta H_q$ purely charged. Then
$$\lvert\Delta(t)\rvert\le 4\lVert\delta H\rVert\,\mathrm{osc}(O)\sum_{q\neq0}\lVert\delta H_q\rVert\,\varphi(2\gamma_cq^2,t),\qquad\varphi(a,t)=\frac ta-\frac{1-e^{-at}}{a^2}.$$
Since $\varphi(a,t)\le t/a$, this gives $\lvert\Delta(t)\rvert\le2\lVert\delta H\rVert\,\mathrm{osc}(O)\,t\sum_{q\ne0}\lVert\delta H_q\rVert/(\gamma_cq^2)$. Since $\varphi\to t^2/2$ as $\gamma_c\to0$, it reduces to the closed-system second-order bound $2\lVert\delta H\rVert\mathrm{osc}(O)t^2\sum\lVert\delta H_q\rVert$.

*Proof.* In (D2) the first term vanishes by Theorem B1. The second term equals, in Heisenberg form,
$$\int_0^t\!ds\!\int_0^s\!du\;\mathrm{Tr}\Big[\mathcal V^\dagger\big(e^{(s-u)\mathcal L^\dagger}\mathcal V^\dagger(O_{t-s})\big)\rho'_u\Big].$$
Here $O_{t-s}\in\mathcal B_0$. Then $\mathcal V^\dagger(O_{t-s})=\sum_{q\ne0}A_q$ with $A_q=i[\delta H_q,O_{t-s}]\in\mathcal B_q$ and $\lVert A_q\rVert\le2\lVert\delta H_q\rVert\mathrm{osc}(O)$. By Lemma B.0, $\lVert e^{(s-u)\mathcal L^\dagger}A_q\rVert\le e^{-2\gamma_cq^2(s-u)}\lVert A_q\rVert$. The outer $\mathcal V^\dagger$ costs at most $2\lVert\delta H\rVert$, and $\lvert\mathrm{Tr}[\,\cdot\,\rho'_u]\rvert\le\lVert\cdot\rVert$. Finally $\int_0^tds\int_0^sdu\,e^{-a(s-u)}=\varphi(a,t)$. $\square$

*Interpretation.* The symmetry-breaking imperfection can only act by an excursion into a charged sector, and collective noise damps such excursions at rate $2\gamma_cq^2$. Its accumulated effect grows at rate $\sim\lVert\delta H\rVert^2/\gamma_c$ instead of $\lVert\delta H\rVert^2t$. This is a quantitative, all-orders version of the Zeno protection of decoherence-free subspaces (cf. quant-ph/0008050; Stannigel et al. 2014), stated for many-body observables.

### Theorem B4 (error anatomy)

Under the assumptions of B3, but with general $\delta H=\delta H_0+\delta H_{\rm ch}$ (charge-0 part plus charged part):
$$\lvert\Delta(t)\rvert\le\underbrace{\int_0^t\lVert[\delta H_0,O_u]\rVert\,du}_{\text{symmetric: not suppressed by }\gamma_c}+\underbrace{4\lVert\delta H_{\rm ch}\rVert\,\mathrm{osc}(O)\sum_{q\ne0}\lVert\delta H_q\rVert\,\varphi(2\gamma_cq^2,t)}_{\text{breaking: Zeno-suppressed}}.$$

*Proof.* Let $\mathcal L'_0=\mathcal L-i[\delta H_0,\cdot]$, which is covariant and still contains $\mathcal D_c$. Split $\Delta=(\langle O\rangle_{\mathcal L'_0}-\langle O\rangle_{\mathcal L})+(\langle O\rangle_{\mathcal L'}-\langle O\rangle_{\mathcal L'_0})$. Apply Theorem A to the first difference and Theorem B3 (with base $\mathcal L'_0$) to the second. $\square$

### Locality: system-size independent versions

Theorems B3 and B4 use the global norms $\lVert\delta H\rVert$ and $\lVert\delta H_q\rVert$. Both are extensive, so the bounds grow as $N^2$ even when the observable is local and the actual error is not (E5). This section removes that growth.

**Notation.**
- Let $\Lambda$ be the set of sites with a distance $d$, and write $S_r=\{x:d(x,S)\le r\}$.
- $\mathfrak A_S$ denotes the operators supported in $S$, i.e. acting as the identity outside $S$.
- For a generator $\mathcal L_0=\sum_Z\mathcal L_Z+\mathcal L_{\rm sp}$ and a region $T$, the restriction $\mathcal L_0^{(T)}$ consists of:
  - the local terms with $Z\subseteq T$;
  - the dephasing matrix restricted to $\Gamma|_{T\times T}$;
  - the on-site terms inside $T$.

  This restriction is again a valid GKSL generator. Its Heisenberg flow $e^{u\mathcal L_0^{(T)\dagger}}$ is a unital CP contraction and maps $\mathfrak A_T$ into itself.

#### Lemma B.2 (support preservation; collective noise does not widen light cones)

1. $P_q(\mathfrak A_S)\subseteq\mathfrak A_S$ for every $q$, because $\mathrm{Ad}_\phi=\bigotimes_x\mathrm{Ad}(e^{i\phi\sigma^z_x/2})$ is a product of on-site conjugations.
2. Take any real symmetric $\Gamma\succeq0$. Then $\mathcal D_\Gamma(X)=-\tfrac12\sum_{ij}\Gamma_{ij}[\sigma^z_i,[\sigma^z_j,X]]$ satisfies $\mathcal D_\Gamma(X)=\mathcal D_{\Gamma|_{S\times S}}(X)\in\mathfrak A_S$ for $X\in\mathfrak A_S$, since $[\sigma^z_j,X]=0$ for $j\notin S$.
   - In particular this holds for collective dephasing ($\Gamma=\gamma_c\mathbf 1\mathbf 1^T$), even though its jump operator $\sum_x\sigma^z_x$ is extensive.
   - It also holds for the long-range correlated $\Gamma$ that BEQS learn.
3. An on-site generator $\mathcal L_x$ satisfies $\mathcal L_x^\dagger(X)=0$ for $X\in\mathfrak A_S$ with $x\notin S$, because $\mathcal L_x^\dagger(\mathbb 1)=0$. It maps $\mathfrak A_S$ into itself when $x\in S$.

Consequently, every Lieb–Robinson (LR) bound obtained by the standard Duhamel iteration holds for $\mathcal L_0=\sum_Z\mathcal L_Z+\mathcal L_{\rm sp}$ with the *same constants* as for its finite-range part $\sum_Z\mathcal L_Z$. This applies to the bounds of Poulin, PRL 104, 190401 (2010); Nachtergaele, Vershynina and Zagrebnov (2011); and Barthel and Kliesch, PRL 108, 230504 (2012). Here $\mathcal L_{\rm sp}$ is any $\mathcal D_\Gamma$ plus on-site terms.

*Proof.*
- Items 1–3 are immediate.
- For the consequence, the iteration compares $e^{u\mathcal L_0^\dagger}$ with $e^{u\mathcal L_0^{(T)\dagger}}$ through
$$e^{u\mathcal L_0^\dagger}Y-e^{u\mathcal L_0^{(T)\dagger}}Y=\int_0^ue^{(u-s)\mathcal L_0^\dagger}\big(\mathcal L_0^\dagger-\mathcal L_0^{(T)\dagger}\big)e^{s\mathcal L_0^{(T)\dagger}}Y\,ds .$$
- By items 2–3, $(\mathcal L_0^\dagger-\mathcal L_0^{(T)\dagger})X=\sum_{Z\cap T\ne\emptyset,\,Z\not\subseteq T}\mathcal L_Z^\dagger X$ for $X\in\mathfrak A_T$. Only boundary-crossing local terms appear, at every level of the iteration.

$\square$

The physical content is that the Zeno-protecting noise ($\mathcal D_c$) and the learned correlated dephasing damp operators without moving them. The light cone is set by the Hamiltonian alone.

#### Theorem B3′ (commutator form, reference dynamics only)

Under the assumptions of Theorem B3, define for $p>0$
$$\delta H_{\pm p}:=\delta H_p+\delta H_{-p}=\delta H_{\pm p}^\dagger .$$
Then
$$\lvert\Delta(t)\rvert\le\sum_{p>0}\int_{a,b\ge0,\;a+b\le t}e^{-2\gamma_cp^2b}\,\big\lVert\big[\delta H,\;e^{b\mathcal L_0^\dagger}\big([\delta H_{\pm p},O_a]\big)\big]\big\rVert\,da\,db,\qquad O_a=e^{a\mathcal L_0^\dagger}(O).$$

*Proof.* Follow the proof of B3 up to
$$\Delta(t)=\int_0^tds\int_0^sdu\,\mathrm{Tr}\big[\mathcal V^\dagger\big(e^{(s-u)\mathcal L^\dagger}\mathcal V^\dagger(O_{t-s})\big)\rho'_u\big].$$
- $O\in\mathcal B_0$ and $\mathcal D_c|_{\mathcal B_0}=0$. Hence $O_{t-s}=e^{(t-s)\mathcal L_0^\dagger}O$.
- Lemma B.0 gives $e^{b\mathcal L^\dagger}\mathcal V^\dagger(O_a)=\sum_{q\ne0}e^{-2\gamma_cq^2b}e^{b\mathcal L_0^\dagger}(i[\delta H_q,O_a])$. The damping is the same for $\pm q$, so the terms with $q=\pm p$ combine into $\delta H_{\pm p}$.
- Change variables to $a=t-s$, $b=s-u$; the Jacobian is 1.
- Finally use $\lvert\mathrm{Tr}[X\rho'_u]\rvert\le\lVert X\rVert$.

$\square$

*Remarks.*
- B3 follows from B3′ by three elementary estimates: $\lVert[\delta H,Y]\rVert\le2\lVert\delta H\rVert\lVert Y\rVert$, contractivity of $e^{b\mathcal L_0^\dagger}$, and $\lVert[\delta H_{\pm p},O_a]\rVert\le2\lVert\delta H_{\pm p}\rVert\,\mathrm{osc}(O)$. Every loss relative to B3′ comes from these three steps.
- The integrand of B3′ involves only the reference dynamics and commutators with $\delta H$. Terms of $\delta H$ outside the light cone of $\mathrm{supp}\,O$ do not contribute, so B3′ is local.
- For quadratic fermions, every operator in the integrand is quadratic. The integrand is then an exact single-particle computation (`qerr.quadratic`), which E7 evaluates at $N=200$.

#### Corollary B3-loc (Zeno bound independent of system size)

**Assumption (LR).** There are $c,v,\xi>0$ such that for all $S\subseteq\Lambda$, $r,u\ge0$ and $Y\in\mathfrak A_S$,
$$\big\lVert e^{u\mathcal L_0^\dagger}Y-e^{u\mathcal L_0^{(S_r)\dagger}}Y\big\rVert\le c\,\lvert S\rvert\,\lVert Y\rVert\,e^{(vu-r)/\xi}.$$
For finite-range $\sum_Z\mathcal L_Z$ on a lattice this holds by the references above. By Lemma B.2 it does not matter what dephasing matrix or on-site terms are added.

**Hypotheses on the perturbation and observable.**
- $\delta H=\sum_{x\in\Lambda}h_x$, where $h_x$ is Hermitian and supported in the ball $\{x\}_\ell$.
- $\varepsilon:=\max_x\lVert h_x\rVert$ and $\varepsilon_p:=\max_x\lVert(P_p+P_{-p})h_x\rVert$. The charge parts of $h_x$ live in $\{x\}_\ell$ by Lemma B.2.
- $O\in\mathcal B_0$ is supported in $A$.

**Light-cone quantities.**
- Light-cone radius: $r_a:=\max\{0,\lceil va+\xi\ln(c\lvert A\rvert)\rceil\}$.
- Shell weights: $\omega_0:=1$ and $\omega_k:=2e^{-(k-1)/\xi}$ for $k\ge1$.
- Reach of a region:
$$\mathcal N_S(b):=\sum_{y\in\Lambda}m_S(y,b),\qquad m_S(y,b)=\begin{cases}1,& d(y,S)\le\ell,\\ \min\{1,\;c\lvert S\rvert e^{(vb-d(y,S)+\ell+1)/\xi}\},&\text{otherwise.}\end{cases}$$
  This is the number of perturbation terms an operator in $\mathfrak A_S$ can reach within time $b$.

**Statement.** Then, for every system size,
$$\lvert\Delta(t)\rvert\le4\,\mathrm{osc}(O)\,\varepsilon\sum_{p>0}\varepsilon_p\int_{a+b\le t}e^{-2\gamma_cp^2b}\,\Xi_A(a,b)\,da\,db,\qquad \Xi_A(a,b):=\sum_{k\ge0}\omega_k\,\lvert A_{r_a+k+\ell}\rvert\;\mathcal N_{A_{r_a+k+2\ell}}(b).$$
$\Xi_A$ does not depend on $N$.

**One dimension.** For an interval $A$ in $D=1$, $\lvert A_r\rvert\le\lvert A\rvert+2r$ and
$$\mathcal N_S(b)\le\lvert S\rvert+2\ell+2+2\max\{0,vb+\xi\ln(c\lvert S\rvert)\}+2/(e^{1/\xi}-1).$$
Hence
$$\Xi_A(a,b)\le C\,(\lvert A\rvert+2v(a+b)+\kappa)^2,$$
where $C$ and $\kappa$ depend only on $c,\xi,\ell$, up to logarithms of $\lvert A\rvert+v(a+b)$. This gives
$$\lvert\Delta(t)\rvert\le4C\,\mathrm{osc}(O)\,\varepsilon\sum_{p>0}\varepsilon_p\,(\lvert A\rvert+2vt+\kappa)^2\,\varphi(2\gamma_cp^2,t).$$
Compare B3, where the squared light-cone volume is replaced by $\lVert\delta H\rVert\lVert\delta H_{\pm p}\rVert/(\varepsilon\varepsilon_p)\sim N^2$.

*Proof.* Insert the following estimate of the integrand of B3′.

1. **Shift.** Replace $O$ by $O-c\mathbb 1$ with $\lVert O-c\mathbb 1\rVert=\mathrm{osc}(O)$. Commutators are unchanged, and $\mathcal L_0^\dagger(\mathbb 1)=0$, so we may assume $\lVert O\rVert=\mathrm{osc}(O)$.
2. **Shells.** Let $T_k=A_{r_a+k}$, and set
   - $\Omega_0=e^{a\mathcal L_0^{(T_0)\dagger}}O$,
   - $\Omega_k=e^{a\mathcal L_0^{(T_k)\dagger}}O-e^{a\mathcal L_0^{(T_{k-1})\dagger}}O$ for $k\ge1$.

   Then:
   - $\Omega_k\in\mathfrak A_{T_k}$.
   - The sum telescopes to $O_a$, because $T_k=\Lambda$ for large $k$.
   - $\lVert\Omega_0\rVert\le\lVert O\rVert$.
   - Applying (LR) to both terms gives $\lVert\Omega_k\rVert\le2c\lvert A\rvert\lVert O\rVert e^{(va-r_a-k+1)/\xi}\le\omega_k\lVert O\rVert$.
3. **Inner commutator.** By Lemma B.2, $\delta H_{\pm p}=\sum_xh_{x,\pm p}$ with $h_{x,\pm p}\in\mathfrak A_{\{x\}_\ell}$.
   - $[h_{x,\pm p},\Omega_k]=0$ unless $x\in A_{r_a+k+\ell}$.
   - Otherwise it lies in $\mathfrak A_{S_k}$ with $S_k:=A_{r_a+k+2\ell}$, and its norm is at most $2\varepsilon_p\lVert\Omega_k\rVert$.
4. **Outer commutator.** For $Y\in\mathfrak A_{S_k}$ and $d(y,S_k)>\ell$, set $r=d(y,S_k)-\ell-1$.
   - $h_y$ commutes with $e^{b\mathcal L_0^{(S_{k,r})\dagger}}Y\in\mathfrak A_{S_{k,r}}$.
   - By (LR), $\lVert[h_y,e^{b\mathcal L_0^\dagger}Y]\rVert\le2\varepsilon\lVert Y\rVert\,c\lvert S_k\rvert e^{(vb-r)/\xi}$. The bound is also at most $2\varepsilon\lVert Y\rVert$.
   - Summing over $y$ gives $\lVert[\delta H,e^{b\mathcal L_0^\dagger}Y]\rVert\le2\varepsilon\lVert Y\rVert\,\mathcal N_{S_k}(b)$.
5. **Combine.** Steps 2–4 give
$$\big\lVert[\delta H,e^{b\mathcal L_0^\dagger}([\delta H_{\pm p},O_a])]\big\rVert\le\sum_k\lvert A_{r_a+k+\ell}\rvert\cdot2\varepsilon\cdot2\varepsilon_p\,\omega_k\lVert O\rVert\,\mathcal N_{S_k}(b)=4\varepsilon\varepsilon_p\,\mathrm{osc}(O)\,\Xi_A(a,b).$$
6. **One-dimensional form.** It follows from $\sum_k\omega_kk^m<\infty$ and the counting above.

$\square$

*Scope.*
- **What is new.** System-size independent error bounds for local observables under local generator errors, growing polynomially in $t$, are known for closed and open analogue simulators (Trivedi, Franco Rubio and Cirac, Nat. Commun. 2024; Kashyap et al. 2025). Applied to $\delta H_{\rm ch}$ at first order, they give $\lvert\Delta\rvert\lesssim\varepsilon\,(vt)^D\,t$. B3-loc adds the two symmetry-resolved ingredients inside the light cone:
  - the first-order term vanishes (B1);
  - the second-order term carries the Zeno factor $\varphi(2\gamma_cp^2,t)\le t/(2\gamma_cp^2)$.
- **Not time-uniform.** The bound is free of $N$ but grows polynomially in $t$. A time-uniform local bound needs forgetting inside the light cone. Rapid-mixing stability results (Cubitt, Lucia, Michalakis and Pérez-García, CMP 337, 1275 (2015); Kashyap et al., PRX 15, 021017 (2025)) provide such forgetting for generators with a unique, rapidly mixing fixed point. U(1)-conserving dynamics has a degenerate fixed-point space, so those results do not apply, while B3-loc does.
- **Power-law interactions.** For $J_{ij}\propto\lvert i-j\rvert^{-\alpha}$, LR bounds are algebraic and require $\alpha>D$. The BEQS ion chains ($\alpha\approx1$, $D=1$) lie at the boundary of that regime, so for them B3′ (evaluated numerically) is the appropriate statement.

### Remark B5 (non-abelian generalization)

Suppose $\mathcal L_0$ is SU(2)-covariant and the collective noise is isotropic, $\mathcal D_{\rm iso}=-\tfrac\gamma2\sum_{a}[Z_a,[Z_a,\cdot]]$ with $Z_a=2S^a$. Then $\mathcal D_{\rm iso}=-2\gamma\,C_{\rm adj}$, where $C_{\rm adj}=\sum_a\mathrm{ad}_{S^a}^2$ acts on spin-$k$ irreducible tensor operators as $k(k+1)$. Theorems B1–B4 hold with the following replacements:
- charge sectors → irreducible tensor sectors
- the charge selection rule → Clebsch–Gordan selection rules
- $q^2$ → $k(k+1)$

### Remark B6 (where this matters, honestly)

- **Inside the BEQS 10-ion window.** With the rate implied by BEQS's dominant dephasing eigenvalue, $\gamma_c\approx46$ rad/s $\approx0.08J_0$, the Zeno factor $\varphi(a,t)/(t^2/2)$ at 10 ms is ≈0.75 for $|q|=1$ and ≈0.40 for $|q|=2$. With our re-learned $\gamma_c=30.5$ rad/s (E6a) it is ≈0.82 and ≈0.51. Either way the suppression is modest.
- **Theorem B1 is independent of $\gamma_c$.** It already removes about two thirds of the parameters from the first-order error budget.
- **Where the Zeno effect is strong.** Platforms or targets with stronger global noise, longer times, or deliberately engineered collective dephasing.

## 5. Positivity of uncertainty ensembles

### Proposition C

Consider a single two-level system with emission jump $\sqrt\gamma\sigma^-$ and $\gamma\sim\mathcal N(\mu,s^2)$, and let $p(t)$ be the excited-state population. Then $\mathbb E\,p(t)=p(0)\,e^{-\mu t+s^2t^2/2}$, which exceeds $p(0)$ for $t>2\mu/s^2$ and diverges as $t\to\infty$. With $\gamma$ log-normal (any parameters), $p(t)\le p(0)$ for every sample.

*Proof.* $p(t)=p(0)e^{-\gamma t}$ for each sample. Then use the Gaussian moment-generating function. $\square$

*Consequence.* The Duhamel-type bounds (and BEQS's replacement $M\to M'$) use $\lVert e^{s\mathcal L'}\rho_0\rVert_1=1$, which fails for non-CP samples. Long-time certification therefore needs ensembles supported on valid generators, e.g. log-normal rates or Cholesky-parametrized $\Gamma=AA^T$. For BEQS's emission rate ($\mu\approx2$ rad/s), $t^*=2\mu/s^2$ is of order 0.1–1 s, far outside the 10 ms window. The issue is conceptual for long-time and steady-state certification, not a flaw of their analysis.

## 6. Numerical checks (experiments in `experiments/`)

| Statement | Experiment | What is checked |
|---|---|---|
| Thm A, Cor A.1/A.2 | E2 | observed $\lvert\Delta\rvert\le$ forgetting bound $\le$ BEQS bias bound; saturation of the forgetting bound |
| Thm B1 | E1, E3, E5 | ε-doubling ratio ≈ 4 for charge-0 $O$ with charged $\delta H$; charged/symmetric ensemble band ratio ∝ σ inside the perturbative window ($\sigma\sqrt M t\lesssim1$); the separation is lost outside it |
| Thm B2 | E1 | charged observables stay below the bound; their plateau decreases with γ_c (faster than the 1/γ_c of the bound) |
| Thm B3/B4 | E1, E3, E5 | charged errors shrink with γ_c, symmetric errors are exactly γ_c-independent; free fermions at N = 20–200 are N-independent |
| Thm A2 vs BEQS | E3b | A2 leading term (nested Heisenberg commutators) ≥ observed ensemble bias, and below BEQS Eq. (12) by 1.8–24× |
| Prop C | E4 | Gaussian vs log-normal rate ensembles |
| Lemma B.1, Thm B1′/B4/B3′ on real data | E6a, E6b | Re-learned BEQS 10-ion generator: U(1)-breaking couplings consistent with zero; structured collective rate; first-order variance share of the symmetric combinations; Néel-quench error budget (B3′ vs Thm A vs B3 vs BEQS bias bound). The 51-ion magnetization moments are a model-free U(1) diagnostic |
| Thm B3′, Cor B3-loc | E7 | free fermions, $N=20$–$200$: the B3′ certificate is $N$-independent and lies above the observed error; global B3 grows as $N^2$ |

## 7. Open items

1. ~~Local (Lieb–Robinson) refinement of B3~~. Done: Lemma B.2, Theorem B3′ and Corollary B3-loc, checked by E7. What remains open is a local bound that is also *time-uniform* for U(1)-conserving dynamics. It would need forgetting inside the light cone for the non-conserved part of $O$, while diffusive transport of the conserved charge keeps the commutator with $O_\infty$ alive (Remark A.3).
   - Phase 3 (Theorem E): no Heisenberg-only bound can be time-uniform for a density.
   - Phase 4 (Theorem F, Corollary F-qf, E13 in `docs/theory/phase4-theory.md`): pairing the observable with the state's forgetting gives a time-uniform bound. It is rigorous for quasi-free fermions with dephasing and conditional on diffusive hypotheses in general.
2. Steady-state version (t → ∞), connected to the BQP-hardness of fixed points (Kashyap et al. 2025).
3. Non-Markovian / quasi-static noise (BEQS footnote 73): the dephasing-by-a-random-static-field ensemble versus Lindblad dephasing.
   - Phase 4 (Theorem G, E12, `docs/theory/phase4-theory.md`): the anatomy holds for any classical collective noise. The protection changes from 1/Q² to 1/|Q| for quasi-static Gaussian noise.
   - BEQS data test: E11/E11b.
