# 연구 설계: Dissipation이 지우는 Hamiltonian 오차 — 대칭 분해 기반 오차 해부

작성일: 2026-11-04 · 목표 저널: Physical Review A (확장판은 PRX)

## 1. 의도와 이유

- **질문.** Hamiltonian이 불완전할 때(보정 오차 + 학습 불확실성) Lindbladian 진화의 예측은 얼마나 정확한가? 그리고 *어떤* Hamiltonian 오차를 dissipation이 지워주고, *어떤* 오차가 살아남는가?
- **위치.** Kraft et al., PRX 16, 031037 (2026) [BEQS, arXiv:2511.23392]의 후속이다. 단, BEQS의 통계 중심 접근(학습 → Gaussian ensemble → 범용 집중 부등식)과는 다르게 **구조 중심**으로 접근한다. 즉 Heisenberg 그림에서 관측량이 잊혀지는 과정과 대칭 선택 규칙으로 오차를 해부한다.
- **동기가 된 BEQS의 열린 문제.**
  - bound가 짧은 시간에서만 의미가 있음 (Eq. 11–12)
  - dissipation을 bound에 활용하지 않음
  - "대칭과 DFS를 검증에 활용하는 방법"을 미래 과제로 명시함
  - 51-ion 실험의 DFS 가정에 대한 오차가 정량화되지 않음

## 2. 목표 · 비목표 · 반목표

- **목표**
  1. 아래 §3의 정리를 엄밀하게 증명한다.
  2. 소규모 정확 시뮬레이션(N ≤ 8)과 free-fermion 대규모 시뮬레이션(N ≤ 200)으로 각 정리와 scaling을 검증한다.
  3. BEQS와 비슷한 파라미터에서 BEQS bound와 정량적으로 비교한다.
- **비목표**
  - 새로운 Hamiltonian learning 프로토콜
  - 실험 수행
  - QNO 구현 (후속 논문에서 다룸)
- **반목표** (기술적으로 맞더라도 이렇게 되면 실패)
  - BEQS를 "틀렸다"고 프레이밍하는 것
  - 이득이 작은 파라미터 구간을 숨기는 것
  - 수치로 검증하지 않은 scaling을 주장하는 것

## 3. 핵심 결과 (증명 대상)

표기:
- 대상 생성자: 유효한 Lindbladian $\mathcal L$
- 불완전한 생성자: $\mathcal L'=\mathcal L+\mathcal V$, 여기서 $\mathcal V=-i[\delta H,\cdot]$
- Heisenberg 진화한 관측량: $O_u=e^{u\mathcal L^\dagger}(O)$
- 오차: $\Delta(t)=\mathrm{Tr}[O e^{t\mathcal L'}\rho_0]-\mathrm{Tr}[Oe^{t\mathcal L}\rho_0]$

- **정리 A (Heisenberg forgetting bound, 모든 차수에서 엄밀).**
  - $|\Delta(t)|\le\int_0^t\|[\delta H,O_u]\|\,du$
  - $\|[\delta H,O_u]\|\le 2\|\delta H\|\|O\|$이므로 BEQS의 Duhamel bound $2t\|\delta H\|\|O\|$보다 항상 작거나 같다.
  - $O_u$가 항등원(또는 보존량)으로 수렴하면 bound가 포화한다.
- **정리 A2 (zero-mean ensemble의 expected error).** 두 단계 Duhamel 전개를 쓰면 1차 항의 기댓값이 정확히 0이 된다.
  - 주항: $\sum_m\sigma_m^2\int\!\!\int\|[h_m,e^{v\mathcal L^\dagger}[h_m,O_w]]\|$
  - 짧은 시간에서는 BEQS의 $2t^2\sigma^2M$을 재현한다.
  - 수축 상수 $(K,\tau)$가 있을 때: $4K^2\sigma^2M\tau^2 f(t/\tau)$, 여기서 $f(x)=1-(1+x)e^{-x}$
- **정리 A3 (concentration).** Gaussian Lipschitz 부등식을 쓴다. 정리 A2의 시간 적분이 τ에서 잘리기 때문에 시간에 무관한 폭이 나온다.
- **정리 B (U(1) charge 분해).** 가정:
  - $\mathcal L$이 $S^z$에 대해 공변이다.
  - $\mathcal L$이 collective dephasing (rate $\gamma_c$)을 포함한다. 이때 charge가 $q$인 연산자 섹터 $\mathcal B_q$에서 감쇠가 정확히 $e^{-2\gamma_c q^2 t}$이고 $K=1$이다.

  결과:
  - **B1 (선택 규칙).** charge-0 관측량과 charge-0 상태에서는 charged 섭동의 1차 기여가 정확히 0이다.
  - **B2 (charged 관측량).** $|\Delta(t)|\le\|O\|\|\delta H\|(1-e^{-2\gamma_cq^2t})/(\gamma_cq^2)$로, 시간에 무관하게 묶인다.
  - **B3 (Zeno 억제, 모든 차수).** $|\Delta(t)|\le 2\|O\|\|\delta H\|\,t\sum_{q\neq0}\|\delta H_q\|/(\gamma_cq^2)$. charged 오차의 누적률이 $\|\delta H\|$에서 $\|\delta H\|^2/\gamma_c$로 바뀐다.
- **보조정리 (보호 collective rate).**
  - 학습된 dephasing 행렬 $\Gamma$에서 $\Gamma-\gamma_c\mathbf 1\mathbf 1^T\succeq0$을 만족하는 최대값은 $\gamma_c^*=1/(\mathbf 1^T\Gamma^+\mathbf 1)$이다($\mathbf 1\in\mathrm{range}\,\Gamma$일 때).
  - 정리 B에는 이 $\gamma_c^*$를 쓴다.
- **명제 C (Gaussian rate ensemble의 장시간 병리).**
  - rate가 $\gamma\sim\mathcal N(\mu,s^2)$인 amplitude damping에서 $\mathbb E[P_{\rm exc}(t)]=e^{-\mu t+s^2t^2/2}$이다.
  - 이 값은 $t>2\mu/s^2$에서 1을 넘는다. 따라서 장시간 이론에는 CP를 보존하는 ensemble(log-normal 등)이 필요하다.
  - BEQS가 다룬 시간 창(수 ms)에서는 이 병리가 영향을 주지 않는다. 이 점을 정직하게 명시한다.
- **일반화 (논의 절).** Hamiltonian이 SU(2) 대칭이고 collective noise가 등방적이면, 감쇠율이 Casimir $k(k+1)$로 매겨진다.

## 4. 수치 실험

| ID | 내용 | 검증 대상 |
|---|---|---|
| E1 | long-range XY, N=6, Néel 상태. 섭동 charge(0, ±1, ±2) × 관측량 charge × $\gamma_c$ 스윕 | B1–B3, 정리 A |
| E2 | primitive dissipation (local dephasing + emission)에서 정리 A의 bound와 실제 오차를 비교하고, BEQS Duhamel bound와 대조 | 정리 A의 포화 |
| E3 | BEQS와 비슷한 ensemble(σ≈0.03 J0, 모든 2-local Pauli 항)에서 expected error와 95% 구간. 대칭 성분과 깨는 성분의 기여를 분해하고, BEQS bound(Eq. 11)와 대조 | 오차 예산, 실효 M 감소 |
| E3b | 정리 A2의 주항(중첩 Heisenberg 교환자 적분)을 관측된 ensemble 편향 및 BEQS Eq. (12)와 비교 | 정리 A2 |
| E4 | Gaussian rate와 log-normal rate의 장시간 비교 | 명제 C |
| E5 | free fermion(BdG) + local/collective dephasing, N=20–200. pairing 섭동(charge ±2)과 hopping/on-site 섭동(charge 0) | N에 무관함, Zeno scaling |
| E6 | BEQS Zenodo 데이터에 학습된 파라미터가 있는지 확인. 있으면 $\gamma_c^*$와 charge 분해를 적용 | 실데이터 연결 — **보류**: 원시 shot 데이터만 있음 (`docs/literature/novelty-check.md` §5) |

## 5. 결정 사항 (모두 Claude의 판단, 팀 확인 전)

- 도구: Python 3.14, NumPy/SciPy/matplotlib, pytest. venv 위치는 `C:\Users\mwill\.venvs\pra-qno`이다. OneDrive 동기화를 피하려고 이렇게 정했다.
- 단위: $J_0=1$. 결합은 $J_{ij}=J_0/|i-j|^{\alpha}$이고 $\alpha=1$ (BEQS의 51-ion 근사치 0.97에 가까운 값).
- 기저 규약: $|0\rangle$ = spin up ($\sigma^z=+1$), site 0이 kron의 가장 왼쪽 인자. emission은 $\sigma^-=|1\rangle\langle0|$.
- 코드 구조: `src/qerr/` (라이브러리), `tests/`, `experiments/` (그림 생성), `results/` (그림·데이터), `docs/theory/` (영문 정리·증명), `docs/results/` (한글 요약).
- git 저장소가 아니므로 커밋하지 않는다(사용자 요청 시에만).

## 6. 구현자에게 남긴 것

- 그림 스타일, 시간 격자, ensemble 샘플 수 등 결론에 영향을 주지 않는 세부 사항은 구현자가 정한다.
