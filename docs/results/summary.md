# 결과 요약 — Dissipation이 지우는 Hamiltonian 오차 (2026-11-04)

**한 줄 결론.** 제안한 정리들(A, A2, B1–B3, 명제 C)이 모든 수치 검증을 통과했다. 핵심 메시지는 "대칭을 지키는 오차는 collective noise와 무관하게 남고, 대칭을 깨는 오차는 1차에서 사라지며 collective noise가 추가로 억제한다"이고, N=6 정확 계산과 N=200 free-fermion 계산에서 모두 확인됐다. 다만 BEQS 실험 영역(약한 collective noise, 짧은 시간)에서는 Zeno 억제 효과가 크지 않다. 그 영역에서 실질적인 이득은 선택 규칙(B1)에서 나온다.

**2단계 한 줄 결론 (2026-11-04).**
- Zeno bound를 계 크기와 무관한 형태로 만들었다(정리 B3′, 따름정리 B3-loc, E7: 전역 bound보다 약 1000배 촘촘함).
- BEQS 공개 데이터에서 측정 기저를 복원해 10-ion 생성자를 다시 학습했다. U(1)을 깨는 결합은 모두 통계적으로 0과 구별되지 않고, 대칭을 강제하면 표본 밖 예측이 좋아진다.
- 51-ion 데이터의 자화 분포는 자발 붕괴 외에 추가로 전하를 바꾸는 과정이 있음을 보여 준다.
- PRA 원고 초안은 로컬 `paper/`에만 있다. 저장소가 public이라 git에는 올리지 않는다.

**3단계 한 줄 결론 (2026-11-04, 브랜치 `phase3-symmetry-classes`).**
- Collective dephasing이 막아 주는지는 불완전함이 coherent인지 dissipative인지가 아니라 **초연산자 charge**가 정한다(정리 B6). 공변인 flip은 약한 대칭은 지키지만 강한 대칭을 깨고, 1차로 작용하며 γ_c와 무관하다.
- 보존 전하가 있으면 Heisenberg forgetting bound의 증가가 관측량의 다중극 차수로 정해진다. 밀도는 √t, 사중극부터는 시간 균일하다(정리 E). 실제 오차는 bound와 달리 유계일 수 있다.
- 실데이터에서는 구조화 모델 조건부로 γ_c ≥ 20.7 rad/s(bias 보정 95% 하한)를 인증했다. Dissipator 수준의 약한 대칭 시험은 model mismatch 때문에 판정할 수 없었다(합 flip rate가 음수로 나옴).

## 1. 산출물

| 파일 | 내용 |
|---|---|
| `docs/superpowers/specs/2026-11-04-dissipative-error-anatomy-design.md` | 연구 설계 (목표, 정리, 실험 계획) |
| `docs/literature/novelty-check.md` | 선행연구 대비 새로움과 데이터 판정 |
| `docs/theory/error-anatomy-theory.md` | 정리와 증명 (영문, 논문용) |
| `src/qerr/` | 수치 라이브러리 (Lindblad, charge, bound, free fermion, 2단계: quadratic 연산자 엔진, Pauli 대수, Lindbladian 학습, 이온 데이터 I/O, 3단계: 초연산자 charge, 단일 이온 noise 파라미터, 구조화/PSD 제약 fit) |
| `tests/` | pytest (3단계 후 43개, 실데이터가 있을 때만 도는 테스트 1개 포함) |
| `experiments/e1`–`e10`, `e3b`, `e6c` | 그림 생성 스크립트 (E6, E6c, E9 = BEQS 실데이터, E7 = 국소 Zeno bound, E8 = 세 분류, E10 = 다중극 forgetting) |
| `results/*.png`, `*.pdf`, `*.json` | 그림(PNG은 팀용, PDF는 원고용)과 판정 결과 |
| `docs/theory/phase3-theory.md` | 3단계 이론 노트 (정리 B6, 따름정리 B6′, 정리 E). 2026-11-05에 로컬 `paper/notes/`에서 연구 노트로 옮김 |
| `paper/main.tex`, `paper/refs.bib` | PRA 원고 초안 (revtex4-2). **로컬 전용, git 제외.** 2026-11-05부터 작성 중단(3단계 상태로 고정) |
| `docs/data/beqs-data-notes.md` | BEQS 데이터 구조, 추론한 규약, 저자 확인 메일 초안 |
| `data/fetch_beqs.py` | Zenodo 데이터 다운로드와 md5 검증 (원시 데이터는 git에 올리지 않음) |

재현 방법은 아래와 같다(Python은 `C:\Users\mwill\.venvs\pra-qno\Scripts\python.exe`).

```
python data/fetch_beqs.py                # BEQS 원시 데이터 (6 MB, md5 검증)
python -m pytest -q                      # 프로젝트 루트에서 실행
cd experiments
python e7_local_zeno.py                  # 약 3분 (N ≤ 200)
python e6_ion_learning.py                # 약 20분 (10-qubit 학습, bootstrap, 검증)
python e6_symmetry.py                    # 약 20분 (e6_ion_learning 다음에 실행)
python e1_charge_anatomy.py              # 약 1분
python e2_forgetting.py
python e3_ensemble.py                    # 약 1분, 병렬 실행
python e3b_expected_error_bound.py       # 약 1분, e3 실행 후
python e4_cp_pathology.py
python e5_free_fermion.py                # 약 1.5분
```

## 2. 실험별 결과

**E1 — charge별 오차 해부** (`results/e1_charge_anatomy.png`, N=6 long-range XY, Néel 상태)
- **선택 규칙 (B1).** charge-0 관측량($Z_2Z_3$)에 charged 섭동을 줄 때 ε를 2배로 하면 오차가 3.97–4.00배가 된다. 즉 1차 기여가 정확히 0이다. 대칭 섭동에서는 이 비율이 2.035배(1차)다.
- **도달 불가능한 charge.** 대칭 섭동은 charged 기댓값을 만들 수 없고, |q|=2 섭동은 홀수 charge 관측량을 만들 수 없다. 두 경우 모두 오차가 정확히 0이다.
- **대칭 오차는 γ_c와 무관하다.** 네 가지 γ_c 곡선이 완전히 겹친다.
- **Zeno 억제.** |q|=2 오차(XX−YY 비대칭에 해당)의 최대값은 γ_c = 0 → 0.1 → 0.5 → 2에서 3.99e-3 → 2.61e-3 → 1.22e-3 → 3.60e-4로 줄어 약 11배 억제된다. |q|=1 오차의 억제는 더 약해서(1.23e-3 → 4.9e-4), 감쇠율이 q²에 비례한다는 예측과 맞는다.
- 모든 시간과 모든 조합에서 증명된 bound(정리 A, B2, B3)가 관측값보다 크다.

**E2 — Heisenberg forgetting (정리 A)** (`results/e2_forgetting.png`)
- 항상 관측 오차 ≤ 정리 A ≤ BEQS Duhamel bound 순서가 성립한다.
- primitive dissipation(emission 포함)에서는 정리 A bound가 포화한다. t=30 이후 증가분이 0.014%다. 반면 BEQS bound는 계속 선형으로 증가해서 t=40에서 41배 차이가 난다.
- U(1) 보존 dissipation에서는 $O_u$가 u≈30 이전에 0이 아닌 보존 성분 $O_\infty$로 수렴한다. 그래서 bound가 기울기 $\|[\delta H,O_\infty]\|$로 계속 선형 증가한다(Remark A.3의 예시). 이 기울기 일치는 정의상 거의 자명한 일관성 확인이지 독립 검증은 아니다. 실제 관측 오차도 이 경우 계속 증가한다(그림 (a)).

**E3 — BEQS와 비슷한 학습 불확실성 ensemble** (`results/e3_ensemble.png`; σ=0.03 J0, M=99, 그중 M_sym=33)
- **섭동 영역(σ√M t ≲ 1).** charged 성분과 대칭 성분의 95% 폭 비율이 σ에 비례한다. t≈1에서 σ=0.03일 때 0.21, σ=0.01일 때 0.075로, 예측(1/3)대로 줄어든다. 따라서 BEQS Eq. (11)을 M 대신 M_sym으로 계산해도 관측 band를 덮는다. √3배 더 촘촘한 bound다.
- **비섭동 영역(t=20, σ√M t ≈ 6).** charged 성분의 폭이 오히려 더 커진다($Z_2Z_3$에서 비율 2.26). 이 영역에서는 B1′의 "실효 M 감소"가 성립하지 않는다. **논문에서 반드시 밝혀야 하는 한계다.**
- 대칭 성분만 있는 ensemble은 γ_c와 1.6e-15 수준으로 무관하다(정확한 대칭).
- charged 성분의 편향은 γ_c를 키우면 줄어든다. t=20에서 $Z_2Z_3$는 0.095 → 0.059, staggered magnetization은 0.0022 → 0.0002(11배)다.

**E3b — 정리 A2와 BEQS Eq. (12) 비교** (`results/e3b_expected_error_bound.png`)
- A2 주항이 관측 편향을 위에서 덮는다(t ≤ 1에서 검사했고, 그림상 t ≤ 4 전체에서 성립).
- A2는 BEQS Eq. (12)보다 항상 작다.
  - γ_c=0.1: t=1에서 1.8배, t=4에서 13배 작다.
  - γ_c=1: t=1에서 2.5배, t=4에서 24배 작다.
- collective noise가 bound 안에서 charged 성분의 몫을 63%에서 33%로 줄인다.
- **한계.** σ=0.03, γ_c=0.1이면 A2 주항도 t≈4에서 1을 넘는다. BEQS는 이미 t≈2에서 1을 넘는다.

**E4 — Gaussian rate 불확실성의 장시간 병리 (명제 C)** (`results/e4_cp_pathology.png`)
- rate가 μ=2, s=2인 Gaussian이면 샘플의 15.9%가 비물리적이다(rate < 0). ensemble 평균 여기 확률은 t* = 2μ/s² = 1 s 이후 1을 넘고, Monte Carlo가 해석식과 1.7% 이내로 일치한다.
- 같은 평균과 분산을 갖는 log-normal rate는 모든 시간에서 물리적이다.
- BEQS의 10 ms 창에는 영향이 없다. 장시간·정상상태 이론에서만 필요한 수정이다.

**E5 — free fermion, N=20–200** (`results/e5_free_fermion.png`)
- 이 계산은 다체 Jordan–Wigner 정확해와 1e-8까지 일치함을 테스트로 확인했다.
- pairing(|q|=2) 섭동이 중앙 점유수에 주는 오차는 N≥50에서 N과 무관하다(편차 0.7%). 오차가 국소적이라는 뜻이다.
- 같은 오차가 γ_c = 0 → 1 → 4에서 0.019 → 0.0024 → 0.00054로 35배 억제된다. 마지막 구간은 1/γ_c에 비례한다.
- on-site(charge 0) 섭동의 오차는 γ_c와 정확히 무관하다.
- ε를 2배로 했을 때의 오차 비율은 t=2에서 3.76–3.88(2차)이다. t=10에서는 2.5–3.7로 떨어지는데, 비섭동 영역에 들어가기 때문이다. collective noise가 강할수록 섭동 영역에 더 오래 머문다.

## 3. 증명된 것과 수치로만 확인된 것

- **증명 완료** (`docs/theory`): 정리 A, Corollary A.1/A.2, 정리 A2(주항과 3차 나머지), Lemma B.0, B.1, 정리 B1–B4, 명제 C.
- **증명 스케치만 있음**: 정리 A3(concentration). 균일 forgetting 가정과 Lipschitz 확장 부분을 논문 단계에서 엄밀히 채워야 한다.
- **수치로만 확인 → 2단계에서 증명됨**: 국소 관측량 오차가 N과 무관하다는 점(E5)은 이제 정리 B3′과 따름정리 B3-loc으로 증명했고, E7로 확인했다.
- **증명 완료 (2단계)**: Lemma B.2(지지집합 보존), 정리 B3′, 따름정리 B3-loc.

## 4. 정직한 한계

1. **BEQS 실험 영역에서는 Zeno 억제가 약하다.**
   - γ_c ≈ 0.08 J0, 측정 창 ≲ 6 J0⁻¹이면 억제는 |q|=2에서 약 2.5배, |q|=1에서 약 1.3배다.
   - 이 영역에서 주된 이득은 선택 규칙(B1)과 교환자 구조(A2)에서 나온다.
   - Zeno 효과가 크게 작동하는 곳은 collective noise가 강하거나, 장시간이거나, collective dephasing을 의도적으로 설계한 경우다.
2. **bound들은 관측값보다 10–100배 느슨하다.** 정리 A와 A2가 모든 초기 상태에 대한 상한이기 때문이다. BEQS보다는 항상 촘촘하다.
3. **B2는 1/γ_c로 줄어들지만 실제 오차는 더 빨리 준다.** $\sigma^x$의 경우 대략 1/γ_c²다. charged 응답이 결맞음 동역학을 거쳐야 생기기 때문으로 보인다.
4. ~~실데이터(E6)는 보류했다.~~ 2단계에서 해결했다(§5.2). 다만 다음 한계가 남는다.
   - 측정 기저는 추론이다. 저자 확인 전이다.
   - 학습 창(1.5 ms) 밖의 예측 오차는 rms 0.10으로 통계 오차(0.015)보다 크다.
   - 엄밀한 $\gamma_c^*$는 잡음에 취약해서 0이 된다. 구조화 추정치는 30.5 ± 5.2 rad/s다.
5. **국소 bound는 N과 무관하지만 시간에는 무관하지 않다.** U(1) 보존 동역학에서 시간에도 무관한 국소 bound는 열린 문제다.

## 5. 2단계 결과 (2026-11-04, 브랜치 `phase2-local-data-draft`)

### 5.1 B3의 국소화 (이론: Lemma B.2, 정리 B3′, 따름정리 B3-loc / 수치: E7)

- **Lemma B.2.** charge 사영, 상관이 있는 모든 z-dephasing(집단 포함), on-site 항은 연산자의 지지집합을 넓히지 않는다. 그래서 Lieb–Robinson 속도는 Hamiltonian만으로 정해진다. 집단 dephasing은 연산자를 "움직이지 않고 감쇠만" 시킨다.
- **정리 B3′.** 교환자 형태의 증명서로, 기준 동역학만 쓴다: $\sum_p\int\!\!\int_{a+b\le t}e^{-2\gamma_cp^2b}\lVert[\delta H,e^{b\mathcal L_0^\dagger}([\delta H_{\pm p},O_a])]\rVert$.
- **따름정리 B3-loc.** LR 가정 아래에서 N과 무관하다. 1차원에서는 $\propto(|A|+2vt)^2\varphi(2\gamma_cp^2,t)$.
- **E7** (free fermion N=20–200, Majorana 엔진으로 정확히 계산):
  - 증명서는 N=100과 200에서 3×10⁻⁹까지 같다(N 무관).
  - 모든 시간에서 관측 오차 위에 있다(N=100, t=10에서 33–53배).
  - 전역 B3은 N=50→200에서 13.8배 커진다. 증명서는 전역 bound보다 700–1300배 촘촘하다.
  - γ_c = 0, 1, 4에서 증명서가 0.64 → 0.080 → 0.029로 줄어 광원뿔 안의 Zeno 효과를 보여 준다.
  - 격자를 반으로 줄였을 때의 변화는 1.7%다.
- **선행연구와의 관계.** N 무관성 자체는 Trivedi 2024와 Kashyap 2025에 있다. 우리 몫은 광원뿔 안의 선택 규칙과 Zeno 인자다.

### 5.2 BEQS 10-ion 데이터 재학습 (E6a)

- **기저 복원.** 측정 기저는 파일에 없지만 t=0 통계로 복원했다.
  - X/Y/Z = 675/669/656, 최소 여유 3.1 SEM.
  - 두 상태의 판정이 일치한다.
  - 저자 확인은 아직이고, 메일 초안이 `docs/data/beqs-data-notes.md`에 있다.
- **학습.** integral method (t ≤ 1.5 ms, 파라미터 275개, 식 2514개) + defect correction 3회 (변화량 58 → 10 → 2.9 rad/s) + bootstrap 100회.
  - **결합.** 최근접 $J$ = 617 rad/s (BEQS 부록 B 추정 576). power-law 지수는 1.47 ± 0.28.
  - **dephasing.** Γ의 최대 고유값은 322 ± 54 rad/s (BEQS 460)이고, 고유벡터가 균일 벡터와 0.98 겹친다. 즉 집단 dephasing이 지배적이며, BEQS와 같은 결론이다.
  - **구조화 집단률.** $\gamma_c$ = 30.5 ± 5.2 rad/s ≈ 0.05 $J_{nn}$. 엄밀한 Lemma B.1 rate는 잡음 때문에 0으로 붕괴했다(18% 음의 고유값 가중치). 이론 노트에 강건성 주의를 추가했다.
  - **U(1)을 깨는 결합.** 3σ를 넘는 것이 하나도 없다: $(J^{xx}-J^{yy})/2$, $(J^{xy}+J^{yx})/2$ 각 45개, 가로장 20개 중 0개, 최대 |z| = 2.7. 학습된 생성자는 U(1) 대칭과 일치한다.
- **검증** (1.5–10 ms, 표본 밖):
  - 전체 모델 rms 0.104, U(1) 대칭화 모델 rms 0.094, 통계 오차 0.015.
  - **대칭을 강제하면 표본 밖 예측이 좋아진다.**
  - 남는 편차는 주로 2-site 가로 상관에서 나온다. 예를 들어 state 2의 ⟨X₅X₆⟩는 데이터에서 0.5로 유지되지만 모델에서는 0.2로 감쇠한다.

### 5.3 U(1) 진단과 오차 예산 (E6b)

**51-ion Néel 데이터 (모델 없이, z-기저 1000 shot × 11 시점).**
- 엇갈린 자화(charge 0)는 0.98에서 출발해 4 ms에 −0.35까지 내려갔다가 0으로 이완한다.
- ⟨M⟩은 시간당 −0.0218 ± 0.0036 /ms로 이동한다. D₅/₂ 자발 붕괴만으로 예측되는 값(25/τ, τ = 1.168 s)과 정확히 같다.
- Var M은 0.129 ± 0.011 /ms로 커진다. 붕괴만으로 기대되는 값(Poisson)의 약 6배다.
- 초기 자화 섹터에 남을 확률은 0.70에서 0.36으로 떨어진다.
- **해석.** 자발 붕괴 외에 양방향으로 전하를 바꾸는 과정이 있다. BEQS가 언급한 DFS 근사와의 차이와도 일치한다.
- **주의.** 이런 비가간섭 점프는 선택 규칙(가간섭 결합에만 적용)으로 보호되지 않는다. 진단 자체는 z-기저 데이터만 있으면 되므로 다른 U(1) 시뮬레이터에도 쓸 수 있다.

**학습된 10-ion 모델의 오차 예산** (Néel 퀜치, 엇갈린 자화).
- **설정.** 데이터가 배제하지 못하는 U(1)을 깨는 결합(합친 연산자 노름 1.26 rad/ms, ±1과 ±2가 반반)을 섭동으로 둔다.
- **실제 오차.** 10 ms 동안 0.02 미만이다(2 ms에서 0.016). 2차로만 들어오기 때문이다.
- **2 ms에서의 bound.**
  - BEQS bias bound 15와 전역 B3 15는 무의미하다.
  - **B3′ 증명서는 0.39, 정리 A는 0.58**로 유효하다.
- **10 ms에서.** 증명서 5.3, 정리 A 2.2로 둘 다 자명한 bound(2)를 넘는다. ‖δH_ch‖t ≈ 13이라 섭동 영역 밖이다.
- **1차 요동 예산 (정리 B1′).** 대칭 조합이 bootstrap 분산의 67%만 차지한다. 따라서 charge-0 예측의 1차 집중 bound가 √0.67 ≈ 0.82배로 줄어든다.

## 6. 다음 단계 (우선순위 순)

1단계에서 정한 우선순위 1–3 (B3 국소화, 10-ion 재학습, 원고 초안)은 2단계에서 끝냈다. 남은 일은 다음과 같다.

1. ~~저자 확인~~ **메일은 보내지 않기로 결정했다(사용자).** 원고에는 기저를 "추론"으로 쓰고, 원저자와 무관한 독립 분석임을 밝힌다.
   - 원고는 이전 논문 템플릿(REVTeX `reprint`, 로컬 전용)으로 옮겼다.
   - Table I(관측량 charge × 섭동 charge, slashbox)을 추가했다.
2. ~~인증된 $\gamma_c^*$~~ **3단계에서 함(§7.2).** 구조화 모델 조건부로 γ_c ≥ 20.7 rad/s를 인증했다. 엄격한 γ_c*는 noise 때문에 0으로 무너진다(오목 함수의 Jensen bias).
3. ~~51-ion 자화 분산 증가의 원인~~ **3단계에서 정리함(§7.2).** 강한 대칭은 깨져 있다(균형 섹터 변화율 2.10 ± 0.22 s⁻¹/ion). 이것이 class I/III인지 II인지는 z-기저 데이터만으로는 가릴 수 없고, γ_c scan으로 가린다(따름정리 B6′).
4. ~~시간에 무관한 국소 bound~~ **3단계에서 부분 해결(§7.4).** Heisenberg forgetting형 bound로는 불가능함을 보였다(밀도는 √t). Hydrodynamic 구조를 쓰는 시간 균일 bound는 열린 문제로 남아 있다(PRX 방향).
5. **원고 마무리 — 보류 (사용자 결정, 2026-11-05).** 당분간 원고는 쓰지 않고 실험과 연구만 진행한다. 결과는 이 문서와 `docs/theory/`의 연구 노트에 기록한다. 원고(`paper/`, 로컬)는 3단계 상태(15쪽)로 둔다.
6. **확장(PRX).** 정상상태 인증, 비마르코프(준정적) 노이즈, 의도적으로 설계한 집단 dephasing을 인증 자원으로 쓰는 방향, 실험 협업.
7. **후속 논문(QNO).** 감쇠가 있는 trunk를 갖는 QuanONet. 이 논문의 bound를 물리 기반 기준선으로 쓴다.

## 7. 3단계 결과 (2026-11-04, 브랜치 `phase3-symmetry-classes`)

이론 노트: `docs/theory/phase3-theory.md` (2026-11-05에 연구 노트로 옮김).

### 7.1 약한/강한 대칭과 소산 불완전함 (이론: 정리 B6, 따름정리 B6′ / 수치: E8)

- **초연산자 charge.** 초연산자 𝒮의 charge-Q 부분 𝒮_Q는 B_p를 B_{p+Q}로 보낸다. Hamiltonian 오차와 dissipator 오차를 같은 틀로 분해할 수 있다.
- **세 분류.**
  - I: 대칭을 깨는 Hamiltonian 오차. 2차이고 Zeno로 억제된다.
  - II: 공변 불완전함. 1차이고 γ_c와 무관하다. σ± flip은 강한 대칭만 깬다.
  - III: dissipator의 charged 부분. 2차이고 Zeno로 억제된다. 예: x dephasing = 공변 flip + V_{±2}.
- **E8 (N=6).**
  - Leakage의 κ 배증 비율: I은 4.00, II와 III는 1.91.
  - II의 leakage는 γ_c와 무관하다(상대 차이 10⁻¹⁵).
  - III와 II의 leakage는 정확히 같다(차이 3×10⁻¹⁶). 균일한 flip에서는 dP(M)/dt의 flip 부분이 배치에 M을 통해서만 의존하고, charged 부분은 대각 연산자를 없앤다. 그래서 P(M)이 닫힌 birth–death 방정식을 따르고, charged dissipation은 어떤 차수에서도 P(M)을 바꾸지 못한다. 새로 얻은 정확한 결과다.
  - Staggered magnetization에서 x dephasing의 charged 부분은 flip rate를 고정하고 세기만 바꾸면 배증 비율이 4.000이다. γ_c를 0에서 4로 올리면 2.0×10⁻⁷에서 7.7×10⁻¹¹로 줄어든다.
  - κ와 flip rate를 함께 바꾸면 γ_c ≥ 2에서 비율이 4에서 벗어난다(5.07). Zeno로 억제된 κ² 항에 Zeno로 억제되지 않는 flip × charged 교차 κ³ 항이 경쟁하기 때문이다(ledger ruling).
  - 정리 B6 bound는 모든 시간에서 성립한다.

### 7.2 실데이터: 약한/강한 대칭과 인증된 γ_c (E9)

- **확장 ansatz.** 이온당 noise 파라미터 8개를 넣어 모두 345개 파라미터로 학습했다(full rank).
  - 합 flip rate S_xx+S_yy가 음수로 나온다: 사이트 평균 −50 ± 13 rad/s, bias 보정 후 −31 rad/s. 물리적인 생성자에서는 불가능한 값이다.
  - Charged 조합이 이온마다 같은 부호로 치우쳐 있다(S_xx−S_yy ≈ −160 rad/s, S_xz ≈ +75 rad/s). 60개 중 10개가 3σ를 넘는다.
  - t=0 contrast로 보정해도 남는다(S_xx−S_yy = −139 rad/s).
  - 결론: model mismatch(상태 준비 오차, 비마르코프 noise 등)로 해석하고, 약한 대칭 위반으로 보고하지 않는다. 분해능 한계는 이온당 약 10² rad/s다.
- **Hamiltonian의 U(1) 깨는 결합.** 확장 fit에서도 0과 양립한다. K, L, Bx의 최대 |z|는 2.0–2.3이다. By는 세 이온에서 3.0–3.2σ인데, charge ±1 noise와 서로 상쇄하는 관계다.
- **구조화 γ_c.**
  - 29.9 ± 4.1 rad/s.
  - Bootstrap 평균이 2.1 rad/s 높다(errors-in-variables bias). 그래서 percentile 하한 26.0은 보수적이지 않다.
  - Basic bootstrap 하한 **20.7 rad/s를 인증값**으로 쓴다(확장 ansatz에서는 22.1).
  - 10 ms에서 Zeno 인수는 |q|=1에서 0.88 이하, |q|=2에서 0.62 이하다.
- **엄격한 γ_c\*.** Lemma B.1을 PSD 제약 full Γ에 적용하면 bootstrap 100개 중 100개에서 0이다. γ_c*가 오목 함수라 noise가 있으면 아래로 편향된다(Jensen).
- **Shadow 판정.** 10 qubit 순도 추정의 표준오차가 0.16(state 1), 0.31(state 2)이다. Charge 분해 신호는 0.16이라 판정할 수 없다. 10⁴개 정도의 설정이 필요하다.
- **51 ion.**
  - ⟨M⟩ drift는 D5/2 붕괴와 일치한다(class II).
  - Var M의 초과분은 균형 섹터 변화율 2.10 ± 0.22 s⁻¹/ion에 해당한다. 강한 대칭은 깨져 있지만, 약한 대칭 위반은 이 데이터로 보이지 않는다.

### 7.3 검증 편차 (E6c)

- **설정.** 학습 창(1.5, 3, 5 ms)과 dephasing 모델(full Γ, 구조화 Γ)을 조합해 6개 모델을 만들고, 모두 t > 5 ms에서 검증했다. 균일 z장 제약은 데이터가 기각하므로(z 최대 8.5) 돌리지 않았다.
- **전체 오차.** rms가 0.117(창 1.5 ms, full Γ)에서 0.067(창 5 ms, 구조화 Γ)로 줄었다. 긴 창과 구조화 Γ가 각각 도움이 된다.
- **Charge별 오차.**
  - 2-site charged 상관(q=±1 XZ형, q=±2): 통계오차의 1.5–2.6배.
  - 단일 사이트 가로 자화(q=±1): 5–8배.
  - 가장 큰 오차는 charge-0이다. σ^z는 0.160에서 0.074로 줄었고(SEM 0.008), flip-flop은 0.166에서 0.088로 줄었다(SEM 0.010).
- **⟨X5X6⟩ 문제의 정체.**
  - 이온 5–6의 charge-0 flip-flop 상관 C0가 데이터에서는 10 ms 동안 0.495 ± 0.009로 유지된다.
  - 모든 모델이 이 상관을 dephase한다. 학습된 charge-0 rate는 79–189 rad/s다.
  - 가장 좋은 모델도 10 ms에서 0.30을 예측한다.
- **창에 따른 표류.** 학습된 결합이 창에 따라 바뀐다(J_nn 617 → 573 → 504 rad/s). 시간에 무관한 Markovian 생성자 하나로는 모든 시간을 맞출 수 없다. E9의 model mismatch와 일치하고, 준정적이거나 시간 상관이 있는 noise일 가능성을 시사한다.
- **의미.** Mismatch가 charge-0 섹터에 있으므로 1차로 작용하고 collective dephasing의 보호도 받지 못한다(정리 B6). 인증에서 가장 중요한 부분이다.

### 7.4 보존 전하와 hydrodynamic forgetting (정리 E / 수치: E10)

- **정리 E.** 관측량의 밀도 성분이 다중극 차수 m을 가지면 integrand가 u^{−(m+1)/2}로 줄어든다. 그래서 정리 A의 bound는 m=0에서 √t, m=1에서 log t로 자라고, m≥2에서는 시간 균일하다. 보조정리 E.1(summation by parts, C_ℓ = 2ℓ(ℓ+1))은 엄밀하다. 확산 가정은 조건으로 명시했다.
- **E10 (free fermion, N=400, g=1, ε=0.01).**
  - 지수: monopole −0.506, dipole −1.022, quadrupole −1.527, bond current −1.029.
  - Monopole bound: u=10, 100, 1000에서 0.031, 0.067, 0.176.
  - 밀도파의 peak 오차는 λ=20–160에서 2.96–3.26×10⁻³이다. 확산계수 재규격화로 예측한 2.94×10⁻³과 맞는다. 실제 오차는 시간에 대해 균일하지만, bound는 peak 시각에서 0.027에서 0.107로 자란다.
- **의미.** 보존 밀도에 대해서는 Heisenberg forgetting형 시간 균일 bound가 존재할 수 없다. Hydrodynamic 구조를 쓰는 bound는 열린 문제다(PRX 방향). 공변 flip이 보존을 깨면 forgetting이 회복된다.
