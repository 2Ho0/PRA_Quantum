# 결과 요약 — Dissipation이 지우는 Hamiltonian 오차 (2026-11-04)

**한 줄 결론.** 제안한 정리들(A, A2, B1–B3, 명제 C)이 모든 수치 검증을 통과했다. 핵심 메시지는 "대칭을 지키는 오차는 collective noise와 무관하게 남고, 대칭을 깨는 오차는 1차에서 사라지며 collective noise가 추가로 억제한다"이고, N=6 정확 계산과 N=200 free-fermion 계산에서 모두 확인됐다. 다만 BEQS 실험 영역(약한 collective noise, 짧은 시간)에서는 Zeno 억제 효과가 크지 않다. 그 영역에서 실질적인 이득은 선택 규칙(B1)에서 나온다.

## 1. 산출물

| 파일 | 내용 |
|---|---|
| `docs/superpowers/specs/2026-11-04-dissipative-error-anatomy-design.md` | 연구 설계 (목표, 정리, 실험 계획) |
| `docs/literature/novelty-check.md` | 선행연구 대비 새로움과 데이터 판정 |
| `docs/theory/error-anatomy-theory.md` | 정리와 증명 (영문, 논문용) |
| `src/qerr/` | 수치 라이브러리 (Lindblad, charge, bound, free fermion) |
| `tests/` | 테스트 16개, 전부 통과 |
| `experiments/e1`–`e5`, `e3b` | 그림 생성 스크립트 |
| `results/*.png`, `results/*.json` | 그림과 판정 결과 |

재현 방법은 아래와 같다(Python은 `C:\Users\mwill\.venvs\pra-qno\Scripts\python.exe`).

```
python -m pytest -q                      # 프로젝트 루트에서 실행: 16 passed
cd experiments
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
- **수치로만 확인**: 국소 관측량 오차가 N과 무관하다는 점(E5). 정리 B3은 전역 $\|\delta H\|$를 쓰므로 N에 따라 커진다. Lieb–Robinson을 이용한 국소화 버전은 아직 없다.

## 4. 정직한 한계

1. **BEQS 실험 영역에서는 Zeno 억제가 약하다.**
   - γ_c ≈ 0.08 J0, 측정 창 ≲ 6 J0⁻¹이면 억제는 |q|=2에서 약 2.5배, |q|=1에서 약 1.3배다.
   - 이 영역에서 주된 이득은 선택 규칙(B1)과 교환자 구조(A2)에서 나온다.
   - Zeno 효과가 크게 작동하는 곳은 collective noise가 강하거나, 장시간이거나, collective dephasing을 의도적으로 설계한 경우다.
2. **bound들은 관측값보다 10–100배 느슨하다.** 정리 A와 A2가 모든 초기 상태에 대한 상한이기 때문이다. BEQS보다는 항상 촘촘하다.
3. **B2는 1/γ_c로 줄어들지만 실제 오차는 더 빨리 준다.** $\sigma^x$의 경우 대략 1/γ_c²다. charged 응답이 결맞음 동역학을 거쳐야 생기기 때문으로 보인다.
4. **실데이터(E6)는 보류했다.** Zenodo에는 원시 shot 데이터만 있다.

## 5. 다음 단계 (우선순위 순)

1. **B3의 국소화 버전.** Lieb–Robinson bound와 dissipation이 제한하는 light cone을 써서 N에 무관한 bound를 만든다. E5가 이미 수치 근거를 보여 준다. 이 결과가 PRA 논문의 완성도를 크게 올린다.
2. **10-ion 데이터 재학습.** integral method로 다시 학습한 뒤, 학습된 Γ에 $\gamma_c^*$와 charge 분해를 적용한다(실데이터 그림 1장 확보). random unitary의 기저 정보가 파일에 있는지 먼저 확인해야 한다. README의 연락처(M. K. Joshi)에 물어보는 것이 자연스럽다.
3. **논문 초안 구성 제안.** 이 구성의 장점은 대칭 분해라는 우리만의 관점이 그림 1과 정리 B로 바로 드러난다는 것이다.
   - 그림 1: 개념도 + E1 축약판
   - 그림 2: E2 (forgetting)
   - 그림 3: E3/E3b (BEQS 대비)
   - 그림 4: E5 (N 확장)
   - Appendix: 명제 C (E4)
4. **확장(PRX).** 정상상태 버전, 비마르코프 노이즈(BEQS 각주 73), 실험 협업.
5. **후속 논문(QNO).** 감쇠가 있는 trunk를 갖는 QuanONet. 이 논문의 bound를 물리 기반 기준선으로 쓴다.
