# 선행연구 · 새로움 점검 (2026-11-04)

검색은 표본 수준이다(WebSearch와 초록·본문 일부 열람). 투고 전에 Google Scholar와 PRX 페이지의 "Cited by"로 한 번 더 확인해야 한다.

## 1. BEQS를 인용한 후속 연구

- 검색 범위에서는 Kraft et al. (PRX 16, 031037 (2026), arXiv:2511.23392)을 인용한 후속 논문이 보이지 않았다.
- 따라서 "장시간·dissipation·대칭" 방향은 아직 비어 있는 것으로 보인다(2026-11-04 기준).

## 2. 가장 가까운 선행연구와 우리 결과의 차이

| 선행연구 | 내용 | 우리 정리와의 차이 |
|---|---|---|
| Cai–Tong–Preskill, TQC 2024 (2311.14818) | 닫힌계에서 무작위 섭동의 확률적 상쇄 | 열린계, Heisenberg forgetting(정리 A, A2), 대칭 선택 규칙(B) |
| Rao–Eisert–Guaita (2510.08467, v2 2026.5) | Hamiltonian 타깃에서 Gaussian/Brownian 노이즈 bound | 모든 bound가 t의 다항식으로 증가하고 Lindbladian 타깃이 없다. 우리는 dissipation이 지우는 성분을 분해한다 |
| Kashyap et al., PRX 15, 021017 (2025) | rapidly mixing Lindbladian의 worst-case 노이즈 안정성 | 확률적 ensemble, charge별 선택 규칙, Zeno 억제율 |
| Angrisani et al. (2609.40174) | dissipation이 Trotter 오차를 억제 | 알고리즘 오차만 다룬다. 모든 Pauli 방향이 감쇠해야 해서 보존량이 있는 계는 제외된다 |
| Daryanoosh, PRA 114, 022406 (2026) | non-normality가 오차를 증폭함(휴리스틱) | 우리 정리 B는 collective 섹터에서 K=1이 정확히 성립한다(non-normality 없음) |
| Stannigel et al., PRL 112, 120406 (2014) | engineered noise의 Zeno 효과로 gauge 제약 유지 | 우리는 장치에 원래 있는 collective noise를 다루며, 목적이 불확실성 전파와 certification이다 |
| quant-ph/0008050, PRA 62, 062305 | 2-qubit DFS, 불완전한 collective decoherence에서 생기는 leakage를 Zeno 효과가 억제 | 단일 인코딩이 아니라 다체 시뮬레이터 전체에 대한 charge별 엄밀 bound |
| Tran et al., PRX Quantum 2, 010323 (2021) (2006.16248) | 디지털 시뮬레이션에서 대칭 회전으로 오차를 상쇄 | 디지털 회로가 아니라 아날로그 열린계이고, 회전 대신 dissipation이 보호한다 |
| 1802.05293 (PRA 97, 052106) · 2603.10654 | Liouvillian의 대칭 섹터별 gap, correlated dephasing의 선택 규칙 | 스펙트럼 구조만 다룬다. 우리는 그 구조를 오차 bound로 연결한다 |
| 2601.05952 · 2602.21879 | 아날로그 열린계의 연속시간 error cancellation/mitigation | 오차 완화 프로토콜이고, 오차 예측(certification)이 아니다 |

### 2단계 추가 (2026-11-04): 국소화와 실데이터 관련 선행연구

| 선행연구 | 내용 | 우리 결과와의 차이 |
|---|---|---|
| Trivedi, Franco Rubio, Cirac, Nat. Commun. 15, 6507 (2024) (2212.04924) | 국소 오차가 있는 아날로그 시뮬레이터에서 국소 관측량 오차가 N과 무관하다. Gaussian fermion과 일부 스핀계 | 대칭을 쓰지 않는다. 우리 B3-loc은 같은 광원뿔 안에서 선택 규칙(1차 소거)과 Zeno 인자 φ(2γ_c p², t)를 더한다 |
| Kashyap et al., PRX 15, 021017 (2025) (2404.11081) | 열린계 아날로그 시뮬레이션에서 N과 무관한 정확도. rapid mixing이면 시간에도 무관하다 | 유일한 고정점(rapid mixing)이 필요하다. U(1) 보존 동역학은 고정점이 여러 개라 적용되지 않는다. 우리 결과는 바로 이 경우를 다룬다 |
| Cubitt, Lucia, Michalakis, Pérez-García, CMP 337, 1275 (2015) | 국소 dissipative 계의 안정성 (rapid mixing) | 위와 같은 차이 |
| Poulin PRL 2010 · Nachtergaele–Vershynina–Zagrebnov 2011 · Barthel–Kliesch PRL 2012 | Lindbladian의 Lieb–Robinson bound | 우리 Lemma B.2를 보탠다: 임의의 상관 dephasing Γ(집단 포함)와 on-site 항은 지지집합을 보존하므로 LR 상수를 바꾸지 않는다. 짧은 관찰이지만 문헌에서 명시적으로 쓰인 것은 찾지 못했다(표본 검색) |
| Angrisani, Puig, Teng, Holmes, arXiv:2609.40174 (2026) | "Dissipation accelerates quantum and classical simulation of open-system dynamics" | 알고리즘(시뮬레이션 가능성) 관점이다. 모든 Pauli 방향의 감쇠가 필요해서 보존량이 있는 계는 제외된다 |

**실데이터 재분석.** BEQS 데이터에서 측정 기저를 t=0 통계로 복원해 재학습한 연구는 검색 범위에서 보이지 않았다. 51-ion Néel 데이터의 자화 분포 모멘트를 대칭 진단에 쓴 사례도 보이지 않았다. 다만 무작위 측정 데이터의 대칭 분해 일반론은 있다(Elben et al. toolbox, Nat. Rev. Phys. 5, 9 (2023)). 원고에서는 "독립적 재분석"으로만 표현하고, 저자 확인 메일 초안을 `docs/data/beqs-data-notes.md`에 두었다.

## 3. 새로움 주장 (방어 가능한 형태)

1. **Heisenberg forgetting bound (정리 A).** 기술적으로는 단순하다. Duhamel 공식에 Heisenberg 그림을 쓴 것이다. 주장의 초점은 "BEQS의 Duhamel bound를 항상 개선하고, dissipation이 있으면 포화한다"는 것이다. 이것 자체를 핵심 새로움으로 내세우지는 않는다.
2. **U(1) charge 분해 정리 B1–B4.** 내용은 다음과 같다.
   - 선택 규칙: charge-0 관측량에서 charged 섭동의 1차 기여가 0이다.
   - charged 관측량: 시간에 무관한 bound가 나온다.
   - Zeno 억제: 모든 차수에서 성립하는 bound가 나온다.

   검색 범위에서 이 조합을 다룬 연구는 보이지 않았다. 핵심 새로움이다.
3. **학습된 ensemble에서의 실효 파라미터 수 감소.** 2-local 기준으로 Hamiltonian 파라미터의 약 2/3가 U(1)을 깬다. 이 성분들은 BEQS 유형의 1차 요동 예산에서 빠진다. 실용적 새로움이다.
4. **보호 collective rate $\gamma_c^*=1/(\mathbf 1^T\Gamma^+\mathbf 1)$.** 학습된 dephasing 행렬을 정리 B에 연결하는 조작적 정의다. 실데이터에서는 이 rate가 Γ의 작은 고유값에 민감하다. 그래서 E6에서는 구조화 추정(Γ ≈ γ_c11ᵀ + diag(g))을 함께 쓴다.
5. **N과 무관한 Zeno bound (B3′, B3-loc).**
   - 교환자 형태의 증명서는 기준 동역학만 쓰고, free fermion에서는 정확히 계산된다.
   - Lemma B.2(dephasing은 지지집합을 보존)로 광원뿔이 Hamiltonian만으로 정해진다.
   - 국소성 자체는 Trivedi 2024와 Kashyap 2025에 있다. 우리 몫은 대칭 분해된 부분(선택 규칙, Zeno 인자)이다.
6. **공개 데이터의 독립 재분석.**
   - 10-ion 측정 기저를 t=0 통계로 복원해 integral method로 재학습했다. 이산화 편향은 defect correction으로 제거했다.
   - 학습된 생성자를 charge별로 분해했다.
   - 51-ion 자화 분포로 U(1)을 진단했다.
7. **(3단계) 약한/강한 대칭에 따른 불완전함의 분류 (정리 B6, 따름정리 B6′).**
   - 초연산자 charge로 Hamiltonian 오차와 dissipator 오차를 하나의 틀로 분해한다. "Zeno 보호 여부는 coherent냐 dissipative냐가 아니라 초연산자 charge가 정한다."
   - 약한/강한 대칭이라는 개념 자체는 Buča–Prosen 2012와 Albert–Jiang 2014에 있다. 그 연구들은 이 구분을 정상상태 구조에 썼다. 우리는 오차 예산과 인증 bound에 쓴다. 이 점을 원고에 명시했다.
   - 새로 얻은 정확한 결과: 균일한 flip rate에서는 P(M)이 닫힌 birth–death 방정식을 따르므로, charged dissipation이 P(M)을 바꾸지 못한다. 이것을 바탕으로 γ_c scan 진단을 제안했다.
8. **(3단계) 보존 전하 아래 다중극 forgetting (정리 E).**
   - Heisenberg forgetting의 시간 거동이 관측량의 다중극 차수로 정해진다(√t, log t, 균일).
   - 그 결과 Heisenberg형 bound로는 시간 균일한 국소 bound를 얻을 수 없다. 실제 오차는 확산계수 재규격화 때문에 유계일 수 있다(E10).
   - Rapid-mixing 안정성(Cubitt 2015, Kashyap 2025)은 유일 고정점을 가정해서 이 경우를 다루지 않는다. Dephasing에 의한 확산 자체는 잘 알려져 있다(Haken–Strobl, Eisler 2011).
9. **(3단계) 인증된 collective rate.** 엄격한 γ_c*가 오목 함수라서 noise에 의해 아래로 편향됨(Jensen)을 보였다. 그래서 구조화 모델에 조건부로 bias를 보정한 하한을 쓴다.
10. **(4단계) 기억이 있는 집단 noise에서의 오차 해부 (정리 G).**
    - Lemma G.0: 고전 집단 noise ξ(t)S^z는 공변 생성자 아래에서 charge 섹터마다 χ_q(t)를 곱할 뿐이다. 어떤 확률과정에도 정확하다. Gaussian이면 시간 의존 rate를 갖는 정확한 time-local 방정식이 된다.
      - 이런 "정확한 고전 noise master equation" 자체는 Kiely, EPL 134, 10001 (2021)에 있다. BEQS의 ref. [72]다. 반드시 인용한다.
      - 비마르코프 고전 noise의 근사 master equation으로 arXiv:2207.03980이 있다. 우리 Lemma G.0은 공변 생성자와 교환하는 집단 noise라는 특수한 경우이고, 그래서 근사 없이 정확하다.
    - 새로운 부분: 선택 규칙과 공변 오차의 noise 무관성은 noise 스펙트럼과 무관하게 성립한다. Zeno 억제는 기억이 있어도 남지만, charge 스케일링이 1/Q²(white)에서 1/|Q|(준정적)로 바뀐다. 같은 T₂에서 |Q|=2 보호가 약 1.8배 약하다.
    - 표본 검색에서 이 스케일링을 다룬 연구는 보이지 않았다. 가까운 것은 집단 noise로 DFS를 보호하는 APS 초록(2025)과 DFS의 비마르코프 조건(quant-ph/0505051)이다.
11. **(4단계) 두 그림 forgetting에 의한 시간 균일 bound (정리 F, 따름정리 F-qf).**
    - 정리 E의 √t 장애물(Heisenberg형 bound)을 넘는다. 관측량의 기울기와 상태의 전류·기울기를 짝지어서, 둘 다 확산으로 u^{−1/2}씩 감쇠하면 합성곱이 π/D로 유계가 된다.
    - Quasi-free fermion에서는 가정 없이 엄밀하다(이차 모멘트로 계산 가능).
    - 표본 검색에서는 "유일 고정점이나 rapid mixing 없이, 확산하는 보존량으로 얻는 시간 균일 아날로그 bound"가 보이지 않았다. 가까운 것은 다음과 같다.
      - Trivedi–Franco Rubio–Cirac 2024와 Kashyap 2025: N 무관이지만 시간 균일성에는 mixing이 필요하다.
      - Heyl–Hauke–Zoller, Sci. Adv. 2019: 디지털 Trotter 오차, 창발 보존량.
      - Cai–Tong–Preskill: t^{d/2+1} 증가.
    - 수치해석의 포물형 오차 추정과 같은 구조라는 점을 원고에서 밝혀야 한다.
12. **(4단계) BEQS 데이터의 모델 선택과 학습 방법의 편향 (E11, E11b, E11c).**
    - 물리적 제약(집단 dephasing만, D₅/₂ emission, U(1) 대칭 결합)을 둔 최소 모델이 원래 E6a의 275-파라미터 fit보다 표본 밖 예측이 3배 좋다(rms 0.118 → 0.040). BEQS 저자들의 학습 결과를 반박하는 것이 아니다. 같은 데이터에서 대칭과 물리 제약이 예측력을 높인다는 독립 재분석이다.
    - 시뮬레이션 기반 보정으로 integral method의 errors-in-variables 편향을 정량화했다. 늦은 창의 J 붕괴와, 집단 rate 기울기의 귀무 분포가 −20만큼 치우친 것이 이 편향이다.
    - 순수 white 집단 noise를 4.9–8.1σ로 기각했다. BEQS가 각주 73과 결론에서 남긴 질문, 즉 느린 장 요동과 유색 noise 학습에 답한다. 표본 검색에서 같은 데이터로 이를 시험한 연구는 보이지 않았다.
    - 해결되지 않은 것: 단일 이온 noise block의 charged 섹터 mismatch(E11c)와, 편향을 넘는 누적 창 표류(−15%).

## 4. 리스크

- Zeno 억제 자체는 잘 알려진 물리다. 도입부에서 Stannigel 2014, PRA 62 062305, Tran 2021을 먼저 인용해서 새로움이 "certification용 엄밀 bound와 선택 규칙"에 있다는 점을 분명히 해야 한다.
- (3단계) 정리 E는 확산 가정(커널의 기울기 bound)을 조건으로 둔다. 수치로는 E10에서 확인했지만 일반적인 증명은 아니다. 원고에서도 조건부 정리로 썼다.
- (3단계) 확장 ansatz에서 dissipator 파라미터가 비물리적 값을 낸다. 이것을 대칭 위반으로 읽으면 안 된다. 원고에는 model mismatch로 썼다.
- (4단계) white noise 기각(E11b)의 근거는 두 가지다. 독립 Gaussian noise로 만든 합성 데이터에서 얻은 귀무 분포, 그리고 시간에 대해 선형인 rate 모델이다. 실제 noise는 같은 shot을 공유해 상관되어 있다. 혼합 참으로 보정을 넓히고, 상관 noise를 넣어 다시 검증하기 전에는 "준정적 성분 60–80%"를 정량 결론으로 쓰지 않는다.
- (4단계) 정리 F는 일반 계에서 확산 가정에 조건부다. 엄밀한 것은 quasi-free fermion(따름정리 F-qf)뿐이다. 상태 쪽 인자를 전역 최대값으로 짝짓기 때문에 domain wall에서는 14배 느슨하다.
- BEQS 실험 영역(10 ions, $\gamma_c\approx0.08J_0$, 측정 시간 ≲ 6 $J_0^{-1}$)에서는 Zeno 억제가 |q|=2 오차에서만 O(1) 수준이다. 선택 규칙(B1)은 $\gamma_c$와 무관하게 성립하므로 이 영역에서도 쓸모가 있다. 수치로 확인해야 한다(E3).

## 5. 데이터 판정 (E6)

- Zenodo 10.5281/zenodo.17750419: CC BY 4.0, 6.2 MB, 파일 4개(state1_2.zip, DataHLBicL51Ions.zip, NeelDataZ51.zip, ReadMe.txt).
- **원시 측정 데이터만 들어 있다.** shot 단위 bitstring이 10진수로 압축되어 있고, 학습된 $J_{ij}$, $\Gamma$, 공분산은 없다.
  - 10 ions: 상태 2개 × random unitary 200개 × 50 shot 배치 4개 × 21 시점
  - 51 ions: 11개 시점 파일
  - Néel 51 ions: z 기저, 20세트 × 50 shot × 11 시점 (0–10 ms)
- 결론: 실데이터를 쓰려면 integral method로 재학습하는 별도 작업이 필요하다. random unitary의 기저 정보가 파일 안에 있는지 확인해야 하고, README의 연락처는 M. K. Joshi다. 이번 세션에서는 BEQS 논문에 실린 파라미터 규모를 쓴 합성 모델로 대체한다(E3).

## 6. 4단계 선행연구 재점검 (2026-11-05)

사용자 요청으로 지금까지의 기여 전체를 다시 검색했다(WebSearch, 초록과 본문 일부). 결론부터 정리한다.
- 우리 작업 전체, 즉 "학습된 열린계 생성자에 대한 대칭 분해 인증 bound + 실데이터"를 그대로 한 연구는 찾지 못했다.
- 그러나 **핵심 물리 메커니즘은 이미 잘 알려져 있다.** 대칭을 깨는 오차를 dissipation, noise, 에너지 벌점, 무작위 대칭 변환이 Zeno 효과로 억제한다는 것은 gauge 보호 문헌의 중심 주제다. 이것이 가장 큰 겹침 위험이다.

| 우리 주장 | 가장 가까운 선행연구 | 겹침 | 우리 몫으로 남는 것 |
|---|---|---|---|
| 대칭을 깨는 오차의 Zeno 억제 (정리 B2–B4, B3′) | Stannigel et al. PRL 2014(gauge 생성자에 결합한 고전 noise, 보호 시간 κ/λ²)<br>Halimeh–Hauke 계열: PRX Quantum 2021, 리뷰 2022, Commun. Phys. 2025(에너지 벌점, Zeno 증명, λ²/V²)<br>Lamm–Lawrence–Yamauchi 2020과 후속 연구(무작위 gauge 변환 → 결맞은 표류가 무작위 걸음으로)<br>Tran et al. 2021(대칭 보호)<br>Schmale–Weimer 2024(dissipation으로 gauge 위반 교정)<br>Zanardi–Campos Venuti 2014/2015(dissipation 투영 동역학)<br>Burgarth–Facchi 외(강결합·Zeno 극한의 엄밀 bound) | **높음 (메커니즘)** | 학습된 생성자에 쓰는 **관측량 수준의 인증 bound**, 유한 γ_c에서 상수가 명시된 모든 차수 bound, 장치에 원래 있는 집단 dephasing을 자원으로 쓰는 것, 광원뿔 안의 국소 증명서(B3′) |
| 선택 규칙(정리 B1)과 실효 파라미터 수 감소 | 위 문헌, 대칭 검증 기반 오차 완화 | 중간 | 오차 예산(BEQS형 M → M_sym)과 인증 bound로의 연결 |
| 초연산자 charge로 본 불완전함의 분류, 약한/강한 대칭 (정리 B6) | Buča–Prosen 2012, Albert–Jiang 2014, Lieu et al. PRL 2020(약한/강한 대칭과 오류 정정) | 중간 | 이 개념을 **오차 예산과 인증**에 쓴 것: 공변 flip은 보호받지 못하는 1차 오차, P(M)이 닫힌 방정식을 따른다는 정확한 결과, γ_c scan 진단 |
| 기억이 있는 집단 noise와 보호 스케일링 1/Q² vs 1/\|Q\| (정리 G) | Kumar–Hauke–Halimeh 2022(에너지 벌점이 1/f^β noise를 1/V^β로 억제: noise가 오차원인 쌍대 상황)<br>Kiely EPL 2021, arXiv:2207.03980(고전 noise master equation)<br>Monz et al. 2011(상관 dephasing에서 GHZ의 N² 초결어긋남)<br>준정적 noise의 Gaussian 감쇠(T₂* ∝ 1/q)는 교과서 내용 | **중간** | noise를 **보호 자원**으로 볼 때의 정확한 해부(임의 고전 집단 noise), 보호 시간 τ_Q의 charge 스케일링, BEQS 데이터에서의 white 기각. 1/\|Q\| 자체는 알려진 결어긋남 법칙의 직접적 귀결이므로 "따름정리 수준"으로 주장한다 |
| 다중극 forgetting, 밀도의 √t (정리 E) | 보존 전하가 있을 때 연산자 확산의 확산 꼬리(Khemani–Vishwanath–Huse 2018, Rakovszky–Pollmann–von Keyserlingk 2018)<br>Han–Hartnoll 2018(dissipative 수송의 국소성 bound) | 중간 | 오차 bound의 시간 거동을 다중극 차수로 정리한 것, Heisenberg형 bound의 no-go |
| 두 그림 forgetting으로 얻는 시간 균일 bound (정리 F) | Trivedi et al. 2024, Kashyap et al. 2025(rapid mixing에서 시간 균일)<br>Trivedi–Cirac 2025(사상의 noise 강건성, O(γ^α τ^{d+1}))<br>Heyl–Hauke–Zoller 2019<br>operator backflow 2021(확산계수 오차의 대수적 감쇠)<br>포물형 PDE의 오차 추정 | **낮음** | rapid mixing 없이 보존 밀도에 대해 시간 균일한 국소 bound(quasi-free에서는 엄밀). 표본 검색에서 같은 결과는 보이지 않았다 |
| BEQS 데이터의 모델 선택, 보정, 기억 시험 (E11, E11b, E11c) | BEQS 자체: 부록 A3(재시뮬레이션에서 최근접 항이 조금 위로 편향), 학습 오차 plateau, 각주 73, 결론의 유색 noise 학습 제안<br>Olsacher et al. QST 2025, Guo et al. Sci. Adv. 2025(300 ion, 훈련/검증 오차, 물리 기반 모델 비교)<br>van den Berg et al. 2025(대규모 Lindblad 학습)<br>Gleinig–Eisert(모델 선택은 열린 문제) | 낮음 | 이 데이터셋의 독립 재분석, 대칭 제약 모델 선택, 창 의존성의 errors-in-variables 정량화, white 집단 noise 기각. 회귀 변수에 noise가 있을 때의 편향 자체는 통계학에서 알려진 현상이므로 그 문헌을 인용한다 |

**포지셔닝 권고 (원고를 재개할 때).**
1. 도입부에서 "dissipation과 noise가 대칭을 깨는 오차를 Zeno 효과로 억제한다"를 **알려진 사실로 먼저 인정**한다. Stannigel 2014, Halimeh–Hauke 계열, Lamm 2020, Zanardi–Campos Venuti, Burgarth–Facchi를 인용한다. 우리 기여는 그 위의 네 가지로 좁혀 말한다.
   - 학습된 생성자에 적용할 수 있는 상수가 명시된 인증 bound.
   - 관측량 수준의 선택 규칙과 오차 예산.
   - dissipative 불완전함까지 포함하는 charge 분류.
   - 장치 고유 noise의 스펙트럼 의존성.
2. 정리 G는 Kumar–Hauke–Halimeh 2022의 쌍대 상황임을 밝힌다. 그쪽은 noise가 오차원이고 우리는 noise가 보호 자원이다. 1/|Q|는 따름정리로 제시한다.
3. 정리 E, F는 확산 꼬리 문헌과 Han–Hartnoll을 인용하고, 포물형 평활화와의 유사성을 밝힌다. 새로움은 "rapid mixing 없이 보존 밀도에 대한 시간 균일 인증"에 둔다.
4. 데이터 분석은 BEQS 부록 A3의 재시뮬레이션과 Olsacher 2025의 plateau 진단을 인용한다. 우리 분석이 그 위에서 무엇을 더했는지(창 의존성, 기억 시험, 모델 선택)를 분명히 한다.

**출처 (4단계 재점검).**
- Zeno 보호와 gauge 보호:
  - [Stannigel et al., arXiv:1308.0528](https://arxiv.org/pdf/1308.0528)
  - [Halimeh et al., Gauge-symmetry protection using single-body terms, PRX Quantum 2, 040311](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040311)
  - [Halimeh–Hauke review, arXiv:2204.13709](https://arxiv.org/pdf/2204.13709)
  - [Local pseudogenerators, Commun. Phys. (2025)](https://www.nature.com/articles/s42005-025-02035-y)
  - [Kumar–Hauke–Halimeh, 1/f noise, arXiv:2210.06489](https://arxiv.org/pdf/2210.06489)
  - [Schmale–Weimer, arXiv:2404.16454](https://arxiv.org/pdf/2404.16454)
  - [Symmetry-protection Zeno phase transition, arXiv:2405.18504](https://arxiv.org/pdf/2405.18504)
- 무작위 gauge 변환:
  - [Lamm–Lawrence–Yamauchi, arXiv:2005.12688](https://arxiv.org/abs/2005.12688)
  - [Zeno suppression of gauge drift, arXiv:2405.09462](https://arxiv.org/pdf/2405.09462)
  - [Ball, PRA 111, 042428](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.042428)
- 강한 dissipation과 Zeno 극한의 엄밀 결과:
  - [Zanardi–Campos Venuti, arXiv:1404.4673](https://www.arxiv.org/abs/1404.4673)
  - [Zanardi–Campos Venuti, arXiv:1412.6198](https://ar5iv.labs.arxiv.org/html/1412.6198)
  - [Burgarth et al., arXiv:1807.02036](https://arxiv.org/pdf/1807.02036)
  - [Burgarth et al., arXiv:2111.08961](https://arxiv.org/pdf/2111.08961v2)
- 약한/강한 대칭:
  - [Lieu et al., arXiv:2008.02816](https://www.arxiv.org/abs/2008.02816)
- 시간 균일성과 수송:
  - [Trivedi–Cirac, arXiv:2509.17579](https://arxiv.org/pdf/2509.17579)
  - [Han–Hartnoll, arXiv:1806.01859](https://arxiv.org/html/1806.01859)
  - [Operator backflow, arXiv:2111.09904](https://arxiv.org/pdf/2111.09904)
  - [Poggi et al., arXiv:2007.01901](https://arxiv.org/abs/2007.01901)
- 학습:
  - [Olsacher et al., arXiv:2405.06768](https://arxiv.org/pdf/2405.06768)
  - [Guo et al., arXiv:2408.03801](https://arxiv.org/pdf/2408.03801)
  - [van den Berg et al., arXiv:2512.08165](https://arxiv.org/pdf/2512.08165)
  - [Gleinig–Eisert, arXiv:2209.14328](https://arxiv.org/pdf/2209.14328)
- 대칭을 쓰는 오차 완화:
  - [GUESS, arXiv:2603.13060](https://arxiv.org/abs/2603.13060)
- 상관 dephasing:
  - [Open-system dynamics of entanglement review (Monz 2011 논의), arXiv:1402.3713](https://arxiv.org/pdf/1402.3713)

## 출처

- [Kraft et al., arXiv:2511.23392](https://arxiv.org/abs/2511.23392) · [Zenodo 17750419](https://zenodo.org/records/17750419)
- [Cai–Tong–Preskill, arXiv:2311.14818](https://arxiv.org/abs/2311.14818)
- [Rao–Eisert–Guaita, arXiv:2510.08467](https://arxiv.org/abs/2510.08467)
- [Kashyap et al., arXiv:2404.11081](https://arxiv.org/abs/2404.11081)
- [Angrisani et al., arXiv:2609.40174](https://arxiv.org/abs/2609.40174)
- [Daryanoosh, arXiv:2604.16869](https://arxiv.org/abs/2604.16869)
- [Stannigel et al., arXiv:1308.0528](https://arxiv.org/abs/1308.0528)
- [Zeno-protected DFS, quant-ph/0008050](https://arxiv.org/abs/quant-ph/0008050)
- [Tran et al., symmetry protection](https://hub.jqi.umd.edu/publications/faster-digital-quantum-simulation-symmetry-protection)
- [Symmetry-protected coherent relaxation, arXiv:1802.05293](https://arxiv.org/abs/1802.05293)
- [Graph symmetry and correlated dephasing, arXiv:2603.10654](https://arxiv.org/abs/2603.10654)
- (4단계) [Heyl, Hauke, Zoller, Sci. Adv. 5, eaau8342 (2019)](https://www.science.org/doi/10.1126/sciadv.aau8342) · [Trivedi et al., Nat. Commun. 15, 6507 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11297267/) · [Kashyap et al., PRX 15, 021017 (2025)](https://dx.doi.org/10.1103/PhysRevX.15.021017) · [Paladino et al., 1/f noise review, arXiv:1304.7925](https://arxiv.org/pdf/1304.7925) · [Non-Markovian DFS, quant-ph/0505051](https://arxiv.org/pdf/quant-ph/0505051) · [APS March Meeting 2025 abstract on noise-protected DFS](https://archive.aps.org/smt/2025/mar-c31/3)
- [Time-continuous error cancellation, arXiv:2601.05952](https://arxiv.org/pdf/2601.05952)
- [QEM for non-Hermitian dynamics, arXiv:2602.21879](https://arxiv.org/html/2602.21879v1)
