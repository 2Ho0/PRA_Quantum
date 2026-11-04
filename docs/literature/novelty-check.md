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

## 3. 새로움 주장 (방어 가능한 형태)

1. **Heisenberg forgetting bound (정리 A).** 기술적으로는 단순하다. Duhamel 공식에 Heisenberg 그림을 쓴 것이다. 주장의 초점은 "BEQS의 Duhamel bound를 항상 개선하고, dissipation이 있으면 포화한다"는 것이다. 이것 자체를 핵심 새로움으로 내세우지는 않는다.
2. **U(1) charge 분해 정리 B1–B4.** 내용은 다음과 같다.
   - 선택 규칙: charge-0 관측량에서 charged 섭동의 1차 기여가 0이다.
   - charged 관측량: 시간에 무관한 bound가 나온다.
   - Zeno 억제: 모든 차수에서 성립하는 bound가 나온다.

   검색 범위에서 이 조합을 다룬 연구는 보이지 않았다. 핵심 새로움이다.
3. **학습된 ensemble에서의 실효 파라미터 수 감소.** 2-local 기준으로 Hamiltonian 파라미터의 약 2/3가 U(1)을 깬다. 이 성분들은 BEQS 유형의 1차 요동 예산에서 빠진다. 실용적 새로움이다.
4. **보호 collective rate $\gamma_c^*=1/(\mathbf 1^T\Gamma^+\mathbf 1)$.** 학습된 dephasing 행렬을 정리 B에 연결하는 조작적 정의다.

## 4. 리스크

- Zeno 억제 자체는 잘 알려진 물리다. 도입부에서 Stannigel 2014, PRA 62 062305, Tran 2021을 먼저 인용해서 새로움이 "certification용 엄밀 bound와 선택 규칙"에 있다는 점을 분명히 해야 한다.
- BEQS 실험 영역(10 ions, $\gamma_c\approx0.08J_0$, 측정 시간 ≲ 6 $J_0^{-1}$)에서는 Zeno 억제가 |q|=2 오차에서만 O(1) 수준이다. 선택 규칙(B1)은 $\gamma_c$와 무관하게 성립하므로 이 영역에서도 쓸모가 있다. 수치로 확인해야 한다(E3).

## 5. 데이터 판정 (E6)

- Zenodo 10.5281/zenodo.17750419: CC BY 4.0, 6.2 MB, 파일 4개(state1_2.zip, DataHLBicL51Ions.zip, NeelDataZ51.zip, ReadMe.txt).
- **원시 측정 데이터만 들어 있다.** shot 단위 bitstring이 10진수로 압축되어 있고, 학습된 $J_{ij}$, $\Gamma$, 공분산은 없다.
  - 10 ions: 상태 2개 × random unitary 200개 × 50 shot 배치 4개 × 21 시점
  - 51 ions: 11개 시점 파일
  - Néel 51 ions: z 기저, 20세트 × 50 shot × 11 시점 (0–10 ms)
- 결론: 실데이터를 쓰려면 integral method로 재학습하는 별도 작업이 필요하다. random unitary의 기저 정보가 파일 안에 있는지 확인해야 하고, README의 연락처는 M. K. Joshi다. 이번 세션에서는 BEQS 논문에 실린 파라미터 규모를 쓴 합성 모델로 대체한다(E3).

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
- [Time-continuous error cancellation, arXiv:2601.05952](https://arxiv.org/pdf/2601.05952)
- [QEM for non-Hermitian dynamics, arXiv:2602.21879](https://arxiv.org/html/2602.21879v1)
