# BEQS 공개 데이터 노트 (Zenodo 10.5281/zenodo.17750419)

- 출처: Kraft et al., PRX 16, 031037 (2026). 라이선스 CC BY 4.0.
- 내려받기: `python data/fetch_beqs.py`. 받은 파일은 `data/raw/`에 저장되며, 이 폴더는 git에서 제외된다. md5 검증을 포함한다.

## 1. 파일 구조 (직접 확인함)

| 묶음 | 구조 | 사용 여부 |
|---|---|---|
| `state1_2/state{1,2}/DataHLRU{1..200}.mat` | `dataT` = 1×4 셀(50 shot 배치). 각 셀은 (21, 51)이고 1열은 시간(μs, 0–10 ms, 0.5 ms 간격), 나머지 50열은 shot이다 | **사용 (E6a)** |
| `DataHLBicL51Ions/DataHLBicL51Ions{0..1000}.mat` | 셀 18개, 각 (50, 51). 1열은 설정 번호(1–50)뿐이다 | 사용 불가: 무작위 초기 상태와 측정 설정의 정의가 없다 |
| `NeelDataZ51/DataHL51IonmeasType14Base3.mat` | 셀 20개, 각 (11, 51). 1열은 시간(μs, 0–10 ms)이고 z-기저 50 shot이다 | **사용 (E6b)** |

**비트 규약 (ReadMe).**
- shot 값은 10진수로 저장되어 있고, qubit 1이 최하위 비트다.
- 비트 0은 측정 기저 고유값 +1, 비트 1은 −1이다.
- Néel 데이터에서 "1은 spin down"이다. t=0의 Néel 10진수 1501199875790165는 qubit 1이 down인 교대 패턴으로 풀린다(테스트로 고정함).

**불규칙한 점.** state 2의 설정 RU1과 RU59에는 배치가 4개가 아니라 8개다. 모든 설정의 가중치를 같게 하려고 처음 4개만 쓴다(ledger ruling).

## 2. 측정 기저 복원 (추론이며, 저자 확인 전)

BEQS 부록 B 3에 따르면 10-ion 학습 데이터의 설정은 무작위 Pauli 기저 $\bigotimes_iP_i$, $P_i\in\{x,y,z\}$이다. 그러나 파일에는 설정 번호만 있다. 두 초기 상태가 알려진 곱 상태이므로 t=0 통계만으로 기저를 정할 수 있다.

- **상태 1** (θ=0.33π, 회전축 −y): Bloch 벡터 $(\sin0.33\pi,0,-\cos0.33\pi)$. 예측 평균은 X ≈ +0.86, Y ≈ 0, Z ≈ −0.51이다.
- **상태 2** (θ=0.5π, 회전축 x): Bloch 벡터 $(0,1,0)$. 예측 평균은 Y ≈ +1, X·Z ≈ 0이다.
- **판정 규칙.** 상태 2의 평균이 0.5보다 크면 Y. 아니면 상태 1의 평균이 0.2보다 크면 X, 그렇지 않으면 Z.
- **결과 (2000 설정·qubit).**
  - X/Y/Z = 675/669/656.
  - 군집 평균: X는 0.867(상태 1), Y는 0.993(상태 2), Z는 −0.448(상태 1).
  - 판정 경계까지의 최소 여유는 3.1 SEM이고, 5 SEM 미만은 87개다.
  - 상태 1과 상태 2는 같은 설정을 쓴다. 모든 설정에서 Y qubit이 일치한다.
- Z 군집이 예측보다 약간 작은 것(−0.45 대 −0.51)은 ion별 회전각 편차(BEQS 부록 B: 최대 5%)와 측정 오차로 설명된다.

## 3. 저자에게 보낼 확인 메일 초안 (보내지 않음)

> **To:** manoj.joshi@uibk.ac.at
> **Subject:** Question on the measurement settings in the BEQS Zenodo dataset (10.5281/zenodo.17750419)
>
> Dear Dr. Joshi,
>
> Thank you and your coauthors for making the raw data of "Bounded-Error Quantum Simulation via Hamiltonian and Lindbladian Learning" publicly available. We are preparing a theory paper on symmetry-resolved error bounds for open-system quantum simulation that builds on your work, and we would like to include an independent analysis of your 10-ion data, with full credit to your paper and dataset.
>
> The files DataHLRU1–200.mat store the random measurement settings only as indices. Using the known initial product states (θ = 0.33π about −y and θ = 0.5π about x), we inferred the Pauli basis of every qubit in every setting from the t = 0 statistics. The three clusters are well separated, and states 1 and 2 give consistent assignments. Could you confirm that the settings are random Pauli bases shared between the two states, and, if possible, share the list of bases so that we can check our inference?
>
> We also noticed that settings 1 and 59 of state 2 contain eight batches of 50 shots instead of four. Are the additional batches repetitions of the same setting?
>
> Finally, if the learned Hamiltonian and Lindbladian parameters (with uncertainties) for the 10-ion data are available, we would be glad to compare them with our re-analysis before submission.
>
> With best regards,
> [name, affiliation]
