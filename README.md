# Which imperfections does dissipation erase?

This repository holds theory, numerics and a manuscript draft (target: Physical Review A) on certified open-system quantum simulation under imperfect Hamiltonians and noise models. It builds on the bounded-error quantum simulation (BEQS) framework of Kraft et al., [PRX 16, 031037 (2026)](https://doi.org/10.1103/s96t-n8tx).

**Core idea.** Use dissipation and symmetry to sharpen error bounds.
- **Heisenberg forgetting.** The error is bounded by ∫‖[δH, O_u]‖du.
- **U(1) charge anatomy.** Imperfections, Hamiltonian or dissipative, split by their superoperator charge Q. Charged ones vanish from the linear response of charge-conserving observables, and collective dephasing then suppresses them at rate 2γ_c Q². Covariant ones (for example incoherent spin flips, which break only the strong symmetry) are unprotected.
- **Size-independent commutator-form certificates.**
- **Conserved charges.** The forgetting of an observable follows its multipole order: bounds grow as √t for densities and are time-uniform from quadrupoles on.
- **Collective noise with memory.** The selection rule and the noise-independence of covariant errors hold for collective noise of any spectrum; the Zeno suppression of charged errors survives memory but scales as 1/|Q| for quasi-static Gaussian noise instead of 1/Q² for white noise.
- **Forgetting in both pictures.** Pairing the observable's density gradients with the state's currents and gradients gives a time-uniform local bound for conserved densities, where every Heisenberg-only bound grows as √t.
- **Application to the public BEQS trapped-ion data.** Re-learned 10-ion generator, a certified collective dephasing rate within a structured model, a charge-resolved validation, and strong-symmetry diagnostics of a 51-ion quench. Phase 4 adds model selection (a minimal U(1)-symmetric model with collective dephasing only and a growing collective rate predicts the 5–10 ms data three times better than the full fit), a simulation-based calibration of the learning method, and a calibrated rejection of white collective noise.

## Layout

| Path | Content |
|---|---|
| `docs/theory/error-anatomy-theory.md` | Theorems and proofs, phases 1–2 (English) |
| `docs/theory/phase3-theory.md` | Weak/strong symmetry classes (Theorem B6), multipole forgetting (Theorem E) |
| `docs/theory/phase4-theory.md` | Collective noise with memory (Theorem G), forgetting in both pictures (Theorem F) |
| `docs/results/summary.md` | Results summary with limits and next steps (Korean) |
| `docs/research-qa.md` | Plain-language Q&A: what the project does, why symmetry and dissipation help, contributions (Korean) |
| `docs/literature/novelty-check.md` | Related work and novelty (Korean) |
| `docs/data/beqs-data-notes.md` | Structure of the BEQS data, the inferred conventions, and a draft e-mail to the authors |
| `docs/superpowers/` | Design specs and execution plans |
| `src/qerr/` | Library: Lindblad solver, charge sectors, bounds, free fermions, quadratic-operator engine, Pauli algebra, Lindbladian learning (windows, weighted columns), ion-data I/O, classical collective noise (`noise`), two-sided pairing (`hydro`), shot-level sampling (`shots`) |
| `tests/` | pytest suite |
| `experiments/` | E1–E16 figure scripts (E6, E6c, E9, E11, E11b–E11e, E14, E15 use the BEQS data) |
| `results/` | Figures (PNG and PDF) and JSON acceptance checks |
| `paper/` | Manuscript draft, kept locally only and not committed (this repository is public); writing is paused, the research notes in `docs/` are the record |
| `data/fetch_beqs.py` | Downloads the BEQS raw data (Zenodo, CC BY 4.0) into `data/raw/` and checks the md5 checksums |

## Reproduce

Requirements: Python ≥ 3.11 with numpy, scipy, matplotlib and pytest.

```
pip install numpy scipy matplotlib pytest
python data/fetch_beqs.py            # about 6 MB of raw shots
python -m pytest -q                  # whole suite
cd experiments
python e1_charge_anatomy.py          # E1–E5: about 1 min each
python e2_forgetting.py
python e3_ensemble.py
python e3b_expected_error_bound.py   # run after e3
python e4_cp_pathology.py
python e5_free_fermion.py
python e7_local_zeno.py              # about 3 min (N up to 200)
python e6_ion_learning.py            # about 25 min (10-qubit learning, bootstrap, validation)
python e6_symmetry.py                # run after e6_ion_learning
python e8_symmetry_classes.py        # about 20 min (N = 6, dense superoperators)
python e9_symmetry_data.py           # about 5 min on 12 cores; run after e6_ion_learning and e6_symmetry
python e6c_validation.py             # several hours on 3 cores (learning windows up to 5 ms); run after e9
python e10_hydro_forgetting.py       # about 1 h on 12 cores (N = 400 to t = 1500)
python e11_model_selection.py        # about 1.5 h (10-qubit trajectory, 10 variants, bootstrap, validation); caches e11_stage1.npz
python e11b_calibration.py           # about 40 min; run after e11 (simulation-based calibration, memory test)
python e11c_noise_ansatz.py          # a few minutes; run after e11 (single-ion noise block on the minimal model)
python e11d_mismatch_diagnosis.py    # about 5 min on 4 cores; run after e11 (six explanations of the charged mismatch)
python e11e_noise_block_sbc.py       # about 2 h on 8 cores; run after e11c and e14 (calibration of the single-ion noise block)
python e12_noise_memory.py           # about 20 min on 16 cores (N = 6, exact noise averages)
python e12b_all_orders.py            # seconds; run after e12 (Markov-conditioned all-orders bound)
python e13_two_sided_bound.py        # about 30 min on 4 cores (N = 400 to t = 1500)
python e13b_local_pairing.py         # about 1.5 h on 3 cores; run after e13 (pointwise pairing in Corollary F-qf)
python e13c_kspace_check.py          # seconds; run after e13 (exact k-space density propagator)
python e14_shot_calibration.py       # about 25 min on 8 cores; run after e11 (shot-level calibration of the memory test)
python e15_coverage.py               # about 2.5 h on 16 cores, resumable; run after e11 (certified and statistical bands against held-out data)
python e16_interacting_forgetting.py # about 4 h on 2 cores (memory-bound, slower next to other jobs); run any time (dephased XXZ chain, N = 10, exact)
```

E8–E16 and E6c cache their results in `results/*.npz`; `--replot` redraws the figure from the cache. Parallel runs are memory-bound on Windows (commit limit): run the long scripts one or two at a time.

## Data and attribution

The experimental data are those of Kraft et al. (Zenodo [10.5281/zenodo.17750419](https://doi.org/10.5281/zenodo.17750419), CC BY 4.0). They are not redistributed here.

The measurement bases of the 10-ion data are not stored in the record. We infer them from the t = 0 statistics of the two known initial states; see `docs/data/beqs-data-notes.md`. Our analysis is independent and is not endorsed by the original authors.
