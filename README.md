# Which Hamiltonian errors does dissipation erase?

This repository holds theory, numerics and a manuscript draft (target: Physical Review A) on certified open-system quantum simulation under imperfect Hamiltonians. It builds on the bounded-error quantum simulation (BEQS) framework of Kraft et al., [PRX 16, 031037 (2026)](https://doi.org/10.1103/s96t-n8tx).

**Core idea.** Use dissipation and symmetry to sharpen error bounds.
- **Heisenberg forgetting.** The error is bounded by ∫‖[δH, O_u]‖du.
- **U(1) charge anatomy.** Symmetry-breaking imperfections vanish from the linear response of charge-conserving observables. Collective dephasing then suppresses them at rate 2γ_c q².
- **Size-independent commutator-form certificates.**
- **Application to the public BEQS trapped-ion data.**

## Layout

| Path | Content |
|---|---|
| `docs/theory/error-anatomy-theory.md` | Theorems and proofs (English) |
| `docs/results/summary.md` | Results summary with limits and next steps (Korean) |
| `docs/literature/novelty-check.md` | Related work and novelty (Korean) |
| `docs/data/beqs-data-notes.md` | Structure of the BEQS data, the inferred conventions, and a draft e-mail to the authors |
| `docs/superpowers/` | Design specs and execution plans |
| `src/qerr/` | Library: Lindblad solver, charge sectors, bounds, free fermions, quadratic-operator engine, Pauli algebra, Lindbladian learning, ion-data I/O |
| `tests/` | pytest suite |
| `experiments/` | E1–E7 figure scripts |
| `results/` | Figures (PNG and PDF) and JSON acceptance checks |
| `paper/` | Manuscript draft, kept locally only and not committed (this repository is public) |
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
```

## Data and attribution

The experimental data are those of Kraft et al. (Zenodo [10.5281/zenodo.17750419](https://doi.org/10.5281/zenodo.17750419), CC BY 4.0). They are not redistributed here.

The measurement bases of the 10-ion data are not stored in the record. We infer them from the t = 0 statistics of the two known initial states; see `docs/data/beqs-data-notes.md`. Our analysis is independent and is not endorsed by the original authors.
