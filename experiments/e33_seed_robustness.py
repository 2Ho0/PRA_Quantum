"""E33: do the numbers and verdicts depend on the random seeds? (preregistered)

Sec. V of the manuscript reports one random instance per numerical test, the checks C1-C3 of E28 use 8 synthetic
realisations, and every data bootstrap uses a single seed. E33 repeats these with new seeds.

Method. e33_seedpatch adds k * 1,000,003 to every integer seed given to numpy.random.default_rng (k = 0 leaves the
seeds unchanged), redirects the output folder of the scripts to results/e33_parts/, and passes the same patch to the
worker processes. The experiment scripts themselves run unchanged, as they are in the working tree; the sha256 of every
script and library module used is stored with each part.

Part A. The numerics of Sec. V on 21 instances, k = 0, 1, ..., 20: E1, E2, E3 with E3b, E5, E7, E8. A new k draws new
random z fields of the base model, new random imperfections, new coefficient ensembles (E3) and a new random coupling
pattern of the free-fermion chain (E5, E7).
  A0. k = 0 reproduces the committed result files: every number within 1e-6 relative plus 1e-12 absolute, every flag
      equal (run times excluded). If A0 fails, the cause is found and reported before A1-A3 are read.
  A1. Statements that are theorems or exact identities hold in all 21 instances: E1 bounds hold and the charge-0
      error is exactly independent of gamma_c; E2 observed <= Theorem 1 <= Kraft et al. at all times; E3 the
      symmetric ensemble is independent of gamma_c (max difference < 1e-9, both observables); E5 the on-site
      (charge-0) error is independent of gamma_c (max difference < 1e-12); E7 the certificate lies above the observed
      error at every grid time; E8 D[X]_0 = D[sigma+] + D[sigma-] (< 1e-12), class II is independent of gamma_c
      (leakage and m_s), classes III and II have identical leakage, and the bound of Theorem B6 holds for the
      leakage and for m_s at all times.
  A2. Statements that hold generically but not by theorem hold in at least 19 of the 20 new instances (k >= 1):
      E1 doubling ratio 3.6-4.4 for charged and 1.8-2.2 for symmetric imperfections (charge-0 observable), the
      |q| = 2 error decreases with gamma_c (0.1 -> 0.5 -> 2), and its reduction from gamma_c = 0 to 2 is larger than
      that of the |q| = 1 error; E2 saturation (growth after t = 30 below 5%) and late slope within 10% of
      ||[dH, O_inf]||; E3 the ratio of the charged to the symmetric width scales with sigma (ratio of ratios in
      [0.2, 0.5]), the bound of Kraft et al. covers the observed band where it is below 1, and the charged mean shift
      at t = 20 decreases with gamma_c (both observables); E3b the leading-order expected-error term lies above the
      observed bias for t <= 1 and below the bound of Kraft et al. (both gamma_c), and its charged part at t = 4 is
      smaller for gamma_c = 1; E5 N-independence within 5% and monotone decrease with gamma_c; E7 certificate
      N-independent within 5%, global bound ratio N = 200/50 within 16 +- 25%, certificate decreasing with gamma_c,
      grid convergence below 2%; E8 class I leakage of second order (ratio 3.5-4.5) and decreasing with gamma_c
      (gamma_c >= 0.5), classes II and III of first order (1.8-2.2), the charged remainder of m_s of second order
      (3.9-4.1) and decreasing with gamma_c (gamma_c >= 0.25).
      An item that fails in 2 or more new instances is reported as instance dependent, and the manuscript sentence
      that relies on it must be qualified.
  A3. Every number of Sec. V quoted in the manuscript is reported as median (min-max) over the 21 instances, next
      to its value at k = 0.

Part B. The checks C1-C3 of E28 with 32 realisations instead of 8: the 8 committed ones and r = 8, ..., 31, with the
seeds of E28's own formula (no seed shift).
  B0. The first 8 lower bounds of C1 equal the committed ones (1e-9 relative).
  B1 (C1). Per world, E9's bound lies below the target in at least 29 of 32 realisations (probability 0.93 for an
      exact 95% lower bound). The Clopper-Pearson 95% interval of the coverage is reported.
  B2 (C2). Per group, the pooled slope lies in [0.8, 1.2] and the residual sd is at most 1.5 s_g.
  B3 (C3). The data's relative change of J_nn at [0, 5] ms lies within the data-matched world's mean +- 2 sd over the
      32 realisations. The share explained is reported.
  The verdicts with 8 realisations (C1: met, failed, met for shot-only, data-matched, full drift; C2: met; C3:
  failed) are compared with these.

Part C. The data bootstraps with two new seeds, k = 1 and 2 (k = 0 is committed): E6a and E9 (setting bootstrap of
the ten-ion learning), E30 and E32 (shot bootstraps), and the E11 bootstrap with E15 (whose bands are built from the
bootstrap refits).
  C0. The point estimates do not change: E6a theta and the E11 parameters of every variant equal the committed ones
      (1e-8 relative to the largest entry).
  C1. Every flag (verdict) of these result files equals the committed one.
  C2. E6a: no U(1)-breaking Hamiltonian coupling (K, L, Bx, By) reaches |z| = 3.
  C3. The bootstrap standard deviations that the manuscript quotes or uses change by at most 25% (the sampling error
      of an sd from 100 replicates is about 7%): E6a J0, alpha, dominant Gamma eigenvalue and structured gamma_c;
      E9 structured gamma_c for both ansaetze; E11 J_nn and gamma_c for every bootstrapped variant.
  Reported: the relative change of every number of these files, largest first.

Run: `--part A`, `--part B`, `--part C` (resumable; at most two parts at once), then `--analyse`. The parts run as
child processes at below-normal priority with one BLAS thread each. Output: results/e33_seed_robustness.json; the
parts are kept in results/e33_parts/ (not committed).
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS0 = ROOT / "results"            # committed results (read only)
PARTS = RESULTS0 / "e33_parts"
OUT = RESULTS0 / "e33_seed_robustness.json"
K_A, K_C, N_REAL_B = range(21), (1, 2), 32
SLOTS = {"A": 10, "B": 10, "C": 18}
BLAS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


# ---------------------------------------------------------------- tasks (run in a child process)
def _load(name, k, cap, out):
    import e33_seedpatch as sp
    mod = importlib.import_module(name)
    if hasattr(mod, "ProcessPoolExecutor"):
        mod.ProcessPoolExecutor = sp.pool_factory(k, out, cap)
    return mod


def task_E1(k, cap, out):
    m = _load("e1_charge_anatomy", k, cap, out)
    data, flags = m.run()
    return {"e1_charge_anatomy": m.metrics(data, flags)}


def task_E2(k, cap, out):
    m = _load("e2_forgetting", k, cap, out)
    return {"e2_forgetting": {"acceptance": m.acceptance(m.run())}}


def task_E3(k, cap, out):
    m = _load("e3_ensemble", k, cap, out)
    o = m.run()
    acc = m.acceptance(o)
    np.savez(out / "e3_mean_shift.npz", T=m.T, **{f"s{s}_g{gc}_{ens.split()[0]}": o[(s, gc, ens)]["mean_shift"]
                                                  for s in m.SIGMAS for gc in m.GAMMA_CS for ens in m.ENSEMBLES})
    b = _load("e3b_expected_error_bound", k, cap, out)   # reads e3_mean_shift.npz from `out`
    res, M = b.run()
    return {"e3_ensemble": {"M": o["M"], "M_sym": o["M_sym"], "acceptance": acc},
            "e3b_expected_error_bound": {"acceptance": b.acceptance(res, M)}}


def task_E5(k, cap, out):
    m = _load("e5_free_fermion", k, cap, out)
    _, acc = m.analyse(m.run())
    return {"e5_free_fermion": {"acceptance": acc}}


def task_E7(k, cap, out):
    m = _load("e7_local_zeno", k, cap, out)
    _, _, acc = m.analyse(*m.run())
    return {"e7_local_zeno": {"acceptance": acc}}


def task_E8(k, cap, out):
    m = _load("e8_symmetry_classes", k, cap, out)
    return {"e8_symmetry_classes": {"acceptance": m.analyse(m.run())}}


def task_B_e28(k, cap, out):
    os.environ["E28_WORKERS"] = str(cap)
    for f in ("e6_ion_learning.npz", "e9_symmetry_data.json"):   # inputs of E28's analysis
        shutil.copy(RESULTS0 / f, out / f)
    m = importlib.import_module("e28_claim_checks")   # explicit seeds per realisation: no seed shift, no pool patch
    m.N_REAL = N_REAL_B
    m.RESULTS = out    # the 32-realisation npz goes here; the per-realisation caches stay in results/e28_parts
    m.run()
    o, lbs, tgt = m.analyse()
    return {"e28_claim_checks": o, "lower_bounds": {w: np.asarray(v).tolist() for w, v in lbs.items()},
            "target": float(tgt)}


def task_C_learn(k, cap, out):
    shutil.copy(RESULTS0 / "e6_symmetry.json", out / "e6_symmetry.json")   # input of E9's analysis
    m = _load("e6_ion_learning", k, cap, out)
    r, ans, obs = m.run()
    _, _, acc = m.analyse(r, ans, obs)
    np.savez_compressed(out / "e6_ion_learning.npz", theta=r["theta"], boot=r["boot"], thetas=r["thetas"],
                        h_terms=np.array([repr(p) for p in ans.h_terms]))
    th0 = np.load(RESULTS0 / "e6_ion_learning.npz")["theta"]
    e9 = _load("e9_symmetry_data", k, cap, out)   # E9's main, with its output in `out`
    fits, bloch, purity, n_set, spam = e9.run()
    np.savez_compressed(out / "e9_symmetry_data.npz", bloch=np.array(bloch), n_set=n_set,
                        spam_theta=spam["theta"], spam_contrast=spam["contrast"],
                        purity=np.array([[purity[(s, j)] for j in range(1, e9.N + 1)] for s in (0, 1)]),
                        **{key: np.array([f[key] for f in fits]) for key in e9.FIT_KEYS})
    z = np.load(out / "e9_symmetry_data.npz")
    fits = [{key: z[key][i] for key in e9.FIT_KEYS} for i in range(z["ext"].shape[0])]
    purity = {(s, j): tuple(z["purity"][s, j - 1]) for s in (0, 1) for j in range(1, e9.N + 1)}
    spam = {"theta": z["spam_theta"], "contrast": z["spam_contrast"]}
    o9 = e9.analyse(fits, list(z["bloch"]), purity, int(z["n_set"]), spam)[0]
    return {"e6_ion_learning": {"acceptance": acc}, "e9_symmetry_data": {"results": o9},
            "C0": {"E6a theta": float(np.abs(r["theta"] - th0).max() / np.abs(th0).max())}}


def task_C_digital(k, cap, out):
    res = {}
    for name in ("e30_digital_sector", "e32_charge_dephasing"):
        m = _load(name, k, cap, out)
        sys.argv = [name + ".py"]
        m.main()
        res[name] = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    return res


def task_C_model(k, cap, out):
    shutil.copy(RESULTS0 / "e6_ion_learning.npz", out / "e6_ion_learning.npz")   # E11 starts from the E6a fit
    e11 = _load("e11_model_selection", k, cap, out)
    e11.STAGE1 = out / "e11_stage1.npz"
    z0 = np.load(RESULTS0 / "e11_model_selection.npz")
    path = out / "e11_model_selection.npz"
    c0 = {}
    if not path.exists():
        times, s1, s2, bases = e11.load()
        est, boot = e11.learn_and_bootstrap(times, s1, s2, bases)
        c0 = {f"E11 theta {v}": float(np.abs(est[v][0] - z0[f"theta_{v}"]).max() / np.abs(z0[f"theta_{v}"]).max())
              for v in e11.VARIANTS}
        arrays = {key: z0[key] for key in z0.files if not key.startswith("boot_")}
        arrays.update({f"boot_theta_{v}": np.array([b[v][0] for b in boot]) for v in e11.BOOT})
        arrays.update({f"boot_beta_{v}": np.array([b[v][1] for b in boot]) for v in e11.BOOT})
        np.savez_compressed(path, **arrays)
        (out / "C0.json").write_text(json.dumps(c0))
    c0 = json.loads((out / "C0.json").read_text())
    z = np.load(path)   # E11's main from here on, with the new bootstrap and the committed point fits
    times, data, sem = z["times"], z["data"], z["sem"]
    est = {v: (z[f"theta_{v}"], float(z[f"beta_{v}"]), float(z[f"res_{v}"])) for v in e11.VARIANTS}
    pred = {v: z[f"pred_{v}"] for v in e11.VARIANTS}
    boot = [{v: (z[f"boot_theta_{v}"][i], float(z[f"boot_beta_{v}"][i])) for v in e11.BOOT}
            for i in range(z["boot_theta_X15"].shape[0])]
    o11 = e11.analyse(times, est, boot, pred, data, sem)
    e15 = _load("e15_coverage", k, cap, out)   # E15's main; its parts and inputs live in `out`
    cert, bands = e15.run()
    np.savez_compressed(out / "e15_coverage.npz", **{f"band_{v}": b for v, b in bands.items()},
                        **{f"{n}|osc": c[0] for n, c in cert.items()}, **{f"{n}|integrand": c[1] for n, c in cert.items()},
                        **{f"{n}|hnorm": c[2] for n, c in cert.items()})
    zz = np.load(out / "e15_coverage.npz")
    cert = {n: (float(zz[f"{n}|osc"]), zz[f"{n}|integrand"], zz[f"{n}|hnorm"]) for n in e15.OBS}
    bands = {v: zz[f"band_{v}"] for v in e15.MODELS}
    out_c, _ = e15.analyse_certified(cert, z, bands["X15"])
    out_b, charged = e15.analyse_bands(bands, z)
    o15 = {"certified (Theorem A, X15)": out_c, "statistical bands": out_b, "predictions": {
        "certified band covers >= 90% out of sample for every observable":
            all(r["out-of-sample coverage, Theorem A band + 1.96 SEM"] >= 0.9 for r in out_c.values()),
        "memory model has the smaller calibration factor in every charged class":
            all(out_b["Xb15"][n]["calibration factor t > 1.5 ms"] < out_b["X15"][n]["calibration factor t > 1.5 ms"]
                for n in charged)}}
    return {"e11_model_selection": o11, "e15_coverage": o15, "C0": c0}


TASKS = {"E1": task_E1, "E2": task_E2, "E3": task_E3, "E5": task_E5, "E7": task_E7, "E8": task_E8,
         "B_e28": task_B_e28, "C_learn": task_C_learn, "C_digital": task_C_digital, "C_model": task_C_model}
CAPS = {"E1": 1, "E2": 1, "E3": 4, "E5": 4, "E7": 5, "E8": 1, "B_e28": 10, "C_learn": 4, "C_digital": 1, "C_model": 8}


def plan(part):
    if part == "A":
        return [(t, k) for k in K_A for t in ("E8", "E7", "E3", "E5", "E1", "E2")]
    if part == "B":
        return [("B_e28", 0)]
    return [(t, k) for t in ("C_model", "C_learn", "C_digital") for k in K_C]


def part_dir(part, task, k):
    return PARTS / part / f"{task}_k{k}"


def script_hashes():
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and (Path(f).resolve().parent in (HERE, ROOT / "src" / "qerr")):
            out[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
    return dict(sorted(out.items()))


def child(part, task, k, cap):
    import e33_seedpatch as sp
    out = part_dir(part, task, k)
    out.mkdir(parents=True, exist_ok=True)
    sp.apply(k, None if task == "B_e28" else out)   # before any experiment module is imported
    t0 = time.time()
    res = TASKS[task](k, cap, out)
    from common import save_json
    save_json(out / "result.json", {"part": part, "task": task, "k": k, "seed offset": k * sp.OFFSET,
                                    "seconds": round(time.time() - t0, 1), "modules": script_hashes(),
                                    "result": res})


# ---------------------------------------------------------------- driver
def drive(part):
    todo = [(t, k) for t, k in plan(part) if not (part_dir(part, t, k) / "result.json").exists()]
    env = dict(os.environ, **{v: "1" for v in BLAS}, PYTHONUNBUFFERED="1")
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    running, failed, t0 = [], [], time.time()
    print(f"part {part}: {len(todo)} tasks to run, {SLOTS[part]} slots", flush=True)
    while todo or running:
        for p, t, k, log in running[:]:
            if p.poll() is not None:
                running.remove((p, t, k, log))
                log.close()
                ok = p.returncode == 0 and (part_dir(part, t, k) / "result.json").exists()
                if not ok:
                    failed.append((t, k, p.returncode))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {t} k={k} {'done' if ok else f'FAILED ({p.returncode})'}",
                      flush=True)
        free = SLOTS[part] - sum(CAPS[t] for _, t, _, _ in running)
        for t, k in todo[:]:
            if CAPS[t] <= free:
                d = part_dir(part, t, k)
                d.mkdir(parents=True, exist_ok=True)
                log = open(d / "run.log", "a", encoding="utf-8")
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child", part, t, str(k), str(CAPS[t])]
                running.append((subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 creationflags=flags), t, k, log))
                todo.remove((t, k))
                free -= CAPS[t]
        time.sleep(5)
    print(f"part {part} finished in {(time.time() - t0) / 3600:.2f} h; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------- analysis
def flat(d, prefix=""):
    if isinstance(d, dict):
        for key, v in d.items():
            yield from flat(v, f"{prefix}/{key}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def committed(stem):
    return json.loads((RESULTS0 / f"{stem}.json").read_text(encoding="utf-8"))


def load_part(part, task, k):
    p = part_dir(part, task, k) / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))["result"] if p.exists() else None


def compare(new, old, skip="seconds"):
    """Differences between two result files: (key, new, old) for flags and numbers outside the A0 tolerance."""
    a, b = dict(flat(new)), dict(flat(old))
    diffs = [(key, a.get(key), b.get(key)) for key in sorted(set(a) ^ set(b)) if skip not in key]
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if skip in key:
            continue
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                diffs.append((key, x, y))
        elif abs(x - y) > 1e-6 * abs(y) + 1e-12:
            diffs.append((key, x, y))
    return diffs


O0 = "O0 = Z2 Z3 (q=0)"
E3_OBS = ("Z2 Z3", "staggered magnetization")


def a1_items(r):
    e1 = r["E1"]["e1_charge_anatomy"]["acceptance"]
    e2 = r["E2"]["e2_forgetting"]["acceptance"]
    e3 = r["E3"]["e3_ensemble"]["acceptance"]
    e5 = r["E5"]["e5_free_fermion"]["acceptance"]
    e7 = r["E7"]["e7_local_zeno"]["acceptance"]
    e8 = r["E8"]["e8_symmetry_classes"]["acceptance"]
    return {
        "E1 bounds hold": e1["bounds_hold"], "E1 charge-0 error exactly gamma_c-independent": e1["exact_symmetry"],
        "E2 observed <= Theorem 1 <= Kraft et al.": e2["observed <= ThmA <= BEQS everywhere"],
        "E3 symmetric ensemble gamma_c-independent": all(
            e3[o]["symmetric ensemble independent of gamma_c (max diff)"] < 1e-9 for o in E3_OBS),
        "E5 on-site error gamma_c-independent": e5["on-site (charge-0) error independent of gamma_c (max diff)"] < 1e-12,
        "E7 certificate >= observed": e7["certificate >= observed at every grid time (all N, gamma_c)"],
        "E8 D[X]_0 = D[s+] + D[s-]": e8["D[X]_0 equals D[sigma+] + D[sigma-] (max abs diff)"] < 1e-12,
        "E8 class II leakage gamma_c-independent":
            e8["leakage: class II gamma_c-independent (max rel. spread over gamma_c < 1e-9)"][1],
        "E8 class II m_s gamma_c-independent": e8["m_s: class II gamma_c-independent (max rel. spread over gamma_c < 1e-9)"][1],
        "E8 classes III and II identical leakage":
            e8["leakage: III and II identical (closed birth-death equation for P(M)), max |diff| < 1e-12"][1],
        "E8 Theorem B6 bound holds (leakage)": e8["leakage: Theorem B6 bound >= |Delta_III| at all times"],
        "E8 Theorem B6 bound holds (m_s)": e8["m_s: Theorem B6 bound >= |Delta_III| at all times"],
    }


def a2_items(r):
    e1 = r["E1"]["e1_charge_anatomy"]
    e2 = r["E2"]["e2_forgetting"]["acceptance"]
    e3 = r["E3"]["e3_ensemble"]["acceptance"]
    e3b = r["E3"]["e3b_expected_error_bound"]["acceptance"]
    e5 = r["E5"]["e5_free_fermion"]["acceptance"]
    e7 = r["E7"]["e7_local_zeno"]["acceptance"]
    e8 = r["E8"]["e8_symmetry_classes"]["acceptance"]
    md = e1["max_delta_eps0.04"]
    red = {q: md[f"gc=0.0 | P{q} (|q|={q}) | {O0}"] / md[f"gc=2.0 | P{q} (|q|={q}) | {O0}"] for q in (1, 2)}
    out = {
        "E1 charged imperfections: doubling ratio 3.6-4.4": e1["acceptance"]["first_order_vanishing (ratio 3.6-4.4)"],
        "E1 symmetric imperfection: doubling ratio 1.8-2.2": e1["acceptance"]["symmetric_first_order (ratio 1.8-2.2)"],
        "E1 |q|=2 error decreases with gamma_c": e1["acceptance"]["zeno_monotone_O0P2 (gc 0.1->0.5->2)"],
        "E1 reduction 0 -> 2 larger for |q|=2 than |q|=1": bool(red[2] > red[1]),
        "E2 saturation below 5%": e2["(b) saturation: FB(40)-FB(30) < 5% of FB(40)"],
        "E2 late slope within 10%": e2["(a) late slope matches ||[dH,O_inf]|| within 10%"],
    }
    for o in E3_OBS:
        out[f"E3 width ratio scales with sigma ({o})"] = e3[o]["B1': ratio(0.01)/ratio(0.03) in [0.2, 0.5] (expect 1/3)"]
        out[f"E3 bound of Kraft et al. covers ({o})"] = e3[o]["BEQS Eq.(11) with M_sym covers observed all-terms band wherever bound < 1"]
        out[f"E3 charged mean shift decreases with gamma_c ({o})"] = e3[o]["Zeno: charged mean shift at t=20 decreases gc 0 -> 0.1 -> 1"]
    for gc in ("0.1", "1.0"):
        out[f"E3b leading term >= observed bias, t <= 1 (gc={gc})"] = e3b[f"gc={gc}: A2 leading >= observed bias for 0 < t <= 1"]
        out[f"E3b leading term < bound of Kraft et al. (gc={gc})"] = e3b[f"gc={gc}: A2 leading < BEQS Eq.(12) for all t > 0"]
    out["E3b charged part smaller for gc=1"] = e3b["Zeno in A2: charged part at t=4 smaller for gc=1 than gc=0.1"]
    out["E5 N-independent within 5%"] = e5["N-independence within 5%"]
    out["E5 decreases with gamma_c"] = e5["Zeno: decreases monotonically with gamma_c"]
    out["E7 certificate N-independent within 5%"] = e7["certificate N-independent within 5%"]
    out["E7 global ratio within 16 +- 25%"] = e7["global ratio within 16 +- 25%"]
    out["E7 certificate decreases with gamma_c"] = e7["certificate decreases with gamma_c"]
    out["E7 grid convergence below 2%"] = e7["certificate grid convergence < 2%"]
    out["E8 class I second order"] = e8["leakage: class I second order (ratio in [3.5, 4.5], gamma_c >= 0.5)"]
    out["E8 class I decreases with gamma_c"] = e8["leakage: class I decreases with gamma_c (gamma_c >= 0.5)"]
    out["E8 classes II and III first order"] = e8["leakage: classes II and III first order (ratio in [1.8, 2.2])"]
    out["E8 charged remainder second order"] = e8["m_s: charged remainder second order in its own strength (ratio in [3.9, 4.1], all gamma_c, kappa)"]
    out["E8 charged remainder decreases with gamma_c"] = e8["m_s: remainder decreases with gamma_c (gamma_c >= 0.25)"]
    return {key: bool(v) for key, v in out.items()}


def a3_numbers(r):
    """The numbers of Sec. V as the manuscript quotes them (manuscript value in the key)."""
    e1 = r["E1"]["e1_charge_anatomy"]
    rat, md = e1["ratio_2eps_over_eps"], e1["max_delta_eps0.04"]
    charged = [v for key, v in rat.items() if key.endswith(O0) and "P0" not in key]
    e2 = r["E2"]["e2_forgetting"]["acceptance"]
    e3 = r["E3"]["e3_ensemble"]["acceptance"]["Z2 Z3"]
    e3b = r["E3"]["e3b_expected_error_bound"]["acceptance"]
    e5 = r["E5"]["e5_free_fermion"]["acceptance"]
    e7 = r["E7"]["e7_local_zeno"]["acceptance"]
    e8 = r["E8"]["e8_symmetry_classes"]["acceptance"]
    n5 = e5["|d n_c|(t=10), pairing, N=100, gamma_c = 0, 1, 4"]
    c7 = e7["certificate(t=10), N=100, gamma_c = 0, 1, 4"]
    co = [e7[f"N=100, gamma_c={g}: certificate / observed at t=10"] for g in (0, 1, 4)]
    gc_ = [e7[f"N=100, gamma_c={g}: global / certificate at t=10"] for g in (0, 1, 4)]
    l8 = e8["leakage: |Delta| at t=10, kappa=0.001, per gamma_c"]
    r8 = e8["leakage: doubling ratio at t=10 per gamma_c [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]"]
    m8 = e8["m_s: |remainder III-II| at t=10, kappa=0.001, per gamma_c"]
    q8 = e8["m_s: charged remainder ratio R(k)/R(k/2) at fixed flip rate, per gamma_c, kappa=0.002"]
    return {
        "E1 doubling ratio, charged, min [3.97]": min(charged), "E1 doubling ratio, charged, max [4.00]": max(charged),
        "E1 doubling ratio, symmetric [2.04]": rat[f"gc=0.0 | P0 (|q|=0) | {O0}"],
        "E1 |q|=2 error at gamma_c=0 [3.99e-3]": md[f"gc=0.0 | P2 (|q|=2) | {O0}"],
        "E1 |q|=2 error at gamma_c=2 [3.60e-4]": md[f"gc=2.0 | P2 (|q|=2) | {O0}"],
        "E1 |q|=1 error at gamma_c=0 [1.23e-3]": md[f"gc=0.0 | P1 (|q|=1) | {O0}"],
        "E1 |q|=1 error at gamma_c=2 [4.9e-4]": md[f"gc=2.0 | P1 (|q|=1) | {O0}"],
        "E2 late growth after t=30, relative [1.4e-4]": e2["(b) relative late growth"],
        "E2 Kraft et al. / Theorem 1 at t=40 [41]": e2["BEQS/ThmA at t=40, (b)"],
        "E3 width ratio at sigma=0.03 [0.21]": e3["ch/sym width ratio at t=1.05, sigma=0.03"],
        "E3 width ratio at sigma=0.01 [0.075]": e3["ch/sym width ratio at t=1.05, sigma=0.01"],
        "E3b Kraft et al. / leading term, gc=0.1, t=1 [1.8]": e3b["gc=0.1: BEQS/A2 at t=1"],
        "E3b Kraft et al. / leading term, gc=0.1, t=4 [13.4]": e3b["gc=0.1: BEQS/A2 at t=4"],
        "E3b Kraft et al. / leading term, gc=1, t=1 [2.5]": e3b["gc=1.0: BEQS/A2 at t=1"],
        "E3b Kraft et al. / leading term, gc=1, t=4 [23.9]": e3b["gc=1.0: BEQS/A2 at t=4"],
        "E5 relative spread over N >= 50 [0.007]": e5["N-independence: |d n_c|(t=10), pairing, N>=50, max rel spread"],
        "E5 reduction from gamma_c=0 to J [35]": n5[0] / n5[2],
        "E7 global bound ratio N=200/N=50 [13.8]": e7["global B3 ratio N=200 / N=50 (expect ~16)"],
        "E7 certificate / observed, min [33]": min(co), "E7 certificate / observed, max [53]": max(co),
        "E7 global / certificate, min [700]": min(gc_), "E7 global / certificate, max [1300]": max(gc_),
        "E7 certificate at gamma_c=0 [0.64]": c7[0], "E7 certificate at gamma_c=J/4 [0.080]": c7[1],
        "E7 certificate at gamma_c=J [0.029]": c7[2],
        "E7 relative spread N=100 vs 200 [3e-9]": e7["certificate(t=10) rel. spread N=100 vs 200 (max over gamma_c)"],
        "E8 class I doubling ratio, gamma_c >= 0.5, min [4.00]": min(r8["I"][2:]),
        "E8 class II doubling ratio [1.91]": r8["II"][0],
        "E8 class I leakage at gamma_c=0.5 [8.1e-5]": l8["I"][2], "E8 class I leakage at gamma_c=4 [1.6e-5]": l8["I"][5],
        "E8 charged remainder ratio, min [4.000]": min(q8), "E8 charged remainder ratio, max [4.000]": max(q8),
        "E8 m_s remainder at gamma_c=0 [2.0e-7]": m8[0], "E8 m_s remainder at gamma_c=4 [7.7e-11]": m8[5],
    }


def analyse_A():
    names = ["E1", "E2", "E3", "E5", "E7", "E8"]
    runs = {k: {t: load_part("A", t, k) for t in names} for k in K_A}
    done = [k for k in K_A if all(runs[k].values())]
    out = {"instances complete": done}
    if 0 in done:
        diffs = {}
        for t in names:
            for stem, res in runs[0][t].items():
                d = compare(res, committed(stem))
                if d:
                    diffs[stem] = [list(x) for x in d[:20]]
        out["A0: k = 0 reproduces the committed files"] = not diffs
        out["A0 differences"] = diffs
    new = [k for k in done if k > 0]
    a1 = {k: a1_items(runs[k]) for k in done}
    a2 = {k: a2_items(runs[k]) for k in new}
    out["A1: theorems hold in every instance"] = {
        item: f"{sum(a1[k][item] for k in done)} of {len(done)}" for item in a1[done[0]]} if done else {}
    out["A1 passes"] = bool(done) and all(all(v.values()) for v in a1.values())
    out["A2: generic statements, new instances passing"] = {
        item: f"{sum(a2[k][item] for k in new)} of {len(new)}" for item in a2[new[0]]} if new else {}
    out["A2 instance-dependent items (fail in 2 or more new instances)"] = [
        item for item in (a2[new[0]] if new else {}) if sum(not a2[k][item] for k in new) >= 2]
    out["A2 failing instances per item"] = {item: [k for k in new if not a2[k][item]] for item in (a2[new[0]] if new else {})
                                           if any(not a2[k][item] for k in new)}
    nums = {k: a3_numbers(runs[k]) for k in done}
    out["A3: median (min - max) over instances; k = 0"] = {
        item: {"median": float(np.median([nums[k][item] for k in done])),
               "min": float(min(nums[k][item] for k in done)), "max": float(max(nums[k][item] for k in done)),
               "k = 0": nums[0][item] if 0 in nums else None} for item in (nums[done[0]] if done else {})}
    out["A3 values per instance"] = {item: [nums[k][item] for k in done] for item in (nums[done[0]] if done else {})}
    return out


def analyse_B():
    from scipy.stats import beta, binom
    r = load_part("B", "B_e28", 0)
    if r is None:
        return {"complete": False}
    o, lbs, tgt = r["e28_claim_checks"], r["lower_bounds"], r["target"]
    old = committed("e28_claim_checks")["C1: E9's certified gamma_c"]
    out = {"complete": True, "realisations": N_REAL_B}
    b0 = all(np.allclose(np.array(lbs[w][:8]) * 1e3, old[w]["lower bound per realisation (rad/s)"], rtol=1e-9, atol=0)
             for w in lbs)
    out["B0: first 8 lower bounds equal the committed ones"] = bool(b0)
    out["P(hits >= 29 of 32) for an exact 95% bound"] = float(binom.sf(28, N_REAL_B, 0.95))
    c1 = {}
    for w, lo in lbs.items():
        hits = int(np.sum(np.array(lo) <= tgt))
        new = int(np.sum(np.array(lo[8:]) <= tgt))
        c1[w] = {"bound below the target": f"{hits} of {N_REAL_B}", "of the 24 new realisations": f"{new} of 24",
                 "with 8 (committed)": old[w]["bound below the target"],
                 "coverage, Clopper-Pearson 95%": [float(beta.ppf(0.025, hits, N_REAL_B - hits + 1)) if hits else 0.0,
                                                   float(beta.ppf(0.975, hits + 1, N_REAL_B - hits)) if hits < N_REAL_B else 1.0],
                 "rule B1 (>= 29 of 32) passes": hits >= 29}
    out["B1 (C1)"] = c1
    c2 = o["C2: identification of the U(1)-breaking terms"]
    out["B2 (C2)"] = {g: c2[g] for g in ("K", "L", "B")}
    out["B2 terms with |estimate| > 3 sd per realisation"] = c2["terms with |estimate| > 3 E6a sd, per realisation (of 110)"]
    c3 = o["C3: J against the window"]
    out["B3 (C3)"] = {key: c3[key] for key in c3 if key.startswith(("rule", "share", "data"))}
    out["B3 (C3)"]["data-matched relative change: mean, sd"] = c3["data-matched"]["relative change: mean, sd"]
    out["full E28 analysis with 32 realisations"] = o
    out["post hoc (added after the run; no verdict depends on it): why C1 undercovers"] = posthoc_c1()
    return out


def posthoc_c1():
    """Bias, spread and bias correction of E9's bound in the 32 realisations (post hoc, 2026-11-22)."""
    z = np.load(part_dir("B", "B_e28", 0) / "e28_claim_checks.npz")
    g, b, tgt = z["c1_gamma"] * 1e3, z["c1_boot"] * 1e3, float(z["c1_target"]) * 1e3
    out = {}
    for i, w in enumerate(("shot-only", "data-matched", "full drift")):
        perc = np.percentile(b[i], 5, axis=1)
        basic = 2 * g[i] - np.percentile(b[i], 95, axis=1)
        out[w] = {"bias of the estimate (rad/s)": float(g[i].mean() - tgt),
                  "sd of the estimate over the realisations (rad/s)": float(g[i].std(ddof=1)),
                  "bootstrap sd, mean over the realisations (rad/s)": float(b[i].std(axis=1, ddof=1).mean()),
                  "bootstrap mean minus estimate, mean (rad/s)": float((b[i].mean(axis=1) - g[i]).mean()),
                  "basic bound is the smaller one": f"{int(np.sum(basic <= perc))} of {N_REAL_B}",
                  "percentile bound below the target": f"{int(np.sum(perc <= tgt))} of {N_REAL_B}",
                  "basic bound below the target": f"{int(np.sum(basic <= tgt))} of {N_REAL_B}"}
    return out


C_STEMS = {"C_learn": ("e6_ion_learning", "e9_symmetry_data"), "C_digital": ("e30_digital_sector", "e32_charge_dephasing"),
           "C_model": ("e11_model_selection", "e15_coverage")}
SD_KEYS = {"e6_ion_learning": ["/acceptance/J0 fit bootstrap sd (rad/s)", "/acceptance/alpha fit bootstrap sd",
                               "/acceptance/Gamma dominant eigenvalue sd (rad/s)",
                               "/acceptance/gamma_c structured bootstrap sd (rad/s)"],
           "e9_symmetry_data": ["/results/(b) structured gamma_c, E6a ansatz (rad/s): estimate, bootstrap mean, sd, 95% lower bound (percentile, basic), certified = min[2]",
                                "/results/(b) structured gamma_c, extended ansatz (rad/s): estimate, bootstrap mean, sd, 95% lower bound (percentile, basic), certified = min[2]"]}


def analyse_C():
    out = {}
    for k in K_C:
        runs = {t: load_part("C", t, k) for t in C_STEMS}
        rk = {"complete": {t: r is not None for t, r in runs.items()}}
        flags_same, sds, changes, c0 = {}, {}, {}, {}
        for t, r in runs.items():
            if r is None:
                continue
            c0.update(r.get("C0", {}))
            for stem in C_STEMS[t]:
                new, old = dict(flat(r[stem])), dict(flat(committed(stem)))
                fl = [(key, new.get(key), old.get(key)) for key in sorted(set(new) | set(old))
                      if isinstance(new.get(key), bool) or isinstance(old.get(key), bool)]
                flags_same[stem] = [list(x) for x in fl if x[1] != x[2]]
                rel = []
                for key in set(new) & set(old):
                    x, y = new[key], old[key]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) \
                            and not isinstance(y, bool) and abs(y) > 1e-12:
                        rel.append((abs(x - y) / abs(y), key, x, y))
                rel.sort(reverse=True)
                changes[stem] = {"numbers compared": len(rel), "changed by more than 1e-9": sum(c > 1e-9 for c, *_ in rel),
                                 "largest relative changes": [[key, x, y, c] for c, key, x, y in rel[:15]]}
                keys = SD_KEYS.get(stem, [])
                if stem == "e11_model_selection":
                    keys = [key for key in old if key.startswith("/bootstrap/") and key.endswith((" J_nn sd", "/J_nn sd", "/gamma_c sd"))]
                for key in keys:
                    sds[f"{stem}{key}"] = [new.get(key), old.get(key),
                                           None if new.get(key) is None else new[key] / old[key] - 1.0]
                if stem == "e6_ion_learning":
                    zmax = {g: new[f"/acceptance/{g}: rms (rad/s), max |z|, #|z|>3 of {n}[1]"]
                            for g, n in (("K", 45), ("L", 45), ("Bx", 10), ("By", 10))}
                    rk["C2: largest |z| of the U(1)-breaking couplings"] = zmax
                    rk["C2 passes"] = all(v < 3 for v in zmax.values())
        rk["C0: point estimates unchanged (max relative difference)"] = c0
        rk["C0 passes"] = bool(c0) and all(v < 1e-8 for v in c0.values())
        rk["C1: flags that differ from the committed run"] = flags_same
        rk["C1 passes"] = all(not v for v in flags_same.values())
        rk["C3: bootstrap sd (new, committed, relative change)"] = sds
        rk["C3 passes"] = bool(sds) and all(v[2] is not None and abs(v[2]) <= 0.25 for v in sds.values())
        rk["relative changes"] = changes
        out[f"k = {k}"] = rk
    return out


def analyse():
    from common import save_json
    out = {"A: Sec. V numerics on new random instances": analyse_A(), "B: E28 with 32 realisations": analyse_B(),
           "C: data bootstraps with new seeds": analyse_C()}
    save_json(OUT, out)
    a, b, c = out.values()
    print("A0:", a.get("A0: k = 0 reproduces the committed files"), "| A1:", a.get("A1 passes"),
          "| A2 instance-dependent:", a.get("A2 instance-dependent items (fail in 2 or more new instances)"))
    for item, v in a.get("A3: median (min - max) over instances; k = 0", {}).items():
        print(f"  {item}: {v['median']:.4g} ({v['min']:.4g} - {v['max']:.4g}); k=0 {v['k = 0']:.4g}")
    if b.get("complete"):
        print("B0:", b["B0: first 8 lower bounds equal the committed ones"])
        for w, v in b["B1 (C1)"].items():
            print(f"  C1 {w}: {v['bound below the target']} (8: {v['with 8 (committed)']}), pass {v['rule B1 (>= 29 of 32) passes']}")
        print("  C2:", {g: v for g, v in b["B2 (C2)"].items()})
        print("  C3:", b["B3 (C3)"])
    for k, v in c.items():
        print(k, "C0", v["C0 passes"], "C1", v["C1 passes"], "C2", v.get("C2 passes"), "C3", v["C3 passes"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C"])
    ap.add_argument("--child", nargs=4, metavar=("PART", "TASK", "K", "CAP"))
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.child:
        p, t, k, cap = args.child
        child(p, t, int(k), int(cap))
    elif args.part:
        sys.exit(drive(args.part))
    elif args.analyse:
        analyse()
