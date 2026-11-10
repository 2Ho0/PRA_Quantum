"""R1: does one commit of the code reproduce every committed result from the public raw data, without caches?
(preregistered)

Why. The manuscript will cite numbers from 59 result files (E1-E29b). They were produced over seven days while the
library (src/qerr, 19 commits) and the shared helpers changed, and many runs read caches of earlier runs. This run checks
that a single commit of the code reproduces every committed number from the raw data alone.

Procedure (fixed before the run).
1. Reference: the result JSON files committed at REFERENCE = 0bc5b71 (59 files, 7207 leaves). The commit that adds this
   script changes no result.
2. Setup (`--setup DEST`, in the working tree): a fresh `git clone` of this repository at the current commit, in a
   directory outside the working tree. Everything under results/ is deleted. The raw data are downloaded from Zenodo
   (10.5281/zenodo.17750419; measured by Kraft et al., PRX 16, 031037 (2026); CC BY 4.0) and their md5 checksums are
   checked (data/fetch_beqs.py). Nothing is read from the working tree's results/ or data/.
3. Run (`--run`, inside the clone): the test suite, then every stage of STAGES with the project interpreter, each after
   the stages it depends on. Dependencies are read from the scripts: the result files a script or the modules it imports
   refer to, and the imports themselves. IMPORT_ONLY modules are imported only for helpers that read no results.
   Scripts run unchanged, with their own worker and thread settings; a script without thread settings gets 2 BLAS
   threads. At most two heavy stages run at a time, below normal priority, because the machine is shared.
4. Comparison (`--compare DEST`, in the working tree): each reproduced JSON against the reference, leaf by leaf (dict
   keys and list positions).
   - Structure: the same set of leaf paths and leaf kinds (int and float are both numbers).
   - Booleans, None and strings: identical. Numbers inside a string are compared as numbers (below), the rest of the
     string as text.
   - Numbers: identical if |a - b| <= 1e-12 + 1e-9 max(|a|, |b|); agree if |a - b| <= 1e-9 + 1e-3 max(|a|, |b|);
     otherwise they differ. A non-finite value must be the same.
   - Excluded: wall-clock times (leaf paths containing "seconds per").
   A file is identical, agrees or differs by its worst leaf. A file that is not reproduced differs.
Rule (fixed before the run). R1 passes if all 59 files agree or are identical and no boolean or string leaf changes.
Every file that differs is diagnosed afterwards (post hoc, labelled so): which leaves, by how much, whether a verdict
changes, and why (code changed after the run, non-determinism, library). The research notes and the manuscript then use
the reproduced values, and every change of a value they cite is listed.
Also reported: the test-suite result, the figures regenerated, the wall time of each stage and the software versions.

Usage (the clone path is an example):
  python experiments/reproduce_all.py --plan
  python experiments/reproduce_all.py --selftest
  python experiments/reproduce_all.py --setup <DEST>
  cd <DEST> && python experiments/reproduce_all.py --run       (resumable: finished stages are kept)
  python experiments/reproduce_all.py --status <DEST>
  python experiments/reproduce_all.py --compare <DEST>        (writes results/r1_reproduction.json)
"""

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = "0bc5b71"
PY = sys.executable
MAX_HEAVY, MAX_LIGHT = 2, 2
ATOL_ID, RTOL_ID, ATOL_AG, RTOL_AG = 1e-12, 1e-9, 1e-9, 1e-3
EXCLUDE = ("seconds per",)
IMPORT_ONLY = {"e6c_validation", "e9_symmetry_data", "e11h_transient_truths"}
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
RANK = {"identical": 0, "agree": 1, "differ": 2}

# (stage, script, arguments, heavy, estimated minutes, estimated memory in GB, earlier stages of the same script)
# Estimates come from the original runs (README, logs); they only order the queue and gate on free memory.
STAGES = [
    ("pytest", None, [], False, 10, 2, []),
    ("e1", "e1_charge_anatomy.py", [], False, 1, 1, []),
    ("e2", "e2_forgetting.py", [], False, 1, 1, []),
    ("e3", "e3_ensemble.py", [], False, 2, 1, []),
    ("e3b", "e3b_expected_error_bound.py", [], False, 1, 1, []),
    ("e4", "e4_cp_pathology.py", [], False, 1, 1, []),
    ("e5", "e5_free_fermion.py", [], False, 1, 1, []),
    ("e7", "e7_local_zeno.py", [], False, 12, 2, []),
    ("e8", "e8_symmetry_classes.py", [], True, 20, 6, []),
    ("e10", "e10_hydro_forgetting.py", [], True, 60, 6, []),
    ("e12", "e12_noise_memory.py", [], True, 20, 8, []),
    ("e12b", "e12b_all_orders.py", [], False, 1, 1, []),
    ("e13", "e13_two_sided_bound.py", [], True, 30, 4, []),
    ("e13b", "e13b_local_pairing.py", [], True, 90, 4, []),
    ("e13c", "e13c_kspace_check.py", [], False, 1, 1, []),
    ("e13d", "e13d_hypocoercivity.py", [], False, 1, 1, []),
    ("e13e", "e13e_weighted_l1.py", [], True, 20, 4, []),
    ("e16", "e16_interacting_forgetting.py", [], True, 240, 8, []),
    ("e16b", "e16b_classical_reduction.py", [], True, 30, 6, []),
    ("e16c", "e16c_kls_monte_carlo.py", [], True, 75, 4, []),
    ("e21", "e21_concentration_check.py", [], True, 20, 4, []),
    ("e6", "e6_ion_learning.py", [], True, 25, 6, []),
    ("e6b", "e6_symmetry.py", [], True, 30, 18, []),
    ("e9", "e9_symmetry_data.py", [], True, 10, 6, []),
    ("e6c", "e6c_validation.py", [], True, 240, 6, []),
    ("e11", "e11_model_selection.py", [], True, 90, 6, []),
    ("e11b", "e11b_calibration.py", [], True, 40, 6, []),
    ("e11c", "e11c_noise_ansatz.py", [], False, 5, 3, []),
    ("e11d", "e11d_mismatch_diagnosis.py", [], False, 5, 3, []),
    ("e14", "e14_shot_calibration.py", [], True, 25, 6, []),
    ("e11e", "e11e_noise_block_sbc.py", [], True, 30, 6, []),
    ("e15", "e15_coverage.py", [], True, 150, 8, []),
    ("e17", "e17_neutral_diagnosis.py", [], True, 15, 4, []),
    ("e11f-truths", "e11f_nonmarkov_truths.py", ["--truths"], True, 90, 6, []),
    ("e11f-sbc", "e11f_nonmarkov_truths.py", ["--sbc"], True, 60, 6, ["e11f-truths"]),
    ("e11g", "e11g_time_point_influence.py", [], True, 30, 6, []),
    ("e18", "e18_basis_sensitivity.py", [], True, 30, 6, []),
    ("e18b", "e18b_familywise.py", [], False, 1, 1, []),
    ("e19", "e19_flip_rate_crosscheck.py", [], True, 15, 4, []),
    ("e11h-truths", "e11h_transient_truths.py", ["--truths"], True, 60, 6, []),
    ("e11h-sbc", "e11h_transient_truths.py", ["--sbc"], True, 60, 6, ["e11h-truths"]),
    ("e20", "e20_flip_models.py", [], True, 60, 6, []),
    ("e22", "e22_first_milliseconds.py", [], True, 60, 6, []),
    ("e22b", "e22b_collective_flips.py", [], True, 20, 4, []),
    ("e23", "e23_correlated_process.py", [], False, 5, 2, []),
    ("e23c", "e23c_coupling_noise.py", [], True, 60, 6, []),
    ("e23d", "e23d_scale_bias.py", [], False, 5, 2, []),
    ("e23k-truths", "e23k_imbalance_sources.py", ["--truths"], True, 60, 6, []),
    ("e23k", "e23k_imbalance_sources.py", [], True, 30, 6, ["e23k-truths"]),
    ("e23k-d", "e23k_imbalance_sources.py", ["--part-d"], True, 20, 6, ["e23k"]),
    ("e23s", "e23s_state_generators.py", [], True, 15, 4, []),
    ("e23s-r", "e23s_state_generators.py", ["--run-params"], True, 10, 4, ["e23s"]),
    ("e24", "e24_eiv_correction.py", [], True, 20, 6, []),
    ("e24b", "e24b_simex_parts.py", [], True, 20, 4, []),
    ("e25-band", "e25_structured_eiv.py", ["--band"], True, 117, 6, []),
    ("e25-syn", "e25_structured_eiv.py", ["--synthetic"], True, 104, 6, []),
    ("e25", "e25_structured_eiv.py", ["--analyse"], False, 5, 2, ["e25-band", "e25-syn"]),
    ("e25b", "e25b_bias_decomposition.py", [], True, 25, 6, []),
    ("e26", "e26_setting_drift.py", [], True, 60, 6, []),
    ("e27-data", "e27_marginal_band.py", ["--data"], True, 258, 8, []),
    ("e27-syn", "e27_marginal_band.py", ["--synthetic"], True, 192, 8, []),
    ("e27-sens", "e27_marginal_band.py", ["--sensitivity"], True, 15, 6, ["e27-data", "e27-syn"]),
    ("e27", "e27_marginal_band.py", ["--analyse"], True, 15, 6, ["e27-data", "e27-syn", "e27-sens"]),
    ("e27b", "e27b_coverage_diagnostic.py", [], True, 15, 4, []),
    ("e27c", "e27c_drift_shrinkage.py", [], True, 20, 6, []),
    ("e28-run", "e28_claim_checks.py", ["--run"], True, 75, 6, []),
    ("e28", "e28_claim_checks.py", ["--analyse"], False, 3, 2, ["e28-run"]),
    ("e29-syn", "e29_neutral_trajectory.py", ["--synthetic"], True, 66, 8, []),
    ("e29-data", "e29_neutral_trajectory.py", ["--data"], True, 45, 6, []),
    ("e29", "e29_neutral_trajectory.py", ["--analyse"], False, 3, 2, ["e29-syn", "e29-data"]),
    ("e29b", "e29b_window_diagnostic.py", [], True, 10, 4, []),
]

FILE_RE = re.compile(r"""["'](e\d+[a-z]?(?:_[A-Za-z0-9]+)*)\.(?:npz|json|npy)["']""")
PARTS_RE = re.compile(r"""["'](e\d+[a-z]?)_parts["']""")
IMPORT_RE = re.compile(r"^[ \t]*(?:from|import)[ \t]+(e\d+[a-z]?_[A-Za-z0-9_]+)\b", re.M)
TAG_RE = re.compile(r"^(e\d+[a-z]?)")
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------------------------------- dependencies and plan
def sources():
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(HERE.glob("e*.py"))}


def producer(name, srcs):
    """The script that writes the result file (or parts directory) `name` (without extension)."""
    if name in srcs:
        return name
    tag = TAG_RE.match(name).group(1)
    cands = [s for s in srcs if TAG_RE.match(s).group(1) == tag]
    if len(cands) != 1:
        raise ValueError(f"no unique producer for {name}: {cands}")
    return cands[0]


def needs(stem, srcs, code_only=False, stack=()):
    """Scripts whose results `stem` may read: those of its own file references (skipped for code_only) and, through
    each imported module, the module itself and what it needs (IMPORT_ONLY modules: only what their imports need)."""
    out = set() if code_only else {producer(r, srcs) for r in FILE_RE.findall(srcs[stem])} | {
        producer(t + "_parts", srcs) for t in PARTS_RE.findall(srcs[stem])}
    for m in set(IMPORT_RE.findall(srcs[stem])):
        if m == stem or m not in srcs or m in stack:
            continue
        only = m in IMPORT_ONLY
        out |= needs(m, srcs, only, stack + (stem,))
        if not only:
            out.add(m)
    out.discard(stem)
    return out


def plan():
    """{stage: dict(script, args, heavy, est, mem, deps)} and the critical length of each stage (minutes)."""
    srcs = sources()
    by_script = {}
    for sid, script, *_ in STAGES:
        if script:
            by_script.setdefault(Path(script).stem, []).append(sid)
    missing = sorted(set(srcs) - set(by_script))
    if missing:
        raise ValueError(f"scripts without a stage: {missing}")
    st = {}
    for sid, script, args, heavy, est, mem, own in STAGES:
        deps = set(own)
        if script:
            for y in needs(Path(script).stem, srcs):
                deps |= set(by_script[y])
        st[sid] = dict(script=script, args=args, heavy=heavy, est=est, mem=mem, deps=sorted(deps))
    order, seen, temp = [], set(), set()

    def visit(s):
        if s in seen:
            return
        if s in temp:
            raise ValueError(f"dependency cycle through {s}")
        temp.add(s)
        for d in st[s]["deps"]:
            visit(d)
        temp.discard(s)
        seen.add(s)
        order.append(s)

    for s in st:
        visit(s)
    children = {s: [] for s in st}
    for s, v in st.items():
        for d in v["deps"]:
            children[d].append(s)
    crit = {}
    for s in reversed(order):
        crit[s] = st[s]["est"] + max((crit[c] for c in children[s]), default=0)
    return st, crit


def print_plan():
    st, crit = plan()
    for s, v in st.items():
        print(f"{s:12s} {'heavy' if v['heavy'] else 'light'} {v['est']:4d} min  path {crit[s]:5d} min  <- "
              f"{', '.join(v['deps']) or '-'}")
    print(f"{len(st)} stages, {sum(v['est'] for v in st.values())} min of estimated work, "
          f"critical path {max(crit.values())} min")


# ------------------------------------------------------------------------------------------------------- setup and run
def memory_gb():
    """(available physical memory, available commit, commit limit) in GB."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = MS()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2**30, m.apf / 2**30, m.tpf / 2**30


def setup(dest):
    dest = Path(dest).resolve()
    if dest.exists():
        raise SystemExit(f"{dest} exists; choose a new directory")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    removed = []
    for p in sorted((dest / "results").iterdir()):
        removed.append(p.name)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    fetch = subprocess.run([PY, str(dest / "data" / "fetch_beqs.py")], capture_output=True, text=True)
    info = {"commit": commit, "time": datetime.now().isoformat(timespec="seconds"), "removed from results/": removed,
            "data fetch": fetch.stdout.strip().splitlines(), "data fetch exit code": fetch.returncode}
    (dest / "repro_setup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v if k != "removed from results/" else len(v) for k, v in info.items()}, indent=2))
    if fetch.returncode:
        raise SystemExit("data download failed")


def stage_env(script):
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    if script is None or "NUM_THREADS" not in (HERE / script).read_text(encoding="utf-8"):
        env.update({v: "2" for v in THREAD_VARS})
    return env


def save_state(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_all():
    if not (ROOT / "repro_setup.json").exists():
        raise SystemExit("--run belongs inside a clone made by --setup")
    st, crit = plan()
    logs = ROOT / "repro_logs"
    logs.mkdir(exist_ok=True)
    state_path = ROOT / "repro_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"stages": {}}
    for s in st:
        rec = state["stages"].setdefault(s, {"status": "pending"})
        if rec["status"] in ("running", "blocked"):
            rec["status"] = "pending"
    state["python"] = sys.version
    driver_log = open(logs / "_driver.log", "a", encoding="utf-8")

    def note(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        driver_log.write(line + "\n")
        driver_log.flush()

    running = {}
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    while True:
        for s, (proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            rec = state["stages"][s]
            rec.update(status="done" if rc == 0 else "failed", returncode=rc, end=datetime.now().isoformat(
                timespec="seconds"), minutes=round((time.time() - t0) / 60, 1))
            del running[s]
            note(f"{rec['status']:6s} {s} ({rec['minutes']} min, exit {rc})")
            save_state(state_path, state)
        status = {s: state["stages"][s]["status"] for s in st}
        for s in st:
            if status[s] == "pending" and any(status[d] in ("failed", "blocked") for d in st[s]["deps"]):
                state["stages"][s]["status"] = status[s] = "blocked"
                note(f"blocked {s}")
        ready = sorted((s for s in st if status[s] == "pending" and all(status[d] == "done" for d in st[s]["deps"])),
                       key=lambda s: -crit[s])
        n_heavy = sum(st[s]["heavy"] for s in running)
        for s in ready:
            v = st[s]
            if (v["heavy"] and n_heavy >= MAX_HEAVY) or (not v["heavy"] and len(running) - n_heavy >= MAX_LIGHT):
                continue
            avail, commit_free, commit_limit = memory_gb()
            if avail < v["mem"] + 2:
                continue
            cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"] if v["script"] is None else \
                [PY, "-u", v["script"], *v["args"]]
            fh = open(logs / f"{s}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT if v["script"] is None else HERE, env=stage_env(v["script"]),
                                    stdout=fh, stderr=subprocess.STDOUT, creationflags=flags)
            running[s] = (proc, fh, time.time())
            n_heavy += v["heavy"]
            state["stages"][s].update(status="running", start=datetime.now().isoformat(timespec="seconds"),
                                      command=" ".join(cmd[1:]))
            note(f"start  {s}  [{' '.join(cmd[1:])}]  free RAM {avail:.1f} GB, free commit {commit_free:.1f} of "
                 f"{commit_limit:.1f} GB")
            save_state(state_path, state)
        if not running and not ready:
            break
        time.sleep(20)
    counts = {k: sum(r["status"] == k for r in state["stages"].values()) for k in ("done", "failed", "blocked")}
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    save_state(state_path, state)
    note(f"finished: {counts}")


def status(dest):
    dest = Path(dest).resolve()
    state = json.loads((dest / "repro_state.json").read_text(encoding="utf-8"))
    st, crit = plan()
    by = {}
    for s, r in state["stages"].items():
        by.setdefault(r["status"], []).append(s)
    now = datetime.now()
    for s in by.get("running", []):
        t0 = datetime.fromisoformat(state["stages"][s]["start"])
        print(f"running {s} for {(now - t0).total_seconds() / 60:.0f} min (estimate {st[s]['est']} min)")
    left = [s for s in st if state["stages"].get(s, {}).get("status") != "done"]
    print({k: len(v) for k, v in by.items()}, "| failed:", by.get("failed", []), "| blocked:", by.get("blocked", []))
    print(f"remaining critical path (estimates): {max((crit[s] for s in left), default=0)} min")


# ------------------------------------------------------------------------------------------------------------ compare
def flatten(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, path + (i,))
    else:
        yield path, x


def number_status(a, b):
    if math.isnan(a) or math.isnan(b):
        return ("identical" if math.isnan(a) and math.isnan(b) else "differ"), None
    if math.isinf(a) or math.isinf(b):
        return ("identical" if a == b else "differ"), None
    d, m = abs(a - b), max(abs(a), abs(b))
    rel = d / m if m > 0 else 0.0
    if d <= ATOL_ID + RTOL_ID * m:
        return "identical", rel
    if d <= ATOL_AG + RTOL_AG * m:
        return "agree", rel
    return "differ", rel


def leaf_status(a, b):
    """(status, relative difference or None, verdict_changed)."""
    def is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if is_num(a) and is_num(b):
        s, rel = number_status(float(a), float(b))
        return s, rel, False
    if isinstance(a, str) and isinstance(b, str):
        if NUM_RE.sub("#", a) != NUM_RE.sub("#", b):
            return "differ", None, True
        worst, rel = "identical", 0.0
        for u, v in zip(NUM_RE.findall(a), NUM_RE.findall(b)):
            s, r = number_status(float(u), float(v))
            worst = max(worst, s, key=RANK.get)
            rel = max(rel, r or 0.0)
        return worst, rel, False
    same = type(a) is type(b) and a == b
    return ("identical" if same else "differ"), None, not same


def compare_json(ref, new):
    """Leaf-by-leaf comparison of two parsed JSON documents."""
    fr = {p: v for p, v in flatten(ref) if not any(e in " / ".join(map(str, p)) for e in EXCLUDE)}
    fn = {p: v for p, v in flatten(new) if not any(e in " / ".join(map(str, p)) for e in EXCLUDE)}
    out = {"leaves": len(fr), "counts": {"identical": 0, "agree": 0, "differ": 0}, "max relative difference": 0.0,
           "missing leaves": [list(p) for p in fr if p not in fn], "extra leaves": [list(p) for p in fn if p not in fr],
           "verdict changes": [], "differing leaves": []}
    for p, a in fr.items():
        if p not in fn:
            continue
        s, rel, verdict = leaf_status(a, fn[p])
        out["counts"][s] += 1
        if rel is not None:
            out["max relative difference"] = max(out["max relative difference"], rel)
        if verdict:
            out["verdict changes"].append({"path": list(p), "reference": a, "reproduced": fn[p]})
        elif s == "differ":
            out["differing leaves"].append({"path": list(p), "reference": a, "reproduced": fn[p], "relative": rel})
    out["differing leaves"].sort(key=lambda d: -(d["relative"] or math.inf))
    worst = "differ" if out["missing leaves"] or out["extra leaves"] else "identical"
    for k in ("agree", "differ"):
        if out["counts"][k]:
            worst = max(worst, k, key=RANK.get)
    out["status"] = worst
    return out


def compare(dest):
    dest = Path(dest).resolve()
    names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", REFERENCE, "results/"],
                           capture_output=True, text=True, check=True).stdout.split()
    names = sorted(n for n in names if n.endswith(".json") and not n.startswith("results/r1_"))
    files = {}
    for n in names:
        ref = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"{REFERENCE}:{n}"], capture_output=True,
                                        check=True).stdout.decode("utf-8"))
        path = dest / n
        if not path.exists():
            files[Path(n).name] = {"status": "differ", "reproduced": False}
            continue
        r = compare_json(ref, json.loads(path.read_text(encoding="utf-8")))
        r["differing leaves"] = r["differing leaves"][:40]
        files[Path(n).name] = r
    ref_figs = sorted(Path(n).name for n in subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "--name-only", REFERENCE, "results/"], capture_output=True, text=True,
        check=True).stdout.split() if n.endswith((".png", ".pdf")))
    new_figs = {p.name for p in (dest / "results").glob("*") if p.suffix in (".png", ".pdf")}
    state = json.loads((dest / "repro_state.json").read_text(encoding="utf-8"))
    setup_info = json.loads((dest / "repro_setup.json").read_text(encoding="utf-8"))
    test_log = (dest / "repro_logs" / "pytest.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
    import matplotlib
    import numpy
    import scipy
    counts = {k: sum(f["status"] == k for f in files.values()) for k in RANK}
    verdicts = sum(len(f.get("verdict changes", [])) for f in files.values())
    out = {
        "reference commit": REFERENCE, "reproduced commit": setup_info["commit"], "data fetch": setup_info["data fetch"],
        "files": len(files), "files by status": counts,
        "verdict (boolean or string) changes": verdicts,
        "R1 passes (all files agree or identical, no verdict change)": counts["differ"] == 0 and verdicts == 0,
        "test suite": test_log[-1] if test_log else None,
        "figures in the reference / regenerated": [len(ref_figs), sum(f in new_figs for f in ref_figs)],
        "figures not regenerated": [f for f in ref_figs if f not in new_figs],
        "stages": {s: {k: r.get(k) for k in ("status", "minutes", "returncode")} for s, r in state["stages"].items()},
        "software": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "per file": files,
    }
    (ROOT / "results" / "r1_reproduction.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per file", "stages")}, indent=2))
    for name, f in files.items():
        if f["status"] != "identical":
            print(f"{name}: {f['status']}, max rel {f.get('max relative difference')}, counts {f.get('counts')}, "
                  f"verdict changes {len(f.get('verdict changes', []))}")


# ----------------------------------------------------------------------------------------------------------- selftest
def selftest():
    a = {"x": 1.0, "y": [1, 2.5, True], "z": "7 of 8", "w": None, "seconds per realisation": 5.0}
    assert compare_json(a, a)["status"] == "identical"
    b = json.loads(json.dumps(a))
    b["x"] = 1.0 + 1e-10
    assert compare_json(a, b)["status"] == "identical"
    b["x"] = 1.0 + 1e-5
    assert compare_json(a, b)["status"] == "agree"
    b["x"] = 1.01
    r = compare_json(a, b)
    assert r["status"] == "differ" and r["differing leaves"][0]["path"] == ["x"]
    b = json.loads(json.dumps(a))
    b["y"][2] = False
    r = compare_json(a, b)
    assert r["status"] == "differ" and len(r["verdict changes"]) == 1
    b = json.loads(json.dumps(a))
    b["y"][1] = 2  # a changed number
    assert compare_json(a, b)["status"] == "differ" and not compare_json(a, b)["verdict changes"]
    b = json.loads(json.dumps(a))
    b["z"] = "6 of 8"
    assert compare_json(a, b)["status"] == "differ"
    b["z"] = "7 out of 8"
    assert compare_json(a, b)["verdict changes"]
    b = json.loads(json.dumps(a))
    b["seconds per realisation"] = 9.0
    assert compare_json(a, b)["status"] == "identical"
    b = json.loads(json.dumps(a))
    b["y"].append(3)
    assert compare_json(a, b)["status"] == "differ" and compare_json(a, b)["extra leaves"] == [["y", 3]]
    b = json.loads(json.dumps(a))
    b["y"][0] = 1.0
    assert compare_json(a, b)["status"] == "identical"
    assert compare_json({"v": float("nan")}, {"v": float("nan")})["status"] == "identical"
    assert compare_json({"v": 0.0}, {"v": 3e-13})["status"] == "identical"
    assert compare_json({"v": 0.0}, {"v": 3e-10})["status"] == "agree"
    assert compare_json({"v": None}, {"v": 0.0})["status"] == "differ"
    assert compare_json({"v": "(0.0, 1.5)"}, {"v": "(0.0, 1.5000001)"})["status"] == "agree"
    st, crit = plan()
    assert "e11f-truths" in st["e22"]["deps"] and "e17" in st["e11f-truths"]["deps"]
    assert "e6c" not in st["e11"]["deps"] and "e6" in st["e11"]["deps"]
    assert {"e27-data", "e27-syn", "e27-sens", "e27"} <= set(st["e29-syn"]["deps"])
    assert "e29" in st["e29b"]["deps"] and "e16" in st["e16b"]["deps"]
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--setup", metavar="DEST")
    g.add_argument("--run", action="store_true")
    g.add_argument("--status", metavar="DEST")
    g.add_argument("--compare", metavar="DEST")
    a = ap.parse_args()
    if a.plan:
        print_plan()
    elif a.selftest:
        selftest()
    elif a.setup:
        setup(a.setup)
    elif a.run:
        run_all()
    elif a.status:
        status(a.status)
    else:
        compare(a.compare)
