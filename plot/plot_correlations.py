#!/usr/bin/env python3
import hashlib
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from accumxov.accum_utils import compute_correlations, blockwise_cholesky
from accumxov.Amat import Amat
from scipy.sparse import diags, vstack
import scipy.linalg as la


def _compute_tail_correlation_from_cholesky(L, sigma0, tail_size):
    """
    Compute only the last `tail_size x tail_size` correlation block.
    This avoids building the full covariance/correlation matrices.
    """
    n = L.shape[0]
    if tail_size <= 0 or tail_size > n:
        raise ValueError("tail_size must be in [1, n].")

    first_idx = n - tail_size
    tail_indices = first_idx + np.arange(tail_size)
    return _compute_subset_correlation_from_cholesky(L=L, sigma0=sigma0, indices=tail_indices)


def _compute_subset_correlation_from_cholesky(L, sigma0, indices):
    """Compute correlation matrix only for selected parameter indices."""
    n = L.shape[0]
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError("indices must be a non-empty 1D array")
    if np.min(idx) < 0 or np.max(idx) >= n:
        raise ValueError("indices out of bounds")

    m = len(idx)
    rhs = np.zeros((n, m), dtype=L.dtype)
    rhs[idx, np.arange(m)] = 1.0

    x = la.solve_triangular(L, rhs, lower=True, check_finite=False, overwrite_b=True)
    cx = sigma0 ** 2 * (x.T @ x)
    stddev = np.sqrt(np.clip(np.diag(cx), a_min=np.finfo(float).tiny, a_max=None))
    return cx / np.outer(stddev, stddev)


def _normalize_name(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, bytes):
        return x.decode(errors="ignore").strip()

    arr = np.asarray(x)
    if arr.ndim == 0:
        return str(arr.item()).strip()
    if arr.dtype.kind in ("U", "S"):
        return "".join(arr.astype(str).tolist()).strip()
    return str(x).strip()


def _resolve_par_name(name, idx_map):
    raw = _normalize_name(name)
    candidates = [raw, f"dR/d{raw}"]

    if raw.startswith("dR/d"):
        candidates.append(raw[4:])

    for cand in candidates:
        if cand in idx_map:
            return cand
    return None


def _indices_from_names(available_names, names):
    idx_map = {name: i for i, name in enumerate(available_names)}
    resolved = []
    missing = []
    for name in names:
        got = _resolve_par_name(name, idx_map)
        if got is None:
            missing.append(_normalize_name(name))
        else:
            resolved.append(got)

    if missing:
        raise ValueError(
            f"Requested parameters not found in cached/available names (tried raw and dR/d* forms): {missing}"
        )

    return np.array([idx_map[name] for name in resolved], dtype=int), [_normalize_name(x) for x in names]


def _cache_signature(abmat_path):
    st = os.stat(abmat_path)
    return int(st.st_mtime_ns), int(st.st_size)


def _split_artifacts_exist(abmat_base_path):
    meta = abmat_base_path + ".json"
    mat = abmat_base_path + "_mat.npz"
    mat_nosol = abmat_base_path + "_mat_nosol.npz"
    return os.path.exists(meta) and (os.path.exists(mat) or os.path.exists(mat_nosol))


def _load_amat_with_migration(abmat_base_path):
    split_exists = _split_artifacts_exist(abmat_base_path)

    if split_exists:
        try:
            amat = Amat(vecopts={}).load(
                abmat_base_path,
                read_matrices=True,
                read_matrices_nosol=False,
            )
            print(f"Loaded Abmat split format: {abmat_base_path}.json")
            return amat
        except Exception as exc:
            print(f"Split-format load failed, fallback to pickle: {exc}")

    abmat_pkl_path = abmat_base_path + ".pkl"
    with open(abmat_pkl_path, "rb") as file:
        amat = pickle.load(file)
    print(f"Loaded Abmat legacy pickle: {abmat_pkl_path}")

    if not split_exists:
        try:
            Amat.migrate_legacy(abmat_pkl_path, out_path=abmat_base_path)
            print(f"Migrated Abmat to split format: {abmat_base_path}.json")
        except Exception as exc:
            print(f"Warning: failed to migrate Abmat to split format: {exc}")

    return amat


def _load_cache(cache_file, abmat_sig):
    if not os.path.exists(cache_file):
        return None

    mtime_ns, size_b = abmat_sig
    with np.load(cache_file, allow_pickle=False) as c:
        if int(c["abmat_mtime_ns"]) != mtime_ns:
            return None
        if int(c["abmat_size"]) != size_b:
            return None
        if "R" not in c.files or "param_names" not in c.files:
            return None

        return {
            "R": c["R"],
            "param_names": [_normalize_name(x) for x in c["param_names"]],
            "cache_file": cache_file,
        }


def _save_cache(cache_file, abmat_sig, r_matrix, param_names):
    mtime_ns, size_b = abmat_sig
    np.savez_compressed(
        cache_file,
        R=r_matrix,
        param_names=np.array([_normalize_name(x) for x in param_names]),
        abmat_mtime_ns=np.int64(mtime_ns),
        abmat_size=np.int64(size_b),
    )


def _subset_cache_file(pyout_folder, id_, iter_, suffix_, reorder_pars):
    key = "\x1f".join(sorted([_normalize_name(x) for x in reorder_pars]))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{pyout_folder}/correlation_cache_{id_}_{iter_}_{iter_ + 1}{suffix_}_subset_{digest}.npz"


def _build_plot_from_cache(cache_payload, reorder_pars, ng):
    r = cache_payload["R"]
    names = cache_payload["param_names"]

    if reorder_pars:
        idx, labels = _indices_from_names(names, reorder_pars)
        return r[np.ix_(idx, idx)], labels

    ng_eff = min(ng, len(names))
    if ng_eff <= 0:
        raise ValueError("No parameters available in cache.")
    start = len(names) - ng_eff
    labels = [n[4:] if n.startswith("dR/d") else n for n in names[start:]]
    return r[start:, start:], labels


def plot_correlation_matrix(R, labels=None, title="Parameter Correlation Matrix",
                            figsize=(6, 5), cmap="coolwarm", vmin=-1, vmax=1, fig_name=""):
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(R, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient", rotation=270, labelpad=15)

    if labels is not None:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            value = R[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


# Abmat_BB4_0_1_E
id = "CB9"
iter = 0
suffix = "_A"
data_path = "/explore/nobackup/people/wdesprat/AIUB_backup/pyxover/out/"

# Optional exact order for plotted rows/cols.
# Each entry can be either full name (e.g. "dR/dRA") or short name (e.g. "RA").
reorder_pars = []
# reorder_pars = ["RA","DEC","PM","LIB1","LIB2","LIB3","LIB4","LIB5","LIB6","LIB7","LIB8","LIB9","LIB10","LIB11","h2",]
reorder_pars = ["RA","DEC","PM","L","h2",]

startInit = time.time()

pyout_folder = f"{data_path}/{id}_{iter}"
# pyout_folder = f"{data_path}CC4_0"
abmat_base = f"{pyout_folder}/Abmat_{id}_{iter}_{iter + 1}{suffix}"
abmat_pkl = abmat_base + ".pkl"
abmat_sig = _cache_signature(abmat_pkl)

ng = 15
compute_full_correlation = False

full_cache_file = f"{pyout_folder}/correlation_cache_{id}_{iter}_{iter + 1}{suffix}_full.npz"
tail_cache_file = f"{pyout_folder}/correlation_cache_{id}_{iter}_{iter + 1}{suffix}_tail_ng{ng}.npz"
subset_cache_file = _subset_cache_file(pyout_folder, id, iter, suffix, reorder_pars) if reorder_pars else None

cache_candidates = []
if reorder_pars:
    cache_candidates.extend([full_cache_file, subset_cache_file, tail_cache_file])
else:
    cache_candidates.extend([tail_cache_file, full_cache_file])

R_plot = None
labels_plot = None
t_cache = time.time()
for cf in cache_candidates:
    if not cf:
        continue
    payload = _load_cache(cf, abmat_sig)
    if payload is None:
        continue
    try:
        R_plot, labels_plot = _build_plot_from_cache(payload, reorder_pars, ng)
        print(f"Loaded correlation cache: {cf}")
        break
    except ValueError:
        continue
print(f"[timing] cache lookup/build: {time.time() - t_cache:.2f}s")

if R_plot is None:
    print("Compute correlation...")

    t_step = time.time()
    Abmat = _load_amat_with_migration(abmat_base)
    print(f"[timing] load Abmat: {time.time() - t_step:.2f}s")

    sigma0 = Abmat.resid_wrmse
    all_names = [_normalize_name(x) for x in Abmat.sol4_pars]
    npar = len(all_names)
    ng_eff = min(ng, npar)

    t_step = time.time()
    spA = sum([np.sqrt(w) * diags(mask_obs.astype(float)) @ Abmat.spA_sol4 for (w, mask_obs) in zip(Abmat.vce_obs, Abmat.obs_blocks)])
    if len(Abmat.penalty_mat) > 0:
        spQ = vstack([np.sqrt(w) * diags(p.diagonal() ** 0.5) for w, p in zip(Abmat.vce_pen, Abmat.penalty_mat)])
        spA = vstack([spA, 1. * spQ])
    print(f"[timing] build weighted design matrix: {time.time() - t_step:.2f}s")

    t_step = time.time()
    N = (spA.T * spA).toarray()
    print(f"[timing] form normal matrix N: {time.time() - t_step:.2f}s")

    t_step = time.time()
    if N.shape[0] < 25000:
        L = la.cholesky(N, lower=True, overwrite_a=True, check_finite=False)
    else:
        L = blockwise_cholesky(N, block_size=20000)
    print(f"[timing] factorization (Cholesky): {time.time() - t_step:.2f}s")

    if reorder_pars:
        req_idx, labels_plot = _indices_from_names(all_names, reorder_pars)
        if compute_full_correlation:
            t_step = time.time()
            R_full, _ = compute_correlations(N=None, L=L, sigma0=sigma0)
            R_plot = R_full[np.ix_(req_idx, req_idx)]
            print(f"[timing] full correlation compute: {time.time() - t_step:.2f}s")
            _save_cache(full_cache_file, abmat_sig, R_full, all_names)
            print(f"Saved correlation cache: {full_cache_file}")
        else:
            t_step = time.time()
            R_subset = _compute_subset_correlation_from_cholesky(L=L, sigma0=sigma0, indices=req_idx)
            R_plot = R_subset
            print(f"[timing] subset correlation compute: {time.time() - t_step:.2f}s")
            resolved_names = [all_names[i] for i in req_idx]
            _save_cache(subset_cache_file, abmat_sig, R_subset, resolved_names)
            print(f"Saved correlation cache: {subset_cache_file}")
    else:
        if compute_full_correlation:
            t_step = time.time()
            R_full, _ = compute_correlations(N=None, L=L, sigma0=sigma0)
            R_plot = R_full[npar - ng_eff:, npar - ng_eff:]
            print(f"[timing] full correlation compute: {time.time() - t_step:.2f}s")
            labels_plot = [n[4:] if n.startswith("dR/d") else n for n in all_names[npar - ng_eff:]]
            _save_cache(full_cache_file, abmat_sig, R_full, all_names)
            print(f"Saved correlation cache: {full_cache_file}")
        else:
            t_step = time.time()
            R_tail = _compute_tail_correlation_from_cholesky(L=L, sigma0=sigma0, tail_size=ng_eff)
            tail_names = all_names[npar - ng_eff:]
            R_plot = R_tail
            print(f"[timing] tail correlation compute: {time.time() - t_step:.2f}s")
            labels_plot = [n[4:] if n.startswith("dR/d") else n for n in tail_names]
            _save_cache(tail_cache_file, abmat_sig, R_tail, tail_names)
            print(f"Saved correlation cache: {tail_cache_file}")

fig_name = f"correlation_{id}_{iter}{suffix}.png"
t_step = time.time()
plot_correlation_matrix(R_plot, fig_name=fig_name, labels=labels_plot)
print(f"[timing] plot/save figure: {time.time() - t_step:.2f}s")

endInit = time.time()
print(f"Finished after {str(endInit - startInit)}s")
