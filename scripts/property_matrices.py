#!/usr/bin/env python3
"""Discover property groups from cross-eval results, then compute property matrices.

Workflow (run after sweep + cross_eval):
  1. Load cross-eval faithfulness matrix (cross_eval_results.pkl).
  2. Hierarchical-cluster relations by mutual faithfulness → discover property groups.
  3. Average the best-layer decoder matrices within each cluster.
  4. Save one .npy per property (same [d, d+1] format as individual sweep matrices).

Usage:
    uv run python scripts/property_matrices.py \\
        --cross-eval-pkl  results/cross-eval-counterfact-llama31_cross_eval_results.pkl \\
        --hparams-dir     hparams/llama31 \\
        --sweep-dir       results/llama31-sweep-counterfact/matrices_hstacked \\
        --output-dir      results/llama31-sweep-counterfact/property_matrices \\
        --threshold       0.1
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ── helpers ──────────────────────────────────────────────────────────────────

def load_cross_eval(pkl_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (faithfulness_matrix, relation_names) from a cross-eval pickle.

    faithfulness_matrix[i, j] = faithfulness of relation i's decoder on relation j's samples.
    """
    with open(pkl_path, "rb") as f:
        results: dict[str, dict[str, float]] = pickle.load(f)

    relation_names = sorted(results.keys())
    n = len(relation_names)
    mat = np.zeros((n, n))
    for i, r_const in enumerate(relation_names):
        for j, r_test in enumerate(relation_names):
            mat[i, j] = results.get(r_const, {}).get(r_test, 0.0)
    return mat, relation_names


def symmetrize(mat: np.ndarray) -> np.ndarray:
    """Average with transpose to get a symmetric similarity matrix."""
    return (mat + mat.T) / 2


def cluster_relations(
    sim_matrix: np.ndarray,
    relation_names: list[str],
    threshold: float,
) -> dict[int, list[str]]:
    """Hierarchical clustering on the similarity matrix.

    threshold: cut the dendrogram at this distance (1 - similarity).
    Returns {cluster_id: [relation_name, ...]}.
    """
    # Convert similarity to distance; clip to avoid tiny negatives from float noise
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, None)
    np.fill_diagonal(dist_matrix, 0.0)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    clusters: dict[int, list[str]] = defaultdict(list)
    for rel, label in zip(relation_names, labels):
        clusters[int(label)].append(rel)
    return dict(clusters)


def load_best_matrix(
    rel_name: str,
    hparams_dir: Path,
    sweep_dir: Path,
) -> np.ndarray | None:
    hp_file = hparams_dir / f"{rel_name.replace(' ', '_')}.json"
    if not hp_file.exists():
        print(f"  [skip] no hparams for '{rel_name}'")
        return None

    best_layer = json.loads(hp_file.read_text()).get("h_layer")
    if best_layer is None:
        print(f"  [skip] no h_layer in hparams for '{rel_name}'")
        return None

    mat_file = sweep_dir / f"{rel_name.replace(' ', '_')}_layer={best_layer}.npy"
    if not mat_file.exists():
        print(f"  [skip] matrix not found: {mat_file.name}")
        return None

    return np.load(mat_file)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-eval-pkl",  required=True,
                        help="Path to *_cross_eval_results.pkl from cross_eval.py")
    parser.add_argument("--hparams-dir",     required=True,
                        help="Path to hparams/llama (NOT hparams — the model subdir itself)")
    parser.add_argument("--sweep-dir",       required=True,
                        help="Path to results/.../matrices_hstacked")
    parser.add_argument("--output-dir",      required=True,
                        help="Where to save property matrices")
    parser.add_argument("--threshold",       type=float, default=0.1,
                        help="Dendrogram cut distance (1 - similarity). "
                             "Lower = tighter clusters. Default 0.1")
    args = parser.parse_args()

    hparams_dir = Path(args.hparams_dir)
    sweep_dir   = Path(args.sweep_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cross-eval results
    print(f"Loading cross-eval results from {args.cross_eval_pkl}")
    faith_mat, rel_names = load_cross_eval(Path(args.cross_eval_pkl))
    print(f"  {len(rel_names)} relations, faithfulness matrix shape {faith_mat.shape}")

    # 2. Symmetrise and cluster
    sim = symmetrize(faith_mat)
    clusters = cluster_relations(sim, rel_names, threshold=args.threshold)

    print(f"\nDiscovered {len(clusters)} property groups "
          f"(threshold={args.threshold}, method=average linkage):\n")
    for cid, rels in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"  cluster {cid:2d} ({len(rels):2d} relations): {sorted(rels)}")

    # 3. Save cluster assignments
    assignments = {rel: cid for cid, rels in clusters.items() for rel in rels}
    assign_path = output_dir / "cluster_assignments.json"
    with open(assign_path, "w") as f:
        json.dump(
            {str(cid): sorted(rels) for cid, rels in sorted(clusters.items())},
            f, indent=2
        )
    print(f"\nCluster assignments saved to {assign_path}")

    # 4. Average matrices per cluster → property matrices
    print("\nComputing property matrices...")
    property_matrices: dict[int, np.ndarray] = {}

    for cid, rels in sorted(clusters.items()):
        matrices = []
        for rel in rels:
            mat = load_best_matrix(rel, hparams_dir, sweep_dir)
            if mat is not None:
                matrices.append(mat)

        if not matrices:
            print(f"  cluster {cid}: no matrices available, skipping")
            continue

        avg = np.mean(matrices, axis=0)
        property_matrices[cid] = avg

        out_path = output_dir / f"property_cluster_{cid}.npy"
        np.save(out_path, avg)

        W = avg[:, :-1]
        b = avg[:, -1]
        print(f"  cluster {cid:2d}: averaged {len(matrices)}/{len(rels)} matrices  "
              f"shape={avg.shape}  |W|={np.linalg.norm(W):.2f}  |b|={np.linalg.norm(b):.2f}  "
              f"→ {out_path.name}")

    # 5. Also save a stacked matrix [n_properties, d*(d+1)] for downstream use
    if property_matrices:
        stacked = np.stack(list(property_matrices.values()))  # [P, d, d+1]
        np.save(output_dir / "all_property_matrices.npy", stacked)
        print(f"\nAll property matrices stacked: shape {stacked.shape} "
              f"→ {output_dir}/all_property_matrices.npy")

    print(f"\nDone. {len(property_matrices)} property matrices saved to {output_dir}/")


if __name__ == "__main__":
    main()
