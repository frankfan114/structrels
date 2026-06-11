#!/usr/bin/env python3
"""Plot property-group cross-faithfulness heatmaps from cross_eval results.

This script does not run model inference. It only reads the pickle produced by
structrels/structrels/cross_eval.py and slices it into pre-defined property
groups.

Example:
    python scripts/plot_property_group_heatmaps.py \
        --cross-eval-pkl results_gpt2xl/cross-eval-counterfact-gpt2xl_cross_eval_results.pkl \
        --output-dir results_gpt2xl/property_group_heatmaps \
        --result-prefix cross-eval-counterfact-gpt2xl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PROPERTY_GROUPS: dict[str, list[str]] = {
    # cluster 9
    "COUNTRY": ["P17", "P27", "P495"],
    # cluster 8
    "LANGUAGE": ["P37", "P103", "P364", "P407", "P1412"],
    # cluster 2
    "OWNERSHIP": ["P127", "P178"],
    # cluster 1
    "GENRE_SPORT": ["P136", "P641"],
}


PID_TO_RELATION_NAME: dict[str, str] = {
    "P17": "country",
    "P27": "country of citizenship",
    "P495": "country of origin",
    "P37": "official language",
    "P103": "native language",
    "P364": "original language of film or TV show",
    "P407": "language of work or name",
    "P1412": "languages spoken or written",
    "P127": "owned by",
    "P178": "developer",
    "P136": "genre",
    "P641": "sport",
}


CrossFaithfulness = Mapping[str, Mapping[str, float]]


def load_cross_eval(path: Path) -> dict[str, dict[str, float]]:
    with path.open("rb") as f:
        results = pickle.load(f)
    return {outer: dict(inner) for outer, inner in results.items()}


def make_heatmap_data(
    results: CrossFaithfulness,
    decoder_relations: Sequence[str],
    evaluated_relations: Sequence[str],
) -> np.ndarray:
    data = np.full((len(decoder_relations), len(evaluated_relations)), np.nan)
    for i, decoder_relation in enumerate(decoder_relations):
        for j, evaluated_relation in enumerate(evaluated_relations):
            data[i, j] = results.get(decoder_relation, {}).get(evaluated_relation, np.nan)
    return data


def relation_labels(group: str) -> list[tuple[str, str]]:
    return [(pid, PID_TO_RELATION_NAME[pid]) for pid in PROPERTY_GROUPS[group]]


def available_group_relations(
    results: CrossFaithfulness,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    relation_names = set(results.keys())
    present: dict[str, list[tuple[str, str]]] = {}
    missing: dict[str, list[tuple[str, str]]] = {}

    for group in PROPERTY_GROUPS:
        group_present: list[tuple[str, str]] = []
        group_missing: list[tuple[str, str]] = []
        for pid, relation in relation_labels(group):
            if relation in relation_names:
                group_present.append((pid, relation))
            else:
                group_missing.append((pid, relation))
        present[group] = group_present
        missing[group] = group_missing

    return present, missing


def tick_labels(relations: Sequence[tuple[str, str]]) -> list[str]:
    return [pid for pid, _ in relations]


def format_axis(
    ax: plt.Axes,
    title: str,
    relations: Sequence[tuple[str, str]],
    tick_fontsize: int = 9,
    axis_fontsize: int = 10,
) -> None:
    labels = tick_labels(relations)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(labels, rotation=0, fontsize=tick_fontsize)
    ax.set_xlabel("Evaluated on relation", fontsize=axis_fontsize)
    ax.set_ylabel("Decoder approximated on relation", fontsize=axis_fontsize)


def plot_single_group(
    results: CrossFaithfulness,
    group: str,
    relations: Sequence[tuple[str, str]],
    output_prefix: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    relation_names = [relation for _, relation in relations]
    data = make_heatmap_data(results, relation_names, relation_names)

    side = max(4.5, 1.25 * len(relations) + 2.0)
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        data,
        cmap="plasma",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        xticklabels=tick_labels(relations),
        yticklabels=tick_labels(relations),
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        square=True,
        ax=ax,
    )
    format_axis(ax, group, relations)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(output_prefix.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_combined_groups(
    results: CrossFaithfulness,
    groups: Mapping[str, Sequence[tuple[str, str]]],
    output_prefix: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 17))
    flat_axes = axes.flatten()

    for ax, (group, relations) in zip(flat_axes, groups.items()):
        relation_names = [relation for _, relation in relations]
        data = make_heatmap_data(results, relation_names, relation_names)
        sns.heatmap(
            data,
            cmap="plasma",
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            xticklabels=tick_labels(relations),
            yticklabels=tick_labels(relations),
            cbar=False,
            square=True,
            ax=ax,
        )
        format_axis(
            ax,
            group,
            relations,
            tick_fontsize=8,
            axis_fontsize=9,
        )

    for ax in flat_axes[len(groups):]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.15,
        right=0.82,
        top=0.92,
        bottom=0.08,
        wspace=0.35,
        hspace=0.8,
    )
    mappable = flat_axes[0].collections[0]
    cbar_ax = fig.add_axes([0.89, 0.20, 0.02, 0.60])
    fig.colorbar(mappable, cax=cbar_ax)
    fig.suptitle("Property-group cross faithfulness", fontsize=18, y=0.98)

    for fmt in formats:
        fig.savefig(output_prefix.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def write_summary(
    path: Path,
    present: Mapping[str, Sequence[tuple[str, str]]],
    missing: Mapping[str, Sequence[tuple[str, str]]],
) -> None:
    payload = {
        group: {
            "present": [{"pid": pid, "relation": relation} for pid, relation in present[group]],
            "missing": [{"pid": pid, "relation": relation} for pid, relation in missing[group]],
        }
        for group in PROPERTY_GROUPS
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-eval-pkl", required=True, help="Path to *_cross_eval_results.pkl")
    parser.add_argument("--output-dir", required=True, help="Directory for generated heatmaps")
    parser.add_argument("--result-prefix", default="cross_eval", help="Filename prefix for generated figures")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_cross_eval(Path(args.cross_eval_pkl))
    present, missing = available_group_relations(results)
    groups_to_plot = {group: rels for group, rels in present.items() if rels}

    for group, relations in groups_to_plot.items():
        output_prefix = output_dir / f"{args.result_prefix}_{group.lower()}_cross_faithfulness"
        plot_single_group(results, group, relations, output_prefix, args.formats, args.dpi)
        if missing[group]:
            missing_text = ", ".join(f"{relation} ({pid})" for pid, relation in missing[group])
            print(f"[warn] {group}: missing from cross-eval results: {missing_text}")

    combined_prefix = output_dir / f"{args.result_prefix}_property_groups_cross_faithfulness"
    plot_combined_groups(results, groups_to_plot, combined_prefix, args.formats, args.dpi)

    summary_path = output_dir / f"{args.result_prefix}_property_groups_summary.json"
    write_summary(summary_path, present, missing)
    print(f"Saved {len(groups_to_plot)} property-group heatmaps to {output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
