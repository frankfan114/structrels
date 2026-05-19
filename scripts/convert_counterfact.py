#!/usr/bin/env python3
"""Convert CounterFact dataset to structrels relation JSON format.

Usage:
    python scripts/convert_counterfact.py \
        --input /path/to/counterfact.json \
        --output data/counterfact

Each CounterFact relation_id becomes one JSON file.
Samples use (subject, target_true) pairs — the original ground-truth fact
the model already knows, not the edited target_new.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


# Human-readable names for Wikidata PIDs present in CounterFact
RELATION_NAMES = {
    "P17":   "country",
    "P19":   "place of birth",
    "P20":   "place of death",
    "P27":   "country of citizenship",
    "P30":   "continent",
    "P36":   "capital",
    "P37":   "official language",
    "P39":   "position held",
    "P101":  "field of work",
    "P103":  "native language",
    "P106":  "occupation",
    "P108":  "employer",
    "P127":  "owned by",
    "P131":  "located in administrative entity",
    "P136":  "genre",
    "P138":  "named after",
    "P140":  "religion or worldview",
    "P159":  "headquarters location",
    "P176":  "manufacturer",
    "P178":  "developer",
    "P190":  "twinned administrative body",
    "P264":  "record label",
    "P276":  "location",
    "P364":  "original language of film or TV show",
    "P407":  "language of work or name",
    "P413":  "position played on team",
    "P449":  "original network",
    "P463":  "member of",
    "P495":  "country of origin",
    "P641":  "sport",
    "P740":  "location of formation",
    "P937":  "work location",
    "P1303": "instrument",
    "P1412": "languages spoken or written",
}

# Semantic metadata per relation
RELATION_PROPERTIES = {
    "P17":   {"domain_name": "location",     "range_name": "country",      "symmetric": False, "disambiguating": False},
    "P19":   {"domain_name": "person",       "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P20":   {"domain_name": "person",       "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P27":   {"domain_name": "person",       "range_name": "country",      "symmetric": False, "disambiguating": False},
    "P30":   {"domain_name": "entity",       "range_name": "continent",    "symmetric": False, "disambiguating": False},
    "P36":   {"domain_name": "country",      "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P37":   {"domain_name": "country",      "range_name": "language",     "symmetric": False, "disambiguating": False},
    "P39":   {"domain_name": "person",       "range_name": "position",     "symmetric": False, "disambiguating": False},
    "P101":  {"domain_name": "person",       "range_name": "field",        "symmetric": False, "disambiguating": False},
    "P103":  {"domain_name": "person",       "range_name": "language",     "symmetric": False, "disambiguating": False},
    "P106":  {"domain_name": "person",       "range_name": "occupation",   "symmetric": False, "disambiguating": False},
    "P108":  {"domain_name": "person",       "range_name": "organization", "symmetric": False, "disambiguating": False},
    "P127":  {"domain_name": "entity",       "range_name": "organization", "symmetric": False, "disambiguating": False},
    "P131":  {"domain_name": "location",     "range_name": "location",     "symmetric": False, "disambiguating": False},
    "P136":  {"domain_name": "artist",       "range_name": "genre",        "symmetric": False, "disambiguating": False},
    "P138":  {"domain_name": "entity",       "range_name": "person",       "symmetric": False, "disambiguating": False},
    "P140":  {"domain_name": "person",       "range_name": "religion",     "symmetric": False, "disambiguating": False},
    "P159":  {"domain_name": "organization", "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P176":  {"domain_name": "product",      "range_name": "company",      "symmetric": False, "disambiguating": False},
    "P178":  {"domain_name": "software",     "range_name": "company",      "symmetric": False, "disambiguating": False},
    "P190":  {"domain_name": "city",         "range_name": "city",         "symmetric": True,  "disambiguating": False},
    "P264":  {"domain_name": "artist",       "range_name": "label",        "symmetric": False, "disambiguating": False},
    "P276":  {"domain_name": "entity",       "range_name": "location",     "symmetric": False, "disambiguating": False},
    "P364":  {"domain_name": "work",         "range_name": "language",     "symmetric": False, "disambiguating": False},
    "P407":  {"domain_name": "work",         "range_name": "language",     "symmetric": False, "disambiguating": False},
    "P413":  {"domain_name": "athlete",      "range_name": "position",     "symmetric": False, "disambiguating": False},
    "P449":  {"domain_name": "work",         "range_name": "network",      "symmetric": False, "disambiguating": False},
    "P463":  {"domain_name": "person",       "range_name": "organization", "symmetric": False, "disambiguating": False},
    "P495":  {"domain_name": "work",         "range_name": "country",      "symmetric": False, "disambiguating": False},
    "P641":  {"domain_name": "person",       "range_name": "sport",        "symmetric": False, "disambiguating": False},
    "P740":  {"domain_name": "group",        "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P937":  {"domain_name": "person",       "range_name": "city",         "symmetric": False, "disambiguating": False},
    "P1303": {"domain_name": "person",       "range_name": "instrument",   "symmetric": False, "disambiguating": False},
    "P1412": {"domain_name": "person",       "range_name": "language",     "symmetric": False, "disambiguating": False},
}


def make_zs_template(prompt: str) -> str:
    """Derive a zero-shot template from a fill-in-the-blank prompt.

    Turns "The capital of {} is" into "Q: What is the capital of {}? A:".
    The subject placeholder {} is preserved so structrels can substitute it.
    """
    stripped = prompt.rstrip()
    # Drop trailing "is" / "was" / verb fragments to form a question body
    for suffix in (" is", " was", " are", " were", " has", " have"):
        if stripped.lower().endswith(suffix):
            body = stripped[: -len(suffix)].strip()
            return f"Q: {body}? A:"
    return f"Q: {stripped}? A:"


def convert(input_path: str, output_dir: str, min_samples: int = 10) -> None:
    with open(input_path) as f:
        records = json.load(f)

    # Group by relation_id; collect unique prompts and deduplicate samples by subject
    by_relation: dict[str, dict] = defaultdict(lambda: {"prompts": [], "samples": {}})

    for rec in records:
        rr = rec["requested_rewrite"]
        rid = rr["relation_id"]
        prompt = rr["prompt"]
        subject = rr["subject"]
        obj = rr["target_true"]["str"]

        entry = by_relation[rid]
        if prompt not in entry["prompts"]:
            entry["prompts"].append(prompt)
        # First occurrence wins; target_true is stable across edits for same subject
        if subject not in entry["samples"]:
            entry["samples"][subject] = obj

    os.makedirs(output_dir, exist_ok=True)
    relation_names = []
    skipped = []

    for rid, data in sorted(by_relation.items()):
        samples = [
            {"subject": subj, "object": obj}
            for subj, obj in data["samples"].items()
        ]

        if len(samples) < min_samples:
            skipped.append((rid, len(samples)))
            continue

        name = RELATION_NAMES.get(rid, rid.lower())
        props = RELATION_PROPERTIES.get(rid, {
            "domain_name": "entity",
            "range_name": "entity",
            "symmetric": False,
            "disambiguating": False,
        })

        prompt_templates = data["prompts"]
        prompt_templates_zs = [make_zs_template(p) for p in prompt_templates]

        relation_dict = {
            "name": name,
            "prompt_templates": prompt_templates,
            "prompt_templates_zs": prompt_templates_zs,
            "properties": {
                "relation_type": "factual",
                "domain_name": props["domain_name"],
                "range_name": props["range_name"],
                "symmetric": props["symmetric"],
                "disambiguating": props["disambiguating"],
            },
            "samples": samples,
        }

        filename = name.replace(" ", "_") + ".json"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            json.dump(relation_dict, f, indent=2, ensure_ascii=False)

        print(f"  {rid:6s}  {filename:<50s}  {len(samples):4d} samples  {len(prompt_templates)} templates")
        relation_names.append(name)

    # Write relations.txt listing all relation names (used by tensor network scripts)
    relations_file = os.path.join(output_dir, "relations.txt")
    with open(relations_file, "w") as f:
        for name in sorted(relation_names):
            f.write(name + "\n")

    print(f"\nWrote {len(relation_names)} relations → {output_dir}/")
    if skipped:
        print(f"Skipped {len(skipped)} relations with < {min_samples} samples: "
              + ", ".join(f"{r}({n})" for r, n in skipped))
    print(f"relations.txt: {relations_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to counterfact.json")
    parser.add_argument("--output", required=True, help="Output directory for relation JSON files")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Skip relations with fewer than this many samples (default: 10)")
    args = parser.parse_args()
    convert(args.input, args.output, args.min_samples)


if __name__ == "__main__":
    main()
