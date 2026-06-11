# FYP Property-Level Residue Support

This directory documents how this repository supports the FYP repository:

`Revisiting the Locate-and-Edit Pipeline for Knowledge Editing in Large Language Models`

The AlphaEdit thesis repository uses `structrels` only as a supporting relation-structure analysis repo. It is not imported at runtime by `experiments/evaluate.py`; instead, the validated relation clusters are mirrored into `experiments/property_groups.py` in the main repo.

## Role In The FYP

The property-level residue experiment needs a way to decide which CounterFact relations are semantically close enough to provide negative subjects. `structrels` supplies that evidence by cross-evaluating relation decoders and visualising cross-faithfulness among relations.

The four property groups used by the thesis are:

- `COUNTRY`: `P17`, `P27`, `P495`
- `LANGUAGE`: `P37`, `P103`, `P364`, `P407`, `P1412`
- `OWNERSHIP`: `P127`, `P178`
- `GENRE_SPORT`: `P136`, `P641`

These groups match `scripts/plot_property_group_heatmaps.py` here and `experiments/property_groups.py` in the AlphaEdit thesis repo.

## Relevant Files

- `scripts/convert_counterfact.py` converts CounterFact records into relation JSON files for structrels sweeps.
- `structrels/cross_eval.py` computes relation-decoder cross-faithfulness.
- `scripts/plot_property_group_heatmaps.py` plots and records the manually validated property groups.
- `scripts/property_matrices.py` clusters a cross-faithfulness matrix and averages decoder matrices per discovered property cluster.
- `results_gpt2xl/cross-eval-counterfact-gpt2xl_cross_eval_results.pkl` is the recovered GPT-2 XL CounterFact cross-eval result used for the property-group inspection.
- `results_gpt2xl/property_group_heatmaps/` contains the recovered heatmaps for the groups above.

## Connection To AlphaEdit

The main repository consumes the relation grouping as follows:

- `experiments/property_groups.py` stores the selected relation-to-property mapping.
- `AlphaEdit/AlphaEdit_main_property_keykl.py` uses that mapping to sample sibling-relation negative subjects.
- `AlphaEdit/compute_z_property_keykl.py` applies key-scaled negative KL terms to preserve those property-level neighbours while computing the edit residual.

This makes `structrels` a provenance and analysis submodule for the FYP's property-level residue factor, not a runtime dependency of the edited model experiments.

## Reproduction Sketch

```bash
uv sync

uv run python scripts/convert_counterfact.py \
  --input /path/to/multi_counterfact.json \
  --output data/counterfact

bash scripts/sweep_example.sh
bash scripts/cross_eval_example.sh

uv run python scripts/plot_property_group_heatmaps.py \
  --cross-eval-pkl results_gpt2xl/cross-eval-counterfact-gpt2xl_cross_eval_results.pkl \
  --output-dir results_gpt2xl/property_group_heatmaps \
  --result-prefix cross-eval-counterfact-gpt2xl
```

For matrix averaging experiments:

```bash
uv run python scripts/property_matrices.py \
  --cross-eval-pkl results_gpt2xl/cross-eval-counterfact-gpt2xl_cross_eval_results.pkl \
  --hparams-dir hparams/gpt2xl \
  --sweep-dir results_gpt2xl/matrices_hstacked \
  --output-dir results_gpt2xl/property_matrices
```
