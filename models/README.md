# Models

The three deployed models. Each folder loads with Hugging Face Transformers
(`config.json`, `model.safetensors`, tokenizer files); the weights are stored with
Git LFS.

| Folder | Agent | Extra files |
|---|---|---|
| `scibert-soft` | 1 · discipline | `conformal.json` (prediction sets, calibrated on 2025+ papers), `section_weights.json` (whole-document pooling), `temperature.json` (1.0: temperature scaling was rejected), `thresholds.json`, `added_ambiguous_ids.json` (the soft-label training papers) |
| `field-scibert-v2` | 2 · field | `conformal.json` (field sets at the 90% level) |
| `methodology-facets-v2` | 3 · research design | `thresholds.json` (per-facet thresholds and the no-evidence fallback), `reference_store.npz` (14,209 training papers, used to show the 3 most similar papers) |

All three are fine-tuned from `allenai/scibert_scivocab_uncased` with the same recipe:
learning rate 2e-5, inputs of up to 256 tokens, effective batch 32, 16-bit precision;
3 epochs for Agents 1 and 2, 4 for Agent 3. `training_args.bin` holds the exact
settings.

`val_predictions.npz`, `test_predictions.npz` and `temporal_predictions.npz` (or the
`*_facet_predictions.npz` files for Agent 3) are each model's saved outputs, which the
evaluation scripts read so that results reproduce without re-running the model.
