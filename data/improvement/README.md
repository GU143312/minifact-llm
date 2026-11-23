# Accuracy improvement development and holdout data

This version separates **32 development samples (16 Supported / 16 Refuted)** from **80 project-unseen holdout samples (40 / 40)**. Both come from AVeriTeC's upstream **train** partition; the earlier 110-case experiment came from its dev partition. The new project split names do not rename the upstream partition. Neither new split was selected using model predictions.

| File | Purpose |
| --- | --- |
| `dev_dataset.json`, `holdout_dataset.json` | Whole-sample inherited gold, complete original upstream record, context and record hashes |
| `dev_inputs.json`, `holdout_inputs.json` | Closed-book runner input: exactly `id` and `text` |
| `dev_evidence_inputs.json`, `holdout_evidence_inputs.json` | Supplied-evidence diagnostic input: exactly `id`, `text`, `context`, `evidence` |
| `manifest.json` | Fixed source, filters, selected zero-based train indices, selection audit and six artifact hashes |
| `dataset.schema.json` | Schema for the two gold-data envelopes |
| `NOTICE.md` | Attribution, license and changes |

IDs such as `MFI-D-0001` and `MFI-H-0001` reveal the **project split**, not its label. IDs are for joining/checkpointing and must not be placed in model prompts. The runner should receive only the text for closed-book inference. Gold labels, justifications and the `upstream_record` object are only for offline scoring and audit.

## Fixed source and selection

Source: [author repository, fixed train.json](https://github.com/MichSchli/AVeriTeC/blob/7c62d1ec8df3fb560d6efe2b85fa191135636f81/data/train.json), revision `7c62d1ec8df3fb560d6efe2b85fa191135636f81`. Full source bytes: **10,184,813**, SHA256 `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`, Git blob SHA `0f190e115cf2ee23416e8a539c8d6ac043d7cc83`. The original source stays outside the repository on E: with model/cache files.

The source has 3,068 records: 849 Supported, 1,742 Refuted, 195 Conflicting Evidence/Cherrypicking, and 282 Not Enough Evidence. Selection:

1. Keep only Supported/Refuted and `required_reannotation == false`.
2. Apply the earlier Phase1 structural rules: 8–65 whitespace-separated words, no obvious image/video dependence, no unresolved opening pronoun, usable text/PDF evidence, justification and fact-checking article available. Exclude editorial `Fact Check` wording in the claim and invalid claim dates.
3. Require at least one answer surviving the evidence projection below. The resulting pool contains 462 Supported and 912 Refuted records. This quality filter also applies to the closed-book subsets, so results are conditional on having usable supplied QA.
4. Rank candidates using SHA256 of the fixed namespace, seed 42, source index and original claim. Select balanced development first, then balanced holdout. Globally forbid repeated normalized claim text or canonical fact-checking article across the old 110, development and holdout. All indices, counts and exclusion totals are recorded before model execution.
5. Assign split-only IDs by source index, then deterministically shuffle input order with a separate SHA namespace. Claim text and the original upstream record remain unchanged. JSON is UTF-8 with explicit LF bytes.

The old 110-case dataset is bound by its byte hash and is unchanged. Article/text deduplication does **not** guarantee exhaustive event/story independence. These samples are project-unseen, not guaranteed absent from model pretraining. They are structurally screened, not independently human re-verified; the original benchmark may contain omissions or label issues even when its reannotation flag is false.

## Supplied evidence is a separate diagnostic condition

Each evidence input has context keys `claim_date` (ISO date), `speaker`, and `location_ISO_code`; the last two may be null. Evidence entries have exactly `question`, `answer`, and `url`. The projection:

- accepts non-Unanswerable Web text/PDF answers with an HTTP(S) source;
- omits obvious explicit verdict wording in either the question or answer and omits answers sourced from the original fact-checking article;
- retains at most 8 entries, at most 900 Unicode characters per answer and 5,000 answer characters per sample, in upstream order;
- never reads the upstream `label` or `justification` to construct evidence inputs.

The complete, untruncated QA is retained in the gold dataset's original record. Filtering is conservative and lexical; it cannot prove that no implicit verdict remains. Some curator-authored QA may directly resolve the claim. This is **provided benchmark evidence**, not independent retrieval, and must not be described as end-to-end web fact-checking performance. Evidence-conditioned accuracy must be reported separately from closed-book accuracy.

URL paths can themselves contain verdict cues. **Keep URLs in provenance records but omit the URL strings from model prompts**, replacing them with stable `E1`, `E2`, … reference IDs. The model input must not include gold, curator justification, fact-checking article or article titles. The source dates/context are additional information in this diagnostic condition and must be disclosed when interpreting differences from text-only inference.

## Check and reproduce

```powershell
python scripts/build_improvement_data.py --check
python scripts/build_improvement_data.py --source <cache>/train.json --check
python -m unittest discover -s tests -p test_improvement_data.py -v
```

Offline `--check` verifies schema, original benchmark record hashes, label mapping, input isolation, evidence projection, balanced counts, deterministic order, selected source indices, six frozen artifact hashes, old-data identity and cross-set text/article/ID/index disjointness. With the pinned source it additionally rebuilds selection and every artifact byte. Remove `--check` only to deliberately rebuild this version; a frozen experiment must never be silently overwritten.

Development may guide prompts and selection. Holdout is scored only after the selected method is frozen. If a holdout run is later used for diagnosis, report that exposure and use a new independent evaluation set for future improvement claims. Whole-sample labels do not supply atomic fact, qualifier or span gold.
