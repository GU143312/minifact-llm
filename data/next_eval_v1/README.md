# Prospective news candidates and blinded annotation packet

**Historical initial selection, superseded by [v2](../next_eval_v2/README.md) before any news inference or human semantic annotation.** Retained because the recorded development freeze binds this manifest; the development model inputs themselves are the existing 32 exposed cases. Do not annotate this version for the next formal evaluation.

200 candidate claims (100 Supported / 100 Refuted), 60 fixed semantic-review cases, and 30 separate fictional boundary cases. **Event review and two-human semantic annotation are pending. This is not a completed gold dataset or a model result.**

Source and license: AVeriTeC, Schlichtkrull, Guo and Vlachos (2023), pinned train revision `7c62d1ec8df3fb560d6efe2b85fa191135636f81`; CC BY-NC 4.0. Attribution and source details are in [the upstream-data notice](../improvement/NOTICE.md). Full upstream bytes remain outside the repo. Claims and inherited whole-sample labels are preserved; project IDs, deterministic subsets and filtered QA projections are added. Fictional boundary fixtures are project-authored.

`dataset.json` contains inherited benchmark gold; `inputs.json` contains only id/text/context/evidence. `boundary_inputs.json` is separately projected. `manifest.json` binds immutable data and historical exclusions. The four reviewer files are editable templates; their completed hashes are bound only at formal freeze.

Read [the execution protocol](../../docs/NEXT_EVALUATION.md) and [the human annotation guide](../../docs/NEXT_EVALUATION_ANNOTATION.md). Keep reviewers' independent files separate until adjudication. Formal news inference refuses pending review. Do not copy private shared documents or personal reviewer identities into the public repository.
