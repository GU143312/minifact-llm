# AVeriTeC attribution and license

These selected records are adapted from **AVeriTeC** by **Michael Sejr Schlichtkrull, Zhijiang Guo, and Andreas Vlachos (2023)**.

- Paper: [AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web](https://arxiv.org/abs/2305.13117).
- Author repository: [MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC).
- Source: `data/train.json` at revision `7c62d1ec8df3fb560d6efe2b85fa191135636f81`.
- Dataset license: [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Changes: deterministic structural selection and project development/holdout splitting; new IDs, standardized context, original-record hashes and input projections. Original claim text, labels, QA, justification and original metadata are preserved in the datasets. The separate supplied-evidence projection filters explicit verdict wording and original curator-article sources, then truncates answers using the fixed limits documented in `README.md` and `manifest.json`.

Inherited whole-claim labels are not independent human re-verification, present-day truth labels, or atomic-fact annotations. Public availability does not remove the noncommercial restriction. This notice concerns dataset content; it does not assign this license to unrelated project source code or distribute model weights.
