# AVeriTeC attribution and data license

Source: Michael Sejr Schlichtkrull, Zhijiang Guo, and Andreas Vlachos (2023), **AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web**.

- Paper: https://openreview.net/forum?id=fKzSz0oyaI
- Official dataset: https://fever.ai/dataset/averitec.html
- Author repository: https://github.com/MichSchli/AVeriTeC
- Pinned revision: `7c62d1ec8df3fb560d6efe2b85fa191135636f81`, `data/dev.json`
- Data license: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (**CC BY-NC 4.0**).

This adaptation preserves the 60 previously selected Phase1 Refuted records and adds 50 selected Supported controls from the same pinned dev file. It adds project identifiers, normalized dates, provenance hashes, a standardized envelope and deterministic input ordering. Original claim text, original labels, justifications, QA answers and metadata are preserved. Selection and exclusions are recorded in `curation.json`; this project does not claim independent human re-verification or authorship of the upstream annotations.

The dataset and its source-derived content retain the upstream CC BY-NC 4.0 terms. A software license elsewhere in this repository does not replace that data license. URLs and quoted evidence may point to third-party materials with their own rights. No model weights or downloaded source cache are included here.
