# Attribution and data license

This dataset is adapted from **AVeriTeC**, by **Michael Sejr Schlichtkrull, Zhijiang Guo, and Andreas Vlachos (2023)**, *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web*, NeurIPS Datasets and Benchmarks Track.

- Paper: https://openreview.net/forum?id=fKzSz0oyaI
- Author repository: https://github.com/MichSchli/AVeriTeC
- Source revision: `7c62d1ec8df3fb560d6efe2b85fa191135636f81`, file `data/dev.json`
- Upstream license statement: https://github.com/MichSchli/AVeriTeC/blob/7c62d1ec8df3fb560d6efe2b85fa191135636f81/README.md#license

The adapted data in `dataset.json` and `inputs.json` are distributed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**: https://creativecommons.org/licenses/by-nc/4.0/ . Retain attribution and this notice, indicate changes, and comply with the noncommercial restriction. No commercial or unrestricted-data license is claimed. Rights in linked third-party articles and media remain with their respective owners.

Changes by GU143312 / MiniFact-LLM, 2026-10-21: selected 60 original Refuted records, excluded unsuitable/duplicate records, added project IDs, normalized dates, source hashes and a standardized JSON envelope, and separated model inputs from gold annotations. Original claim text, justification, QA, and supplied provenance fields are preserved. Source labels apply to whole samples; no atomic-level labels have been inferred.

```bibtex
@inproceedings{schlichtkrull2023averitec,
  title={{AV}eri{T}e{C}: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author={Michael Sejr Schlichtkrull and Zhijiang Guo and Andreas Vlachos},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=fKzSz0oyaI}
}
```
