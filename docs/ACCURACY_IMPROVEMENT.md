# Accuracy improvement protocol

The Week 6 data, original code and results remain unchanged. This experiment
uses a separate deterministic, balanced development set of 32 cases and a
project-unseen holdout of 80 cases. Both are from the pinned AVeriTeC train
partition; prior experiments used its dev partition. Text and fact-checking
article overlaps are excluded across all three sets. See
[data provenance and license](../data/improvement/README.md).

## Development candidates

| Arm | Inputs received by the model | Method |
| --- | --- | --- |
| legacy_direct | Original statement only | Frozen Week 6 whole-sample prompt |
| legacy_dual | Statement and same-model analysis | Frozen two-round Phase 2 loop, then frozen sample judge |
| neutral | Statement only | Short TRUE/FALSE/UNKNOWN task, explicitly distinguishing missing evidence from falsity |
| binary | Statement only | Best-guess TRUE/FALSE task with two generic balanced examples |
| structured | Statement only, then source-derived extraction | Literal quotes and unverified propositions/context dependencies kept separate; binary final judge always receives full original statement |
| evidence | Statement plus supplied QA and historical metadata | Separate evidence-conditioned diagnosis, not automatic retrieval |

The binary prompt changes abstention policy. If the model nevertheless emits
UNKNOWN, it is recorded as an abstention, never converted to FALSE. Malformed
labels and truncated decisions remain invalid. Both the three-class and binary
prompt use the same explicit canonical mapping; label tokens are independently
realigned to the actual raw TRUE/FALSE/UNKNOWN output before averaging raw
log-probabilities. Scores are not truth probabilities and are not comparable
calibration scales across different label wordings.

The structured arm preserves source offsets only for literal text. A
reconstructed proposition is never presented as a literal quote. Heuristic
flags identify potential missing complements, pronouns and qualifier changes;
they neither prove atomicity nor decide truth. Original source context remains
available even if parsing or extraction fails. There is still no independent
atomic gold for the formal news samples.

The evidence condition receives curator-provided benchmark QA, with explicit
verdict phrases and the original fact-check article excluded by fixed rules.
URLs are retained for provenance but omitted from prompts. It also receives
claim date, speaker and country metadata. Its scores must be reported separately
from closed-book results; differences cannot isolate QA from metadata or prompt
effects, and cannot measure end-to-end web retrieval performance.

## Model, selection and holdout

All generation candidates use Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16/eager attention, greedy
single-beam inference and the original deterministic settings. Generation caps
are 16 for final decisions, 256 for structured extraction and 64 for the unused
optional reasoned-review arm. Model and large upstream data/cache files reside
outside the repository on E:.

Select the new closed-book prompt using development accuracy, then lower FPR,
then lower mean runtime, then arm name as deterministic tie breakers. Record
the completed development run and metrics SHA256 in the selection artifact.
Before holdout execution, freeze code, data, selected arm, model settings and
selection hashes. Run both legacy methods, the selected new method and the
separate evidence condition on the identical 80 holdout statements. Never
choose prompts or thresholds using holdout outcomes.

The initial incomplete development pilot and runtime-driven changes are
preserved in [development history](../reports/ACCURACY_DEVELOPMENT_HISTORY.md).
Development evidence is not an independent test result. The holdout is unseen
by this project, not guaranteed unseen in pretraining. Article/text deduplication
does not prove event-level independence. Original benchmark labels and QA are
not independently reverified facts.

## Verification and use

`scripts/audit_accuracy.py` checks exact permitted model inputs, original-source
retention, literal extraction parsing, label mapping, EOS, per-role settings,
raw-score means, selected-method ranking, chronological freezing and all code/
input hashes. With `--cache-dir`, it also replays every stored chat prompt and
output with the pinned CPU tokenizer. Auditing is posthoc for development;
its source/dependency hashes can be bound into the holdout freeze separately.

`scripts/check_accuracy.py --text <statement> --output <new JSON>` loads the
development-selected closed-book method. `--method evidence --evidence-file
<packet.json>` is a distinct user-supplied evidence mode; it does not search the
web. Both retain raw outputs and reject overwriting an existing result.

Report whole-sample accuracy, strict FNR, abstention-inclusive missed-error rate,
FPR, invalid/abstention counts and computational cost separately. Increasing
binary coverage alone does not demonstrate better factual reasoning. A future
change chosen after reviewing this holdout needs another evaluation set.

## Frozen candidate-score threshold condition

Before any holdout outcomes were used, a separate rule was fitted on the same
32 development cases: compare the summed teacher-forced log-likelihood of the
complete `TRUE + EOS` and `FALSE + EOS` candidates. Predict SUPPORTED when
TRUE minus FALSE is at least the frozen development threshold
`-9.106303319749713`; otherwise predict REFUTED. Candidate sums are not length
normalized and are not truth probabilities. This condition makes two forward
calls per sample, rather than generating a label. A zero-margin reference was
also declared before holdout execution and reuses the same saved scores.
The threshold and model contract are bound to the development artifact; neither
is refitted from the holdout. Full results, including the false-negative versus
false-positive tradeoff, are in [the final report](../reports/ACCURACY_RESULTS.md).
