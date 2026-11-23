# Accuracy improvement development history

The original Week 6 inputs, code, results and metrics remain frozen.

2026-11-12: the first development protocol, `accuracy_dev_freeze_20261112.json`,
ran three complete development samples before it was interrupted. Raw output is
preserved in `accuracy_dev_run_20261112.json`; changed source files are retained
in `accuracy_development/v1_sources/`. Other frozen sources and data were not
modified. This is an incomplete development pilot, not the final evaluation.

The pilot exposed high runtime: roughly 100-token reviews took tens of seconds
per arm, while the direct decision took about one second. Version 2 keeps a
three-class neutral prompt, adds a forced binary decision candidate with two
generic balanced examples, and removes the extra review from the structured
and supplied-evidence conditions. Original statement context is always retained.
UNKNOWN, if emitted, is still recorded as abstention; it is never converted to
FALSE. Different abstention policies are explicitly compared, with FPR and
strict FNR retained alongside accuracy. The source extraction contract and all
32 development / 80 holdout data bytes remain unchanged.

These changes were made during development, after observing the three pilot
outputs and timing. They are not claimed to be preregistered before development.
No holdout predictions had been generated or inspected. Final selection uses
the completed version-2 development run, then the method is frozen before the
first holdout run. A future improvement after inspecting holdout outcomes would
require a new evaluation set.
