from __future__ import annotations


COMMIT_LABELS: tuple[str, ...] = (
    "feat",
    "fix",
    "refactor",
    "perf",
    "docs",
    "style",
    "test",
    "chore",
)

COMMIT_LABEL_SET = set(COMMIT_LABELS)

# Legacy labels are still stripped from input text so older prefixes do not
# leak unsupported taxonomy choices into prompts or heuristics.
LEGACY_COMMIT_LABELS: tuple[str, ...] = (*COMMIT_LABELS, "build", "ci", "revert")
