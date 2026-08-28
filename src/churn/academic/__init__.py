"""Phase 14A: the academic delivery package, and the honesty it is built to enforce.

This subpackage produces one thing — the machine-readable record of what the academic
delivery actually contains — and its whole design is shaped by a single risk.

**The risk.** An academic deliverable is judged by a rubric, and a rubric rewards
completeness. That creates constant pressure to write ``true`` where the truth is
``false``: to call a local ``nbconvert`` run a Colab execution, to paste a
plausible-looking share link, to drop a ``[[PENDING]]`` marker into a PDF and name it
final. Every one of those is cheap to do and hard to detect later.

**The response.** The record is built by :mod:`churn.academic.results` from what is
**observably present in the repository**, never from what the author intends to do.
Whether the notebook exists is answered by reading the filesystem. Whether it executes
is answered by an execution. Whether it ran on Colab cannot be answered from inside
this process at all — so that field is a constant ``False`` here, and only a human who
did the work may change it, in a commit that says so.

The record therefore has two kinds of field, and the distinction is deliberate:

* **derived** — computed from the repository as it is, and re-derived by ``--verify``;
* **externally attested** — held at their honest default (``False`` / ``null``) until
  a person performs an action this code cannot perform or observe.

A record whose external fields are all ``true`` after a run of this module would mean
the module is lying, not that the work is done.

Deterministic: no timestamp, no hostname, no run id, so ``--verify`` compares bytes.
"""

from __future__ import annotations

__all__ = ["results"]
