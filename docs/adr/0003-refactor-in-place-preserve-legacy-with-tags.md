# 0003. Refactor in place; preserve the legacy stack via SemVer tags

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

The ROS 2 / Python 3 migration (ADR 0002) is effectively a rewrite. We need the
legacy ROS 1 work to stay referenceable, without carrying its EOL code forward as
dead weight, and without fragmenting the project across repositories.

## Considered options

- **Refactor in place** on the existing repo; mark the legacy stack with tags.
- A **new repository** for the v1.x line; freeze/archive the old one.
- A **`v2/` subdirectory** in the same repo, old code left beside new.

## Decision

We will **refactor in place** in this repository. The legacy ROS 1 / Python 2
stack is preserved as annotated SemVer tags — **`v0.1.0`** (baseline),
**`v0.2.0`** (fixed & reproducible), **`v0.3.0`** (enhanced, final ROS 1 line) —
so any old state can be checked out at any time. New work continues on `master`
toward **`v1.0.0`** (the ROS 2 line). No new repo, no parallel `v2/` tree.

## Consequences

- **Positive:** single source of truth and history; the old work is one `git
  checkout v0.3.0` away; `master` becomes clean, modern, and not littered with
  dead ROS 1 code.
- **Negative / trade-offs:** `master` will churn heavily during the rewrite;
  cross-referencing old vs new means jumping between a tag and `HEAD` rather than
  two live trees.
- **Neutral:** the tags double as release points if we ever publish the legacy work.
