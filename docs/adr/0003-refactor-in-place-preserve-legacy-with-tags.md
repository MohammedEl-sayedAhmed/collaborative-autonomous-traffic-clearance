# 0003. Rebuild in this repo; keep the old stack at version tags

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

The move to ROS 2 / Python 3 (ADR 0002) is in practice a rewrite. We want the old ROS 1 work to stay
easy to find and read, without carrying its unsupported code forward as dead weight, and without
splitting the project across several repositories.

## Options

- **Rebuild in place** in this repo, and mark the old stack with git tags.
- A **new repository** for the new line, and freeze the old one.
- A **`v2/` folder** in the same repo, with the old code left next to the new.

## Decision

We **rebuild in place**, in this repository. The old ROS 1 / Python 2 stack is kept as annotated
version tags: **`v0.1.0`** (as it was), **`v0.2.0`** (fixed and reproducible), **`v0.3.0`** (improved;
the last ROS 1 version). Any old state can be checked out at any time. New work continues on `master`
toward **`v1.0.0`** (the ROS 2 line). No new repo, no `v2/` folder.

## Consequences

- **Good:** one place for the code and its history. The old work is one `git checkout v0.3.0` away.
  `master` becomes clean and modern, with no dead ROS 1 code lying around.
- **Cost:** `master` changes a lot during the rewrite. Comparing old with new means switching between
  a tag and `HEAD` instead of looking at two folders side by side.
- **Neutral:** the tags can also serve as release points if we ever publish the old work.
