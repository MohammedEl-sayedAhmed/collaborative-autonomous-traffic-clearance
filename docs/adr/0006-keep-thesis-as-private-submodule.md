# 0006. Keep the thesis as a private git submodule

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

The repository will be made public, but the graduation thesis (LaTeX source +
compiled PDF) must stay private. The thesis had been committed into this repo, so
it also existed in the git history and in merged pull-request diffs.

## Considered options

- **Private submodule:** move the thesis to a separate private repo, referenced
  here as a submodule at `thesis/`.
- Keep it in-repo and make the whole repo public (exposes the thesis).
- Keep the whole repo private.

## Decision

The thesis lives in a **separate private repository**, wired in as a git
**submodule at `thesis/`**. Public clones get only the submodule pointer (a
gated commit ref), not the content. The thesis's compiled `main.pdf` is tracked in
that private repo, and a thesis-vs-code cross-check lives there too.

## Consequences

- **Positive:** the main repo can go public without exposing the thesis; the
  submodule ref keeps the two in lock-step when bumped.
- **Negative / trade-offs:** contributors need access to build the thesis;
  **any thesis change must also bump the submodule ref here**; and note that
  history rewriting alone does not remove the thesis from already-merged PR diffs
  on the host (accepted, given the repo is still private pre-publication).
- **Neutral:** `./run.sh thesis` builds the submodule when it is checked out.
