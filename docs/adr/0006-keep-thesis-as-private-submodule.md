# 0006. Keep the thesis in a private submodule

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

This repository will be made public, but the graduation thesis (LaTeX source and the compiled PDF)
must stay private. The thesis had been committed into this repo, so it also existed in the git history
and in the diffs of merged pull requests.

## Options

- **Private submodule:** move the thesis to its own private repo, and link it here as a submodule at
  `thesis/`.
- Keep it in this repo and make everything public (exposes the thesis).
- Keep the whole repo private.

## Decision

The thesis lives in a **separate private repository**, linked in as a git **submodule at
`thesis/`**. A public clone only gets the submodule pointer (a commit reference behind access
control), not the content. The compiled `main.pdf` is tracked in that private repo, and a check of the
thesis against the code lives there too.

## Consequences

- **Good:** the main repo can go public without exposing the thesis. The submodule pointer keeps the
  two in step when it is updated.
- **Cost:** you need access to build the thesis. **Any change to the thesis must also update the
  submodule pointer here.** Rewriting history alone does not remove the thesis from already-merged PR
  diffs on GitHub (accepted, since the repo was still private at that point).
- **Neutral:** `./run.sh thesis` builds the submodule when it is checked out.
