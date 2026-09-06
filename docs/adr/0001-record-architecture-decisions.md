# 0001. Write decisions down as ADRs

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

This is a six-year-old graduation project. We are bringing it back to life and moving it from a stack
that is no longer supported (ROS Kinetic / Python 2) to one that is (ROS 2 / Python 3). Along the way
we make several big decisions that are hard to undo. The reasons behind the early choices were easy
to lose. We need a simple, low-effort way to write down *why* each decision was made, so that later
work (and later us) can look it up.

## Options

- **ADRs** (Architecture Decision Records) kept next to the code.
- One long design document or wiki page that keeps growing.
- Nothing formal. Rely on commit messages and memory.

## Decision

We keep short **MADR-style ADRs** under `docs/adr/`, one file per decision. They are numbered and
never rewritten to say something different (a new ADR supersedes an old one instead). See
[`README.md`](README.md) for the rules.

## Consequences

- **Good:** each decision keeps its context. The migration has a paper trail. Anyone new can see the
  "why", not just the "what".
- **Cost:** it takes a little discipline to write one for every real decision.
- **Neutral:** the ADRs live with the code and are versioned with it.
