# 0001. Record architecture decisions in ADRs

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

This is a six-year-old graduation project being revived and refactored from an
end-of-life stack (ROS Kinetic / Python 2) to a maintained one (ROS 2 / Python 3).
Several significant, hard-to-reverse decisions are being made, and the original
rationale for early choices was easy to lose. We need a durable, low-ceremony way
to capture *why* decisions were made so future work (and future us) has the
back-history.

## Considered options

- **ADRs** (Architecture Decision Records) committed alongside the code.
- A single growing design doc / wiki page.
- Nothing formal — rely on commit messages and memory.

## Decision

We will keep short **MADR-style ADRs** under `docs/adr/`, one file per decision,
numbered and immutable (superseded, never rewritten). See
[`README.md`](README.md) for conventions.

## Consequences

- **Positive:** decisions carry their context; the migration has a paper trail;
  reviewers/newcomers can see the "why", not just the "what".
- **Negative / trade-offs:** a little discipline needed to add a record per real
  decision.
- **Neutral:** ADRs live with the code and are versioned with it.
