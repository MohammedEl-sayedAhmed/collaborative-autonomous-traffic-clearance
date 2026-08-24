# Thesis — LaTeX source

Graduation-project thesis **"Autonomous Traffic Clearance for Emergency Vehicles:
A Cooperative Reinforcement Learning Approach"** (Zewail City × Valeo, 2020).

This is the verbatim Overleaf source. It compiles to **[`main.pdf`](main.pdf)**
(137 pages), which is **committed alongside the source** — every commit that
touches the thesis ships its rebuilt PDF.

> The thesis covers **two** simulators; only the **Gazebo/ROS** half has code in
> this repository (the SUMO half is not shipped here). For a chapter-by-chapter
> comparison of the thesis against the actual code, see
> **[../docs/THESIS_CROSSCHECK.md](../docs/THESIS_CROSSCHECK.md)**.

## Compile it

Everything runs in Docker — **nothing is installed on your host** — using the
same full TeX Live distribution Overleaf uses.

```bash
# from the repository root:
./run.sh thesis          # -> thesis/main.pdf   (first run pulls TeX Live, a few GB)
./run.sh thesis-clean    # remove build artifacts (keeps main.pdf)
```

Or invoke TeX Live directly (equivalent to what `run.sh` does):

```bash
cd thesis
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  latexmk -pdf -f -interaction=nonstopmode -file-line-error main.tex
```

Pin a specific TeX Live (e.g. to match Overleaf's TL2025) with
`TEXLIVE_IMAGE=texlive/texlive:TL2025-historic ./run.sh thesis`.

### Keeping `main.pdf` in sync

`main.pdf` is committed, so rebuild it whenever you change the source: run
`./run.sh thesis` and include the refreshed `main.pdf` in the same commit. To
automate that, enable the optional pre-commit hook
([`../tools/thesis/pre-commit`](../tools/thesis/pre-commit)) — it rebuilds and
stages the PDF whenever a commit touches the thesis (see the header of that file
for how to install it alongside any existing hooks).

### On Overleaf

Upload this folder (or push the repo and use Overleaf's GitHub sync). Set the
main document to `main.tex`; Overleaf uses `latexmk` + `biber` automatically and
produces the identical PDF.

## Toolchain

- **Engine:** `pdflatex` via `latexmk` (config in [`.latexmkrc`](.latexmkrc)).
- **Bibliography:** `biblatex` with the **biber** backend (`references.bib`).
- **Verified with:** `texlive/texlive:latest` (TeX Live 2026) — Overleaf-equivalent.

> **Why `-f` and a PDF check instead of the exit code?** TeX Live 2025/2026's
> `pdflatex` returns a non-zero exit code on the transient first-pass "undefined
> citation" warnings (before biber runs) and even on a clean compile that only
> emitted warnings. So [`../tools/thesis/build.sh`](../tools/thesis/build.sh)
> judges success the Overleaf way — *was `main.pdf` produced, with no real
> errors, no multiply-defined labels, and no unresolved references?* — rather
> than trusting the exit code.

## Layout

```
thesis/
├── main.tex                # document root (\input's everything below)
├── .latexmkrc              # latexmk configuration
├── titlepage.tex  acknowledgement.tex  abstract.tex
├── chapters/               # introduction, background, literatureReview,
│                           #   projectDesign, methodology,
│                           #   projectTestAndDiscussion, conclusion, appendix
├── references.bib          # bibliography (biber)
├── mendeley.bib            # secondary bib export (not \addbibresource'd)
├── images/                 # all figures
└── main.pdf                # compiled output (tracked)
```
