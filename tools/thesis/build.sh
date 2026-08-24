#!/usr/bin/env sh
# Compile the thesis to thesis/main.pdf.
#
# Runs INSIDE a TeX Live container (see `run.sh thesis` and
# .github/workflows/thesis.yml). latexmk + biber must be on PATH. The working
# directory does not matter — the script locates thesis/ relative to itself.
#
# Why we don't trust the exit code of pdflatex/latexmk:
#   TeX Live 2025/2026's pdflatex returns a non-zero exit code on the
#   transient first-pass "undefined citation" warnings (before biber runs) and
#   even on a perfectly clean compile that merely emitted warnings. Overleaf
#   has the same warnings yet still produces the PDF. So success here is judged
#   the Overleaf way: "was main.pdf produced, with no real errors, no
#   multiply-defined labels, and no unresolved references/citations left".
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
THESIS_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../../thesis" && pwd) || {
  echo "build.sh: cannot locate thesis/ from $SCRIPT_DIR" >&2; exit 2; }
cd "$THESIS_DIR" || exit 2
[ -f main.tex ] || { echo "build.sh: main.tex not found in $THESIS_DIR" >&2; exit 2; }

echo ">> Building thesis in $THESIS_DIR"
echo ">> latexmk -pdf (pdflatex + biber, multiple passes) ..."
# -f: keep going through the benign non-zero exit described above so biber runs
# and all passes complete. We validate the real outcome ourselves below.
latexmk -pdf -f -interaction=nonstopmode -file-line-error main.tex || true

status=0
LOG=main.log

if [ ! -s main.pdf ]; then
  echo "FAIL: main.pdf was not produced." >&2
  exit 1
fi

# Genuine fatal errors. The message text survives -file-line-error (only the
# leading "! " is replaced by "file:line:"), so we match on the text.
ERR_RE='LaTeX Error|Package [^ ]* Error|Undefined control sequence|Emergency stop|Fatal error|Runaway argument'
if [ -f "$LOG" ] && grep -qE "$ERR_RE" "$LOG"; then
  echo "FAIL: LaTeX reported errors:" >&2
  grep -nE "$ERR_RE" "$LOG" | head -20 >&2
  status=1
fi

if [ -f "$LOG" ] && grep -q "multiply defined" "$LOG"; then
  echo "FAIL: multiply-defined labels (fix duplicate \\label{}):" >&2
  grep "multiply defined" "$LOG" | head -20 >&2
  status=1
fi

if [ -f "$LOG" ] && grep -qE "Citation '.*' undefined|There were undefined references" "$LOG"; then
  echo "FAIL: unresolved references/citations remain (biber/bib problem):" >&2
  grep -nE "Citation '.*' undefined|Reference '.*' undefined|There were undefined references" "$LOG" | head -20 >&2
  status=1
fi

if [ ! -f main.bbl ]; then
  echo "WARN: main.bbl not present — biber may not have run." >&2
fi

if [ "$status" -eq 0 ]; then
  # The "Output written on main.pdf (N pages" line is wrapped in the log; join
  # lines before matching so the page count is found reliably.
  pages=$(tr -d '\n' < "$LOG" 2>/dev/null | grep -o 'Output written on main.pdf ([0-9]* pages' | tail -1 | grep -o '[0-9]* pages')
  bytes=$(wc -c < main.pdf | tr -d ' ')
  echo "OK: thesis/main.pdf built — ${pages:-unknown length}, ${bytes} bytes"
fi
exit "$status"
