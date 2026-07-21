# AGENTS.md — talktomeclaude

Stack: Python >=3.11. Env/deps: `uv`, project venv at `.venv`. CLI framework: `click`.
Source layout: `src/talktomeclaude/` (package); console script `talktomeclaude` ->
`talktomeclaude.cli:main`. Fixtures: `tests/fixtures/`.

Test command: `bash .kiln/law/check.sh` (run from project root) — exit 0 is the bar. Do not
edit anything under `.kiln/` — it is Kiln's own control plane and evidence store, never
project source.
