# Contributing

Thank you for your interest in this project. It started as a one-week student
project at Polytech Nice Sophia, and bug reports, new analyses and pull
requests are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** or **suggest an improvement** with the
  [issue forms](https://github.com/BnRomain/himalayan-expedition-success/issues/new/choose).
- **Report a security vulnerability** privately, as described in the
  [security policy](SECURITY.md). Please do not open a public issue for it.
- **Open a pull request** for a fix, a test, a new feature or model, or documentation.

For a larger change, for example another model or a new set of predictors,
please open an issue first so that we can agree on the approach.

## Development setup

Requires Python 3.12.

```bash
git clone https://github.com/BnRomain/himalayan-expedition-success.git
cd himalayan-expedition-success
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m pytest -v                # runs the five scripts and checks their outputs
ruff check .                       # lint
ruff format .                      # format
```

To check the Markdown files like the CI does:

```bash
npx markdownlint-cli2
lychee --offline --include-fragments .
```

## Coding guidelines

The repository provides an [`.editorconfig`](.editorconfig) file: most editors
apply its indentation and whitespace settings automatically.

- The code must pass `ruff check` and be formatted with `ruff format` (a
  Black-compatible style), both configured in [`pyproject.toml`](pyproject.toml).
- The scripts are numbered and run in order. They locate the data and the
  figures from the repository root through `ROOT`, `DATA` and `FIGURES`: keep
  it that way so that they run from any working directory.
- Every decision (a variable kept or rejected, a feature created) is
  documented in a comment next to the code, with the number that supports it.
  Keep that habit: the comments are the reasoning of the report.
- The tests in `tests/` run each script in-process and check the printed facts
  and the files written. Update them when an output changes, and add a check
  for every new result. The CI fails if the coverage drops below 90 %.
- A new predictor must be known before the departure of the expedition. If in
  doubt, check how the variable is collected before using it.
- Figures go to `docs/figures/` and are committed, so that the report, the
  slides and the README always match the code. Rerun the scripts after a change
  that affects them, and update the numbers of the README and the report.
- Pin new dependencies to an exact version in `requirements.txt` (or
  `requirements-dev.txt` for development tools) so that Dependabot can track them.

## Pull request process

1. Create a branch from `main` with a descriptive name, for example
   `fix/ratio-division` or `docs/results-table`.
2. Keep commits focused, with a short summary in the imperative mood
   (for example "Add the altitude of the peak as a predictor").
3. Open a pull request against `main`, fill in the template and add a label
   (`bug`, `enhancement`, `documentation`...): labels sort the release notes.
4. The `main` branch is protected: a pull request can only be merged once the
   required checks (`python`, `pipeline`, `docs` and `dependency-review`) pass
   and the branch is up to date with `main`. CodeQL also analyzes every pull
   request.
5. Update the documentation (README, [wiki](https://github.com/BnRomain/himalayan-expedition-success/wiki))
   when the usage or the results change, and `CHANGELOG.md` under "Unreleased".

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (`2.0.0`): incompatible change, for example to the layout of the
  data files or the columns of `data/exped_clean.csv`;
- **MINOR** (`1.1.0`): new backward-compatible feature, such as a new
  predictor, a new figure or another model;
- **PATCH** (`1.0.1`): backward-compatible bug fix.

Releases are published from `main` with a `vX.Y.Z` tag. GitHub generates their
notes from the merged pull requests, grouped by label as configured in
[`.github/release.yml`](.github/release.yml), and the `version` field of
[`CITATION.cff`](CITATION.cff) is updated at the same time.
