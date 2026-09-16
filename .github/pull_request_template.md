## Summary

<!-- What does this pull request change, and why? Link the related issue, for example "Closes #12". -->

## Type of change

- [ ] Bug fix
- [ ] New feature, model or figure
- [ ] Documentation
- [ ] CI, dependencies or tooling

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `python -m pytest` passes and the coverage stays above the threshold
- [ ] The scripts still run in order (`python src/01_exploration.py` to `python src/05_regression.py`)
- [ ] The figures of `docs/figures/` and the numbers of the README are updated if the results changed
- [ ] `npx markdownlint-cli2` and `lychee --offline --include-fragments .` pass (if Markdown files changed)
- [ ] `CHANGELOG.md` is updated under "Unreleased"
