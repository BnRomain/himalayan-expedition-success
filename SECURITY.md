# Security Policy

## Supported Versions

Only the `main` branch is maintained.

| Component | Path | Supported |
| --- | --- | --- |
| Data processing and regression scripts | `src/` | Yes |
| Tests | `tests/` | Yes |
| GitHub Actions workflows and Dependabot configuration | `.github/` | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please do not disclose it publicly through a GitHub issue.

Instead, please report it privately to the project maintainer through GitHub's private vulnerability reporting: [report a vulnerability](https://github.com/BnRomain/himalayan-expedition-success/security/advisories/new).

When reporting a vulnerability, please provide:

* A short description of the vulnerability
* The affected file or component
* The steps required to reproduce the issue
* Any relevant screenshots, logs, or code examples

## Scope

This policy applies to the source code and configuration contained in this repository, in particular:

* the reading of the CSV files of `data/` with pandas, including the intermediate files written by `src/02_wrangling.py`;
* the GitHub Actions workflows and the Dependabot configuration.

The scripts read plain CSV files only and never unpickle data. As this is an educational project, security issues should be reported responsibly so they can be reviewed and addressed without unnecessarily exposing other users or contributors.

## Security Measures

* **CodeQL** code scanning on Python and GitHub Actions for every pull request and every push to `main`.
* **Dependabot** version updates for the Python dependencies (pinned in `requirements*.txt`) and the GitHub Actions, plus Dependabot security updates. Patch and minor updates are merged automatically only once the required checks of `main` have passed.
* **Dependency review** blocks pull requests that introduce dependencies with known vulnerabilities of moderate severity or higher.
* **Secret scanning** with push protection.
* **Least privilege**: the workflows only get read access to the repository, except the Dependabot auto-merge workflow, which needs to write to pull requests.

## Response

Security reports will be reviewed as soon as reasonably possible.

Depending on the nature and severity of the issue, appropriate corrective actions may include:

* Fixing the vulnerability
* Updating dependencies
* Improving the code or configuration
* Documenting the issue and its resolution
