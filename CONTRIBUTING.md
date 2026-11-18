# Contributing to hello-agent

Thanks for taking the time to look at this project. Bug reports, questions and small,
focused pull requests are all welcome.

## Before you start

- For anything larger than a typo or a small fix, please open an issue first so we can agree
  on the approach.
- Check the open issues to avoid duplicate work.

## Development setup

Python 3.12 or newer is required.

```bash
git clone https://github.com/RezaManzour/hello-agent.git
cd hello-agent
uv sync
```

## Checks to run before opening a pull request

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The same checks run in GitHub Actions on every push and pull request; a pull request is
merged only when they pass.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one logical change, with tests for new behaviour.
3. Write commit messages in the imperative mood ("Add X", "Fix Y").
4. Describe what changed and why in the pull request, and link the related issue.

If a change affects reported numbers or results, say so explicitly and explain how they were
regenerated.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to
uphold it.
