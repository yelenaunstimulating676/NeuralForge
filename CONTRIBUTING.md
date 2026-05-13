# Contributing to NeuralForge

Thanks for your interest! NeuralForge is in early stages and feedback, issues, and PRs are welcome.

## Reporting bugs

Open an issue with:
- What you tried to do
- What you expected to happen
- What actually happened (include stack trace if any)
- Your environment: OS, Python version, GPU model, VRAM

## Requesting features

Open an issue with the `enhancement` label. Describe:
- What use case the feature serves
- Why existing features do not cover it
- Suggested approach (optional)

## Submitting code

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes following existing code style
4. Run the test suite: `cd backend && pytest tests/`
5. Commit with a descriptive message
6. Open a PR explaining the change

## Code style

- **Python**: type hints, docstrings on public functions, follow existing patterns
- **JavaScript/JSX**: functional components, hooks, Tailwind classes for styling
- **Commits**: imperative mood, short subject line, body if needed

## Areas where help is welcome

- Testing on Linux/macOS (current verification is Windows-only)
- Additional dataset format extractors
- Translation/i18n
- Documentation improvements

## License

By contributing, you agree your contributions will be licensed under the MIT License.