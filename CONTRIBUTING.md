# Contributing to ThoraxAI

Thank you for contributing to our Bachelor Thesis Clinical Decision Support System!

## Architectural Guidelines
All contributions must adhere to the layered system design:
- **Core Layer:** Pure algorithmic and model definition logic. No dependencies on application, web, or legacy shims.
- **Application Layer:** Orchestrates business logic, DTO mapping, and execution flow.
- **Infrastructure Layer:** Database adapters, MLflow tracking, PyTorch trainers, and exporters.
- **Web Layer:** API endpoints, WebSockets streaming, and React frontend assets.

## Quality Gates & Verification
Before pushing your changes, please run:
```bash
make lint
make check-imports
make check-lang
make test
```
All checks must compile and verify green before opening a pull request.
