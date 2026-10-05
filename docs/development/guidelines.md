# Tinyopt Software Development Guidelines

This document outlines the core engineering practices, version control standards, commit message conventions, and code review rules for developing in the Tinyopt repository.

---

## 1. Core Software Development Principles

1. **KISS (Keep It Simple, Stupid)**:
   Avoid over-engineering. Optimization code is mathematically intricate; keep implementation paths clear, readable, and direct.

2. **Zero Dynamic Allocation in Critical Paths**:
   Never allocate heap memory inside optimization loops. Use fixed-size types or pass reusable workspace buffers.

3. **Single Responsibility Principle (SRP)**:
   Keep components focused:
   - A *loss function* computes a scalar metric and derivatives.
   - A *solver* computes linear step $\delta x$.
   - An *optimizer* governs the trust region, step acceptance, and outer iteration loop.

4. **Zero Warnings Policy**:
   Both GCC and Clang are configured with `-Wall -Wextra -Werror`. No compiler warnings are permitted.

5. **Defensive Numerical Programming**:
   - Check for non-finite values (`std::isnan`, `std::isinf`) in gradients and Hessians.
   - Return descriptive `StopReason` enum values when numerical failure is encountered rather than crashing.

---

## 2. Agent Commit Policy

> [!IMPORTANT]
> **No Automatic Commits**:
> AI agents and assistants must **NEVER** automatically commit changes unless explicitly requested by the user within the active session (e.g. "ok commit now", "create a commit").
> Working changes should remain staged or unstaged until explicitly instructed.

For versioned package builds and release commits, follow [Packaging and Releasing](packaging-and-releasing.md).

---

## 3. Git Commit Message Conventions (With Emojis)

All commit titles must follow the **Conventional Commits with Emojis** specification:

```text
<emoji> <type>(<optional-scope>): <subject>
```

### Supported Commit Types & Emojis

| Emoji | Type | Purpose | Example |
| :---: | :--- | :--- | :--- |
| 📝 | `docs` | Documentation additions or updates | `📝 docs: add coding style and architecture guides` |
| ✨ | `feat` | New features, optimizers, solvers, or APIs | `✨ feat(solvers): add dogleg trust-region step solver` |
| 🐛 | `fix` | Bug fixes and numerical stabilization | `🐛 fix(lm): prevent division by zero in damping ratio` |
| ⚡ | `perf` | Performance optimizations and memory reductions | `⚡ perf(accumulate): eliminate matrix temporaries with noalias` |
| 🧪 | `test` | Adding or updating unit tests and benchmarks | `🧪 test(diff): add Catch2 gradient validation for Huber loss` |
| ♻️ | `refactor` | Code refactoring without changing behavior | `♻️ refactor(traits): simplify Jet scalar detection` |
| 🎨 | `style` | Formatting, whitespace, clang-format alignment | `🎨 style: format headers according to .clang-format` |
| 🔧 | `chore` | Build configuration, Pixi tasks, CMake flags | `🔧 chore(pixi): add openmp dependency to bench environment` |
| 🔒 | `security`| Sanitizer fixes, bounds checking, security | `🔒 security(asan): fix buffer read beyond bounds in sparse solver` |

### Rules for Commit Messages:
- Use imperative, present tense ("add", not "added" or "adds").
- Do not capitalize the first letter after the type prefix.
- Do not place a period at the end of the subject line.
- Keep the title under 72 characters.

---

## 4. Branching & Pull Request Workflow

1. **Branch Naming**:
   - `feature/<name>` for new algorithms or capabilities.
   - `fix/<issue-name>` for bug fixes.
   - `refactor/<name>` for structural cleanups.

2. **Pre-Submission Checklist**:
   Before asking to commit or opening a PR:
   - [ ] Run formatter: `./scripts/format.sh --check`
   - [ ] Run test suite: `./scripts/run_tests.sh all`
   - [ ] Verify gradients: Any new analytical residual or loss has verified derivatives using `diff::CheckResidualsGradient`.
   - [ ] Zero warnings under `-Werror`.
