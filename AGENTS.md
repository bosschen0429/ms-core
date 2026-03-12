# AGENTS

This file defines the default working contract for coding agents in `ms-core`.

## Scope

- This repository contains the core processing logic used by the MS preprocessing toolkit.
- Changes here should stay focused on shared processing behavior, tests, and supporting configuration.
- If `ms-core` is checked out as a submodule inside another repository, treat it as its own git repository first.

## Task Intake

Before starting substantial work, structure the task in this order:

1. `Goal`
2. `Context`
3. `Constraints`
4. `Done When`

If the request is ambiguous, clarify the missing part instead of guessing.

## Pre-Flight Check

Before any development work, run:

```bash
git status
git branch --show-current
```

Do not proceed blindly if:

- the working tree is dirty and the changes are unrelated
- you are on `master`
- the task should be isolated in a worktree but is not

## Branch And Workspace Rules

- Do not develop directly on `master`
- Use branch names under:
  - `feature/*`
  - `fix/*`
  - `chore/*`
- Prefer git worktrees for behavior changes, multi-file tasks, or work that may span more than one session

Recommended pattern:

```bash
git worktree add ../<repo>.worktrees/<branch-name> -b <type>/<branch-name>
```

## Code Change Rules

- Read the relevant implementation and test files before editing
- Keep changes narrow and local; avoid opportunistic refactors unless required
- Do not change unrelated files just because they are nearby
- When touching behavior, also touch verification
- If a mistake repeats, encode the guardrail here instead of relying on memory

## Verification Rules

Do not claim completion without fresh evidence.

Default verification command:

```bash
pytest tests/ -v --tb=short -x
```

For smaller tasks, run the narrowest sufficient check first, then expand if risk justifies it.

## Root Hygiene Rules

- Do not write test temp directories into the repository root with `TemporaryDirectory(dir=Path.cwd())`
- Prefer fixtures from `tests/conftest.py`:
  - `project_temp_root`
  - `project_temp_dir`
- If pytest/temp/cache behavior changes, verify that root-level `tmp*`, `.pytest*`, and `.tmp` clutter does not reappear
- If old temp folders remain because of local ACL issues, report that clearly instead of pretending cleanup succeeded

## Parent Repo Coordination

- If `ms-core` is being changed from a parent repository that tracks it as a submodule, commit in `ms-core` first
- Push the `ms-core` commit before updating the parent repository's submodule pointer
- Do not leave a parent repository pointing at an unpushed `ms-core` commit

## Prohibited Actions

- No direct feature development on `master`
- No force push to `master`
- No merge or release without verification
- No skipping pre-flight checks
- No silent assumptions about external systems, secrets, or runtime configuration

## Done Standard

A task is only done when all of the following are true:

- requested behavior is implemented
- impacted tests or checks were run
- results were inspected
- important risks or gaps were stated
- branch and parent-repo state are consistent with the requested operation
