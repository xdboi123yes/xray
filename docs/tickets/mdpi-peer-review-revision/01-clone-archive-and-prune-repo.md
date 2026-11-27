# 01: Clone repository, archive full-stack application, and prune to research-only layout

**What to build:** Clone the peer-review revision branch into the local workspace while keeping local agent workflow docs excluded from git tracking; preserve the existing web application, container definitions, remote runner scripts, documentation site config, and legacy Springer manuscript files in a cleanly organized separate local application archive repository; prune the public repository down to its research-only scope so reviewers see only code, configs, notebooks, tests, and MDPI manuscript artifacts; verify the remaining research test suite, linter, and type checker pass green.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The local workspace is synced to the peer-review revision branch with local agent workflow docs excluded from git status via local git exclude rules.
- [x] Domain glossary, architecture decision records, spec, and ticket files remain intact after repository sync.
- [x] A separate local application archive repository exists in a minimal organized folder structure containing the full-stack web app, container files, remote runners, and legacy Springer files, with its own clean initial git commit.
- [x] The public research repository no longer contains web application, container, remote runner, legacy Springer, or UI/Docker CI workflow clutter.
- [x] All remaining research unit and integration tests, linter checks, and static type checks pass cleanly locally.
