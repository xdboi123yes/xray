# 10: Finalize public reproducibility README, merge to default branch, and publish v1.0-mdpi-rev1 GitHub Release

**What to build:** Rewrite the public repository overview into a paper-first reproducibility guide (system summary, dataset preparation instructions, exact table/figure reproduction commands, L4 hardware/timing summary, weight download/verification instructions, and citation BibTeX), verify SHA-256 and git-commit parity between the provenance manifest and release assets, merge the peer-review branch into the default branch, create the v1.0-mdpi-rev1 GitHub Release attaching trained Track B model weights, frozen library versions, and SHA-256 checksums, and confirm public download links match the manuscript's Data Availability Statement.

**Blocked by:** 07: Execute Track B calibration, full ablation/revision analyses, CheXpert, and L4 latency benchmarks; 09: Update MDPI Track B manuscript (clean & highlighted), figures, and point-by-point response letter

**Status:** open

- [ ] Repository overview is rewritten cleanly for MDPI reviewers and researchers with zero stale web-app, Docker, or legacy runner instructions.
- [ ] Peer-review branch merges cleanly into the default branch with green CI checks (tests, lint, type checks).
- [ ] GitHub Release v1.0-mdpi-rev1 is tagged on the merged default branch commit and hosts trained Track B model checkpoints, frozen dependency lockfile, provenance manifest, and SHA-256 checksums.
- [ ] Public repository URL, release tag, and asset download instructions match the manuscript's Data Availability Statement exactly and verify cleanly.
