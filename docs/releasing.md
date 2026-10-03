# Releasing a verified candidate

Use the existing `.github/workflows/release.yml`, triggered once by a published
GitHub Release. Keep the existing PyPI project and no-environment Trusted
Publisher binding. Only the upload job receives `id-token: write`; PR and source
verification jobs have read-only permissions. Keep PyPI attestations enabled.

1. Complete source/core/generated-code checks, minimum and recommended real CPU
   integration, current local CUDA LoRA/QLoRA qualification and CI. Merge under
   repository rules and select the final commit and a new version/tag.
2. Build wheel and sdist **once** from that commit. Run `twine check`; build from
   the sdist into a separate directory to verify completeness. Do not substitute
   the rebuilt file for the qualified candidate.
3. Install the wheel in clean core and training environments outside the checkout.
   Run `scripts/qualify_candidate.py`, then `--training --cuda` on qualified local
   hardware. Keep logs and their artifact SHA-256 association. Check author,
   dependencies, templates, web assets and actual update/save/reload behavior.
4. Write `candidate.json` with version, commit, two distribution SHA-256 values
   and passed qualification receipts (core, CPU, CUDA LoRA/QLoRA). Write
   `SHA256SUMS` for those exact two files. Upload these files to a **draft** release,
   inspect them, then publish it once. Do not move an existing tag.
5. The workflow downloads those same assets, verifies tag/source/resources/hashes,
   tests source and executes real CPU training from the installed candidate.
   It uploads those verified files as an Actions artifact. The OIDC job checks
   hashes again and uploads them without a rebuild or `skip-existing`.
6. The public-install job checks official PyPI JSON and files.pythonhosted.org
   hashes against the candidate, then installs the explicit public version in a
   fresh environment outside the checkout and executes core paths. Perform the
   bounded public CUDA training checks locally too. Verify workflow conclusion,
   release links, version, author and default-branch CI before reporting published.

If upload fails, inspect whether either file is already public before retrying.
Hash differences or changed code require a newly qualified new version, never a
same-version rebuild presented as equivalent. No raw credential belongs in logs.
Local measurement receipts are evidence for the named hardware/stack, not signed
proof of human review or a cross-hardware certification.
