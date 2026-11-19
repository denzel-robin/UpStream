# Repository contents and local setup

> **Superseded (October 2026 progress presentation).** This page describes the repository as it was for the
> earlier study: off-the-shelf NanoVSR, bicubic and direct delivery over the first
> MPEG-TS pipeline (`server/`, `client/`). Its numbers do not describe the final
> system. For the final model, live pipeline and results, see the top-level
> [README](../README.md) and [docs/REPORT.md](../docs/REPORT.md).

This repository contains the BTP streaming implementation, reproducible experiment
scripts/configurations, milestone reports, source attribution, compact measured
CSV/JSON evidence, report figures/crops and the labelled offline comparison backup.
Read `logs/progress-report-handoff.md` for validated scope and limitations.
The offline backup is not a live recording or performance evidence. Road-derived
imagery is CC BY-SA 3.0; see `presentation_assets/ATTRIBUTION.md`.

## Files intentionally kept local

The working `.venv` / `.venv-cda`, pretrained weights, upstream source checkouts,
downloaded source videos and approximately 6.8 GB of raw evaluation results are
not committed. No local files are deleted by these exclusions. Some historical
report links therefore refer to local-only raw evidence. Compact results are in
`presentation_assets/data/`; original protocols/configs and provenance are retained.
A fresh clone needs dependencies, upstream models, checkpoints and prepared video
assets before it can launch the demo. The empty historical `requirements.txt`
is not an installation specification. Recorded dependency versions are in
`presentation_assets/evidence/venv-freeze.txt` and `cda-freeze.txt`; do not blindly
install the CUDA-specific freeze on a different host.

## Upstream models (not our architectures or training)

- NanoVSR: https://github.com/filippawlicki/nanovsr
  Revision `b48e2bad89d01e22226eb8613bee25cb45623fae`.
  Local path `external/nanovsr`; checkpoint `checkpoints/nanovsr_226k.pth`.
  SHA256 `be3df12f2a0d99d9699656aa4a85a51c119663aa84c7040ac6580a0215d0b00e`.
- CDA-VSR: https://github.com/sspBIT/CDA-VSR
  Revision `5630821e5df4e110878d0caae39bcc53f658e9a9`.
  Local path `external/cda-vsr`; checkpoint `pretrained_models/best.pth`.
  SHA256 `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`.

Obtain weights using the upstream projects' documented sources and observe their
licenses. They are not redistributed here. CDA authentic-input integration is
pending; no authentic CDA quality or streaming result is claimed.

Source clip URLs/licenses/hashes are in `configs/natural-study.json` and
`presentation_assets/data/provenance.json`. Reference preparation, offline
scoring and playback commands are in the natural milestone and
`presentation_assets/REPRODUCE.md`. Do not rerun offline scoring alongside timed
playback. Existing local environments/checkpoints should be preserved.

## Git in the current managed workspace

The session exposes an empty read-only `.git` directory. Local repository metadata
is stored in ignored `.git-local/`; use `scripts/git_project.sh status` (or other
Git subcommands) in this workspace. This wrapper uses normal Git in a fresh clone.
Remote clones use an ordinary `.git` directory and normal Git commands.
