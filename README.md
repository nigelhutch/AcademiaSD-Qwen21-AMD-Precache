# AcademiaSD #90 — Experimental Windows process-isolated pre-cache V2

Status: EXPERIMENTAL, bounded local qualification only.
Upstream baseline: ab17ed89fc0474819cb2d6268f9f8ce43ff50dd1
Target: scripts/1_pre_cache_qwen_image21.py
Candidate SHA256: 2523513febcbe2cb4bd7f58876c29f40f223e97afabd3e780e12f25f3c6cf18d

Evidence: 18/18 source and real Windows subprocess checks; 49/49 CPU synthetic tensor-cache checks (PyTorch 2.9.1+rocm7.2.1). No actual Qwen model, full pre-cache, GUI integration or LoRA training run locally. The reporter's HIP 7.13 environment differs. The underlying crash root cause remains unproven.

Test in a separate checkout at the pinned revision, with a separate test cache/project and no running training or other cache writer. Back up existing cache/settings. Review candidate.patch and run git apply --check candidate.patch from repository root before applying. If check fails, stop and report the source revision. Do not force the patch. Use existing venv interpreter and working directory normally used by the application; set LORALAB_ISOLATE_PRECACHE=1 before launching the application from the same environment, or pass --isolate-precache to the pre-cache script when invoked directly with the existing configuration. Do not change dependencies or drivers. The separate preview-only command is unchanged. Keep settings, dataset and models unchanged throughout both phases. The workaround is Windows-only and opt-in; clear the environment variable to disable. For rollback revert only this patch in the disposable checkout. Do not reuse incomplete cache for training after any failure/cancellation; restart with a fresh test cache. Configuration hashing is bounded, not a transactional snapshot of all inputs. A full GUI cancellation and restart test remains required.

Please report source revision, runtime versions, process exit status, full cache generation, cache file counts, and a short LoRA training result. Do not share private dataset images or captions. If either phase fails, stop and report the last flushed stage and native fault module/offset if available.

## Reproducing the local tests
Copy the pinned upstream script into the test directory as baseline.py; normalize CRLF to LF for exact candidate hash reproduction. After applying the patch in the disposable checkout, copy the patched script into the test directory as 1_pre_cache_qwen_image21.py. Run test_v2.py on Windows and test_cache_contract.py with a local PyTorch interpreter using python -X utf8. The latter uses CPU synthetic outputs only. Both receipts are bounded evidence, not upstream acceptance.
