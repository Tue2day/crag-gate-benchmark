# Reproduction and artifact guide

## 1. Three different activities

| Activity | API key | GPU | Network | Meaning |
|---|---|---|---|---|
| Verify published evidence | No | No | No, after installing dependencies | Recompute statistics from recorded decisions and logs |
| Run synthetic mock pipeline | No | No | No, after installing dependencies | Exercise protocol and checkpoint mechanics |
| Rerun real inference | Yes | Recommended for BGE | Yes, unless data/model are transferred offline; API access still required | Generate a new paid observation |

The first two activities must not be described as independently regenerating cloud-model benchmark scores. New inference may differ even with fixed model IDs, especially when a hosted provider exposes an alias rather than a weight commit.

## 2. Environment

Use Python 3.11 or 3.12. Install with `python -m pip install -e ".[dev]"` for tests and result verification, and add the `rerank` extra for real BGE inference. An activated Conda environment is supported. For NVIDIA inference, obtain the appropriate CUDA PyTorch build from https://pytorch.org/get-started/locally/ rather than assuming that the default pip wheel matches your driver.

```text
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m jev_gate doctor --config config.topk-calibrated-v5.toml
```

The recorded machine used Windows, an RTX 4090 Laptop GPU with 16 GB VRAM, PyTorch 2.3.1/CUDA 11.8, Transformers 4.49.0, and NumPy 1.26.4. CPU inference is supported in FP32 but will be slower. CUDA uses FP16, batch size 8, and maximum sequence length 512; out-of-memory handling reduces the batch size rather than silently switching the experiment to mock mode.

## 3. Pinned data and portable manifests

The dataset is `amazon-science/esci-data` at commit `7916cdf6ab75a462e77f20ab40428a10923998d5`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `shopping_queries_dataset_examples.parquet` | 51,286,808 | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` |
| `shopping_queries_dataset_products.parquet` | 1,108,857,465 | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` |

`download-data` rejects Git LFS pointers, truncated Parquet files, changed checksums, and an existing download manifest with a different pinned commit. You can instead transfer the exact two files into the configured data directory.

The experiment uses US-locale judged pools with at least five valid candidates. It does not filter to the reduced `small_version` task-1 benchmark, and must not be compared directly with that task's official ranking scores. Normalization is Unicode NFKC, case folding, and whitespace collapse.

The manifests contain:

- **Dataset identity:** upstream commit, file sizes, and checksums.
- **Exclusion hashes:** `digest(normalize_query(query))` for every previously prepared query, including unused reservations. Here `digest` means the project's SHA-256 of canonical JSON, not a plain-text SHA-256. Round 4 excludes 23,670 prior texts; round 5 excludes 25,680.
- **Split membership:** query ID, normalized-query hash, source/assigned split, and the original judged candidate-pool fingerprint. Membership and order are checked after selection, before paid calls.
- **Historical cost profile:** fixed model IDs, prompt-contract hash, serialized request byte totals, and known usage totals from round 4. Round 5 uses this only for an initial conservative forecast, never for labels or threshold selection.

The default public configs use these manifests. The original `prior_prepared_paths` mode remains available for a user who has the original research archives, but it is mutually exclusive with the portable exclusion mode.

Keep the pinned inputs unchanged to reconstruct the historical split. Changing a seed or sample size while retaining the published membership manifest correctly fails. To design a new study, use a new config/run directory and an independently prepared manifest; do not edit the published results to make a changed study appear identical.

## 4. Offline evidence verification

```text
python -m jev_gate verify-results --experiment v5
python -m jev_gate verify-results --experiment v4
```

An alternative exported directory can be supplied with `--artifacts PATH`. Verification checks:

1. File SHA-256 values and safe relative artifact names.
2. Exactly five unique ranked candidates per model/query and identical gold candidate pools for both models.
3. Raw default decisions at 0.5 and the separately applied calibrated threshold at 0.35.
4. The complete 800-query calibration grid and its precision-maximizing feasible choice.
5. Test candidate/query counts and all seven reported point metrics.
6. The exact recorded query order, seed, 2,000 paired query-cluster bootstrap intervals, and the one-sided retention bound.
7. The failed retention non-inferiority and successful precision-superiority conclusions.
8. Test latency percentiles from non-cache invocation records and model receipt totals across the round.

Checksums detect changes relative to the committed manifest; they are not an external timestamping or adversarial attestation service. Git commits and release checksums identify the published snapshot. Original experiment fingerprints and historical code hashes are separately retained as provenance.

## 5. Detailed release evidence

The `v0.7.0` release includes `crag-gate-benchmark-v0.7.0-artifacts.zip` and an outer SHA-256 file. The ZIP contains per-round:

- Recorded API response answers, native probabilities/booleans, parsed diagnostics, usage, attempts, and durations.
- Query/product evidence and ESCI judgments for the selected ranked groups.
- Frozen historical Python source snapshots and original source/data fingerprints.
- Attribution/license files and an internal file checksum manifest.

This is a **derived public export**, not the byte-identical private archive. Provider trace IDs, account balance snapshots, and personal filesystem paths are removed; answer values, decisions, model IDs, usage amounts, and timing measurements are preserved. Original record fingerprints and new export checksums are distinct.

```text
python -m jev_gate verify-results --experiment v5 --raw-artifacts crag-gate-benchmark-v0.7.0-artifacts.zip
python -m jev_gate verify-results --experiment v4 --raw-artifacts crag-gate-benchmark-v0.7.0-artifacts.zip
```

With the ZIP supplied, the verifier additionally validates archive checksums, reparses the native model responses, and checks the exported candidate decisions and Jev probabilities against those responses.

## 6. Synthetic pipeline

Use the seven mock commands in the README with `config.mock.toml`. The sample sizes remain 800/2,000, but the product/query data and responses are generated deterministic fixtures. The config uses CPU, and the mock path never initializes the real BGE model or sends HTTP requests. All outputs are placed beneath `mock`; charts are marked simulated. Usage and charges are not fabricated.

Tests also use smaller isolated fixtures to cover invalid responses, failure recovery, source/config freeze checks, no feasible threshold, and statistical boundaries. Those fixture sizes are not the real benchmark's dataset size.

## 7. Real inference sequence and controls

Use the ordered real commands in the README. Before any test inference:

1. Prepare and rerank the complete fixed dataset locally.
2. Run preflight to verify available credit, exact model pricing, shared evidence, dataset/source fingerprints, and forecast.
3. Call both models on calibration only.
4. Freeze the selected threshold and calibration receipt fingerprints.
5. Allow test calls only after these frozen checks pass.

The default uses fixed Jev `typesafe/jev-1.13-20260917` and GPT `openai/gpt-4.1-mini`. The GPT ID is the provider-returned hosted identifier; no weight commit is available. If a model version is unavailable or its quoted price differs, stop and define a new explicitly documented protocol. Do not silently change versions.

Each model has two workers and a 2-request/second limiter. Timeouts are 45 seconds, with up to three attempts per invocation. HTTP 429/5xx and transport/SSL failures are eligible for bounded retry; authentication/balance errors stop execution. Repeated schema failures stop the phase. `UNKNOWN` attribute evidence remains a fallback condition, never a confirmed violation or an automatically accepted candidate.

The shared budget reserves in-flight estimates before requests, settles known usage, and conservatively retains unknown-attempt amounts. The historical $4.20 cap includes a $0.10 safety reserve. Before spending, the complete-size forecast including 25% contingency must not exceed $4.05. No sample reduction is performed when a budget check fails.

## 8. Checkpoints and troubleshooting

Successful request caches and append-only journals survive interrupted phases. Resume the **same phase with the same frozen config/source**; successful cache hits are not new inference or new charges. Do not rerun calibration after threshold freezing, and do not prepare or rerank over a frozen run.

If an API lock remains, first confirm the previous process is no longer running. Only then remove the stale lock. Changing code or configuration under a frozen run is rejected; use a new run directory for a deliberate new study.

| Symptom | Action |
|---|---|
| Missing key / 401 / 403 | Configure the local OpenRouter key and confirm model access |
| 402 or insufficient preflight credit | Add credit; retain full sample sizes |
| Price or model identity mismatch | Stop and document a new protocol; no automatic substitution |
| Missing Parquet / LFS pointer / checksum error | Obtain the exact pinned data files |
| CUDA unavailable | Install a compatible build/driver, or explicitly choose CPU before freezing |
| Freeze or manifest mismatch | Restore the intended inputs or create a separate study |
| No feasible calibration point | Report failure and stop before test inference |

## 9. Historical clocks, timings, and reliability

Round 5's ROG wall clock was observed approximately 190 seconds ahead of an independent UTC reference. The raw timestamps are not rewritten. Local freeze and remote provider `created` fields therefore must not be compared as if synchronized. Stage sequencing is enforced by source checks and calibration fingerprints; GPT's provider clock also places all calibration calls before test calls.

Durations use monotonic `perf_counter` and are unaffected by wall-clock skew. HTTP distributions may include failed attempts; a separately disclosed successful-HTTP distribution uses the final HTTP 200 attempt per test invocation. Gate timing includes pacing and retries inside that invocation, but excludes queue waiting, reranking, and manual process-restart time.

Round 4 experienced an SSL interruption and same-protocol resume. Three reserved attempts lack complete timing/usage records, and an account-total difference of $0.0015412 remains unassigned. Round 5 completed after 42 Jev transport errors and one GPT transport error; all 2,800 logical invocations per model ultimately succeeded. The unknown-budget reserve is not an actual charge, and final success does not imply first-attempt or SLA reliability.
