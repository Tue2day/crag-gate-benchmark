# Release validation

The v0.7.0 publication is a new packaging/portability release. It does not alter the historical model answers, thresholds, or conclusions.

## Checks completed before publication

- All 73 existing experiment regression tests retained; publication adds offline verification, portable exclusions, split-pool checks, CLI dispatch, pinned downloads, tampering checks, and Unicode-safe release JSONL parsing.
- The complete local suite passes **86 tests** with external HTTP blocked by the test fixture.
- The v4 and v5 offline verifiers reproduce all seven paired-query bootstrap difference intervals and recorded point/count, latency, and charge results.
- Native-response verification passes on **1,220 v4** and **5,600 v5** records in the detailed derived archive. This is response replay, not newly paid inference.
- Using the original pinned Parquet files on the recorded Windows machine, public code/manifests reconstructed both complete prepared pools with **exact original fingerprints**: v4 **2,010 queries** (including unused challenge reservations), v5 **2,800 queries**. No reranking or paid inference was needed for this check.
- Markdown local links, publication file sizes, credential patterns, personal absolute paths, and private account snapshot fields were checked. Account balance/credit snapshots are not in the public evidence.
- Export checksums are distinct from historical record/source fingerprints; frozen historical source is included in the release archive.

## Automated continuous checks

The GitHub Actions workflow runs the offline suite and both result verifiers on Ubuntu and Windows, using Python 3.11 and 3.12. It uses fixed official Action commits and no inference API credentials. Consult the live Actions status rather than treating a static document as proof that future builds will pass.

## Interpretation retained

The main point-estimate retention loss is **1.72pp**, but its one-sided 95% lower bound is **−2.65pp**. Retention non-inferiority is **not established**, precision superiority is supported, and the joint replacement acceptance condition remains false. A publication cleanup or successful CI run does not change this scientific finding.

Human review sheets remain unreviewed. The public benchmark is not a company deployment, a field-level human-annotated accessory classifier, or an end-to-end RAG SLA evaluation.
