# Third-party attribution

## Amazon Shopping Queries Dataset (ESCI)

- Upstream repository: https://github.com/amazon-science/esci-data
- Exact source commit: `7916cdf6ab75a462e77f20ab40428a10923998d5`
- License: Apache License 2.0; copied in `ESCI-LICENSE`.
- Upstream notice: copied in `ESCI-NOTICE`.
- Derived materials: selected query/product identifiers, ESCI labels, candidate membership, exclusion hashes, and illustrative product/query excerpts. They have been filtered and reorganized for this benchmark; they are not the full original dataset.

These derived materials are separately attributed and are not relicensed as original MIT code. The full Parquet files are downloaded from upstream and are not committed.

## BAAI BGE reranker

- Model: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Recorded model commit: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- The official model card identifies Apache-2.0 licensing. Model weights are not redistributed here.

## Hosted APIs

Jev and GPT-4.1 Mini were accessed through OpenRouter. No hosted model weights are redistributed. Recorded responses, derived decisions, timing observations, and usage amounts are benchmark evidence; the MIT license does not grant rights to third-party models or services.
