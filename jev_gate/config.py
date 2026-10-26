from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
import tomllib


@dataclass
class Config:
    protocol: str = "legacy"
    task: str = "top5_any_exact"
    seed: int = 42
    data_dir: str = "data/esci"
    run_dir: str = "runs/openrouter"
    development_size: int = 200
    validation_size: int = 100
    test_size: int = 500
    challenge_size: int = 100
    challenge_pool_size: int = 1000
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "main"
    reranker_local_path: str = ""
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512
    local_files_only: bool = False
    jev_model: str = "typesafe/jev-1.13"
    gpt_model: str = "openai/gpt-4.1-mini"
    gpt_max_tokens: int = 32
    gpt_input_price_per_million: float = 0.40
    gpt_output_price_per_million: float = 1.60
    concurrency: int = 1
    request_timeout_s: float = 30.0
    max_attempts: int = 3
    requests_per_second: float = 1.0
    target_risk: float = 0.05
    min_accepted: int = 30
    bootstrap_samples: int = 2000
    input_price_per_million: float = 0.042
    title_chars: int = 300
    bullet_chars: int = 500
    prior_prepared_path: str = ""
    development_pool_size: int = 6000
    development_negative_target: int = 200
    development_min_negatives: int = 100
    secondary_risk: float = 0.05
    challenge_s: int = 100
    challenge_c: int = 70
    challenge_i: int = 30
    api_budget_usd: float = 2.0
    budget_safety_usd: float = 0.05
    prior_prepared_paths: list[str] = None
    fit_size: int = 1200
    select_size: int = 800
    certification_size: int = 1500
    fit_min_class: int = 50
    certification_min_accepted: int = 200
    selection_risk_fraction: float = 0.5
    family_alpha: float = 0.05
    preflight_max_estimate_usd: float = 2.70
    dataset_revision: str = ""
    dataset_files_sha256: dict[str, str] = None
    exclusion_manifest_path: str = ""
    exclusion_manifest_sha256: str = ""
    split_manifest_path: str = ""
    split_manifest_sha256: str = ""
    cost_profile_path: str = ""
    cost_profile_sha256: str = ""
    calibration_size: int = 800
    calibration_retention_slack: float = 0.01
    retention_noninferiority_margin: float = 0.02
    core_threshold_grid: list[float] = None

    @classmethod
    def load(cls, path: str | None = None):
        source = Path(path or "config.example.toml").resolve()
        if path and not source.exists():
            raise ValueError(f"Config does not exist: {source}")
        values = tomllib.loads(source.read_text("utf-8")) if source.exists() else {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        cfg = cls(**values)
        base = source.parent if source.exists() else Path.cwd()
        for name in ("data_dir", "run_dir", "reranker_local_path", "prior_prepared_path", "exclusion_manifest_path", "split_manifest_path", "cost_profile_path"):
            value = getattr(cfg, name)
            if value:
                setattr(cfg, name, str((base / value).resolve()))
        if cfg.prior_prepared_paths:
            cfg.prior_prepared_paths = [str((base / p).resolve()) for p in cfg.prior_prepared_paths]
        cfg.validate()
        return cfg

    def validate(self):
        if self.dataset_revision and not re.fullmatch(r"[a-f0-9]{40}", self.dataset_revision):
            raise ValueError('Pin dataset_revision to a full Git commit')
        for field in ('exclusion_manifest', 'split_manifest', 'cost_profile'):
            path, checksum = getattr(self, field + '_path'), getattr(self, field + '_sha256')
            if bool(path) != bool(checksum) or checksum and not re.fullmatch(r'[a-f0-9]{64}', checksum):
                raise ValueError('Manifest/profile path and SHA-256 must be supplied together')
        if self.exclusion_manifest_path and self.prior_prepared_paths:
            raise ValueError('Choose either portable exclusion manifests or original prior paths')
        if self.protocol not in {"legacy", "fusion_v3", "topk_v4", "topk_calibrated_v5"}:
            raise ValueError("Unknown experiment protocol")
        if self.task not in {"top5_any_exact", "top1_exact", "topk_filter"}:
            raise ValueError("Unknown task")
        if (self.task=='topk_filter') != (self.protocol in {'topk_v4','topk_calibrated_v5'}):
            raise ValueError('Top-K task and protocol must be selected together')
        for key in ("development_size", "validation_size", "test_size", "batch_size",
                    "max_length", "concurrency", "max_attempts", "min_accepted",
                    "bootstrap_samples", "title_chars", "bullet_chars", "gpt_max_tokens"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.challenge_size < 0 or self.challenge_pool_size < self.challenge_size:
            raise ValueError("challenge_pool_size must be >= challenge_size >= 0")
        if not 0 <= self.target_risk < 1:
            raise ValueError("target_risk must be in [0, 1)")
        if self.requests_per_second <= 0 or self.request_timeout_s <= 0:
            raise ValueError("Request rate and timeout must be positive")
        if min(self.input_price_per_million, self.gpt_input_price_per_million,
               self.gpt_output_price_per_million) < 0:
            raise ValueError("Price cannot be negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto/cpu/cuda/mps")
        if not re.fullmatch(r"typesafe/jev-\d+\.\d+(?:-\d{8})?", self.jev_model):
            raise ValueError("Use a fixed OpenRouter Jev version, e.g. typesafe/jev-1.13; latest aliases are rejected")
        if self.gpt_model not in {"openai/gpt-4.1-mini", "openai/gpt-4.1-mini-2025-04-14"}:
            raise ValueError("This GPT baseline supports only GPT-4.1 Mini; do not substitute another model")
        if self.protocol in {'topk_v4','topk_calibrated_v5'}:
            if self.task!='topk_filter' or not 0 < self.budget_safety_usd < self.api_budget_usd:
                raise ValueError('Invalid Top-K protocol settings')
            if self.challenge_s+self.challenge_c+self.challenge_i!=self.challenge_size:
                raise ValueError('Challenge quotas must sum to challenge_size')
            if self.gpt_max_tokens<512: raise ValueError('Structured Top-K GPT output needs >=512 token cap')
        if self.protocol == 'topk_calibrated_v5':
            if self.calibration_size < 1 or self.challenge_size or self.challenge_pool_size:
                raise ValueError('Calibrated Top-K requires calibration and natural test only')
            grid=self.core_threshold_grid
            if not grid or grid!=sorted(set(grid)) or any(isinstance(t,bool) or not 0<t<1 for t in grid):
                raise ValueError('Supply a fixed sorted unique core threshold grid in (0,1)')
            if not 0<=self.calibration_retention_slack<self.retention_noninferiority_margin<1:
                raise ValueError('Calibration slack must be below non-inferiority margin')
            if not 0<self.preflight_max_estimate_usd<self.api_budget_usd-self.budget_safety_usd:
                raise ValueError('Forecast ceiling must leave budget safety headroom')
        if self.task == "top1_exact":
            if self.protocol != 'fusion_v3' and not 1 <= self.development_min_negatives <= self.development_negative_target < self.development_size:
                raise ValueError("Invalid development enrichment sizes")
            if self.protocol != 'fusion_v3' and self.development_pool_size < self.development_size:
                raise ValueError("Screening pool smaller than development sample")
            if min(self.challenge_s, self.challenge_c, self.challenge_i) < 0:
                raise ValueError("Negative challenge quota")
            if self.protocol == 'fusion_v3':
                if not 0 < self.secondary_risk < self.target_risk < 1:
                    raise ValueError('Fusion protocol expects secondary < primary risk')
                for name in ('fit_size','select_size','certification_size','fit_min_class','certification_min_accepted'):
                    if getattr(self,name)<1: raise ValueError('Invalid fusion size: '+name)
                if not 0 < self.selection_risk_fraction < 1 or not 0 < self.family_alpha < 1:
                    raise ValueError('Invalid risk audit settings')
                if not 0 < self.preflight_max_estimate_usd < self.api_budget_usd:
                    raise ValueError('Invalid preflight estimate limit')
            elif not 0 < self.target_risk < self.secondary_risk < 1:
                raise ValueError("Expected primary risk < secondary risk")
            if not 0 < self.budget_safety_usd < self.api_budget_usd:
                raise ValueError("Invalid budget reserve")

    def root(self, mock=False):
        return Path(self.run_dir) / ("mock" if mock else "real")

    def dict(self):
        return asdict(self)
