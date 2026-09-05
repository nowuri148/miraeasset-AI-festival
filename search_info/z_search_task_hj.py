from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .search_pipeline import (
    CompanyChromaHybridChunkStore,
    HybridChunkStore,
    ManifestGraphCatalog,
    execute_search_plan,
    group_members_from_keyword_result,
)


def _debug(enabled: bool, stage: str, message: str) -> None:
    if enabled:
        print(f"[SEARCH][{stage}] {message}", flush=True)


def _configured_path(
    root: Path,
    environment_name: str,
    candidates: tuple[str, ...],
    *,
    optional: bool = False,
) -> Path | None:
    configured = os.getenv(environment_name)
    if configured:
        value = Path(configured).expanduser()
        return value if value.is_absolute() else root / value
    resolved = tuple(root / candidate for candidate in candidates)
    for value in resolved:
        if value.exists():
            return value
    return None if optional else resolved[0]


@dataclass(frozen=True)
class SearchRuntimeConfig:
    project_root: Path
    manifest_path: Path
    universe_path: Path
    correction_edges_path: Path | None
    chroma_db_path: Path
    company_map_path: Path
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    event_date_fallback: str | None = "rcept_dt"

    @classmethod
    def from_environment(
        cls,
        project_root: str | Path | None = None,
        *,
        strict_event_date: bool = False,
    ) -> "SearchRuntimeConfig":
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        manifest = _configured_path(
            root,
            "SEARCH_MANIFEST_PATH",
            (
                "corpus/derived/manifest_v3_reviewed.jsonl",
                "corpus/derived/manifest_v3.jsonl",
                "corpus/manifest.jsonl",
            ),
        )
        universe = _configured_path(
            root,
            "SEARCH_UNIVERSE_PATH",
            ("corpus/universe.csv",),
        )
        correction_edges = _configured_path(
            root,
            "SEARCH_CORRECTION_EDGES_PATH",
            (
                "corpus/derived/graph_edges_v1_reviewed.jsonl",
                "corpus/derived/graph_edges_v1.jsonl",
            ),
            optional=True,
        )
        chroma_db = _configured_path(
            root,
            "SEARCH_CHROMA_DB_PATH",
            ("corpus/vector_db/chroma_company",),
        )
        company_map = _configured_path(
            root,
            "SEARCH_COMPANY_MAP_PATH",
            ("corpus/vector_db/company_collection_map.json",),
        )
        assert manifest is not None
        assert universe is not None
        assert chroma_db is not None
        assert company_map is not None
        return cls(
            project_root=root,
            manifest_path=manifest,
            universe_path=universe,
            correction_edges_path=correction_edges,
            chroma_db_path=chroma_db,
            company_map_path=company_map,
            embedding_model_name=os.getenv(
                "SEARCH_EMBEDDING_MODEL", "BAAI/bge-m3"
            ),
            device=os.getenv("SEARCH_DEVICE", "cpu"),
            event_date_fallback=None if strict_event_date else "rcept_dt",
        )

    def validate(self) -> None:
        required = {
            "manifest": self.manifest_path,
            "universe": self.universe_path,
            "Chroma DB": self.chroma_db_path,
            "company collection map": self.company_map_path,
        }
        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing operational search files: " + "; ".join(missing)
            )
        if (
            self.correction_edges_path is not None
            and not self.correction_edges_path.exists()
        ):
            raise FileNotFoundError(self.correction_edges_path)


class OperationalSearchExecutor:
    """서버 시작 시 한 번 생성하고 요청마다 재사용하는 검색 실행기."""

    def __init__(self, config: SearchRuntimeConfig, *, debug: bool = False):
        config.validate()
        self.config = config
        self.debug = debug
        _debug(debug, "CONFIG", f"project_root={config.project_root}")
        _debug(debug, "CONFIG", f"manifest={config.manifest_path}")
        _debug(debug, "CONFIG", f"universe={config.universe_path}")
        _debug(debug, "CONFIG", f"correction_edges={config.correction_edges_path}")
        _debug(debug, "CONFIG", f"chroma_db={config.chroma_db_path}")
        _debug(debug, "CONFIG", f"company_map={config.company_map_path}")
        _debug(
            debug,
            "CONFIG",
            f"embedding_model={config.embedding_model_name} device={config.device}",
        )

        self.catalog = ManifestGraphCatalog(
            config.manifest_path,
            correction_edges_path=config.correction_edges_path,
            universe_path=config.universe_path,
            event_date_fallback=config.event_date_fallback,
            debug=debug,
        )
        self.store = CompanyChromaHybridChunkStore(
            self.catalog,
            config.chroma_db_path,
            config.company_map_path,
            model_name=config.embedding_model_name,
            device=config.device,
            debug=debug,
        )

    def run(
        self,
        *,
        question: str,
        search_plan: dict[str, Any],
        keyword_result: dict[str, Any],
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        return run_search_task(
            question=question,
            search_plan=search_plan,
            keyword_result=keyword_result,
            catalog=self.catalog,
            store=self.store,
            debug=self.debug,
            raise_on_error=raise_on_error,
        )


_operational_executor: OperationalSearchExecutor | None = None


def get_operational_executor(
    *,
    config: SearchRuntimeConfig | None = None,
    project_root: str | Path | None = None,
    debug: bool = False,
) -> OperationalSearchExecutor:
    """서버 프로세스에서 BGE-M3와 Chroma를 한 번만 초기화한다."""
    global _operational_executor
    if _operational_executor is None:
        runtime_config = config or SearchRuntimeConfig.from_environment(project_root)
        _operational_executor = OperationalSearchExecutor(
            runtime_config,
            debug=debug,
        )
    return _operational_executor


def run_search_task(
    *,
    question: str,
    search_plan: dict[str, Any],
    keyword_result: dict[str, Any],
    catalog: ManifestGraphCatalog,
    store: HybridChunkStore,
    debug: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """재질문이 끝난 입력을 문서 확정부터 최종 컨텍스트까지 실행한다."""
    if keyword_result.get("is_complete") is not True:
        _debug(debug, "STOP", "keyword extractor requires clarification")
        return {
            "success": False,
            "stage": "clarification_required",
            "clarification_question": keyword_result.get("clarification_question"),
            "missing_fields": keyword_result.get("missing_fields") or [],
        }

    try:
        group_members = group_members_from_keyword_result(keyword_result)
        context = execute_search_plan(
            search_plan,
            catalog,
            store,
            question=question,
            group_members=group_members,
            debug=debug,
        )
    except Exception as exc:
        _debug(debug, "ERROR", f"{type(exc).__name__}: {exc}")
        if raise_on_error:
            raise
        return {
            "success": False,
            "stage": "search_execution_error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    return {
        "success": True,
        "stage": "context_ready",
        "context": context.to_dict(),
    }
