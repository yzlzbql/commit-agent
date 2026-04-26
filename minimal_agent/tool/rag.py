from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field
import requests

from ..types import ToolCtx, ToolResult, ToolSpec


_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9_/]+")
_SILICONFLOW_URL = "https://api.siliconflow.cn/v1/embeddings"
_CHROMA_STORAGE_PATH = ".cache/chroma_db"


@dataclass(frozen=True, slots=True)
class _BackendConfig:
    corpus: str
    DEFAULT_CONFIG: dict[str, Any]


_RAW_BACKEND = _BackendConfig(
    corpus="raw",
    DEFAULT_CONFIG={
        "chroma_storage_path": _CHROMA_STORAGE_PATH,
        "collection_name": "commit_patch_cci",
        "siliconflow_url": _SILICONFLOW_URL,
        "siliconflow_token": "",
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "embedding_dim": 1024,
        "llm_model": "deepseek-chat",
        "top_k": 5,
        "recall_k": 50,
        "alpha": 0.5,
    },
)

_SUMMARY_BACKEND = _BackendConfig(
    corpus="summary",
    DEFAULT_CONFIG={
        "chroma_storage_path": _CHROMA_STORAGE_PATH,
        "collection_name": "commit_cci_embeddings",
        "siliconflow_url": _SILICONFLOW_URL,
        "siliconflow_token": "",
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "embedding_dim": 1024,
        "llm_model": "deepseek-chat",
        "top_k": 5,
        "recall_k": 50,
        "alpha": 0.5,
    },
)


class RAGArgs(BaseModel):
    query_file: str = Field(default="", description="Batch mode input JSONL path")
    output_file: str = Field(default="", description="Batch mode output JSONL path")
    commit_data: dict[str, Any] | None = Field(
        default=None,
        description="Single-commit mode input. Prefer this for commit classification by passing commit_message and patch directly instead of creating a query file",
    )
    corpus: str = Field(default="raw", description="RAG corpus backend: raw/original or summary/cci; default raw uses collection commit_patch_cci")
    collection_name: str | None = Field(default=None, description="Override Chroma collection name")
    top_k: int = Field(default=5, ge=1, le=50, description="Final number of retrieved commits; default 5")
    recall_k: int = Field(default=50, ge=1, le=500, description="Initial vector recall size before reranking; default 50")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid rerank weight for vector similarity; default 0.5")


def spec() -> ToolSpec:
    return ToolSpec(
        name="rag",
        description=(
            "Retrieve similar historical commits from the external RAG corpus for a single commit or a batch query file. "
            "For commit classification, use single-commit mode with commit_data first; default corpus is raw with top_k=5, "
            "recall_k=50, alpha=0.5. Treat returned matches as supporting evidence rather than the sole basis for the final label."
        ),
        input_model=RAGArgs,
        execute=run,
    )


def run(ctx: ToolCtx, args: RAGArgs) -> ToolResult:
    corpus = _normalize_corpus(args.corpus)
    backend = _load_backend(corpus)
    storage_path = _resolve_storage_path(ctx.project.root, backend)
    collection_name = (args.collection_name or "").strip() or str(backend.DEFAULT_CONFIG["collection_name"])
    top_k = args.top_k
    recall_k = max(args.recall_k, top_k)
    alpha = args.alpha

    has_commit = bool(args.commit_data)
    has_query_file = bool(args.query_file.strip())
    if has_commit == has_query_file:
        raise ValueError("E_INPUT_INVALID: provide exactly one of commit_data or query_file")

    if has_commit:
        commit_data = _sanitize_value(dict(args.commit_data or {}))
        result = _retrieve_commit(
            storage_path=storage_path,
            backend=backend,
            commit_data=commit_data,
            collection_name=collection_name,
            top_k=top_k,
            recall_k=recall_k,
            alpha=alpha,
            corpus=corpus,
        )
        predicted_tags = result["predicted_tags"]
        lines = [
            "RAG process completed successfully!",
            "",
            f"Corpus: {corpus}",
            f"Collection: {collection_name}",
            "",
            "Commit Analysis Results:",
            f"- Retrieval completed with {len(result['similar_commits'])} results",
            f"- Predicted labels: {', '.join(predicted_tags) if predicted_tags else '-'}",
        ]
        if result["generated_intent"]:
            lines.extend(["", "Generated Intent:", result["generated_intent"]])
        return ToolResult(
            title="RAG Commit Analysis",
            output="\n".join(lines),
            metadata={
                "result": result,
                "corpus": corpus,
                "collection_name": collection_name,
                "predicted_tags": predicted_tags,
                "similar_commits": result["similar_commits"],
            },
        )

    query_path = _resolve_path(ctx, args.query_file)
    if not query_path.exists():
        raise ValueError(f"E_PATH_NOT_FOUND: query file not found: {query_path}")
    if not query_path.is_file():
        raise ValueError(f"E_INPUT_INVALID: query_file is not a file: {query_path}")
    if not args.output_file.strip():
        raise ValueError("E_INPUT_INVALID: output_file is required when query_file is provided")
    output_path = _resolve_path(ctx, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    summary = _run_batch(
        storage_path=storage_path,
        backend=backend,
        query_path=query_path,
        output_path=output_path,
        summary_path=summary_path,
        collection_name=collection_name,
        top_k=top_k,
        recall_k=recall_k,
        alpha=alpha,
        corpus=corpus,
    )
    retrieval = summary["evaluation"]["retrieval"]
    classification = summary["evaluation"]["classification"]
    lines = [
        "RAG process completed successfully!",
        "",
        f"Corpus: {corpus}",
        f"Collection: {collection_name}",
        f"Query file: {query_path}",
        f"Output file: {output_path}",
        f"Summary file: {summary_path}",
        "",
        "Evaluation Results:",
        f"- Total queries: {retrieval['total_queries']}",
    ]
    if retrieval["queries_with_gt_sha"]:
        lines.extend(
            [
                f"- Queries with gt_sha: {retrieval['queries_with_gt_sha']}",
                f"- Top-1 accuracy: {retrieval['top_1_accuracy']['percentage']:.2f}%",
                f"- Top-3 accuracy: {retrieval['top_3_accuracy']['percentage']:.2f}%",
                f"- Top-5 accuracy: {retrieval['top_5_accuracy']['percentage']:.2f}%",
            ]
        )
    if classification["valid_predictions"]:
        lines.extend(
            [
                f"- Classification valid predictions: {classification['valid_predictions']}",
                f"- Classification accuracy: {classification['overall_accuracy']:.2f}%",
            ]
        )
    return ToolResult(
        title="RAG Batch Analysis",
        output="\n".join(lines),
        metadata={
            "corpus": corpus,
            "collection_name": collection_name,
            "query_file": str(query_path),
            "output_file": str(output_path),
            "summary_file": str(summary_path),
            "evaluation": summary["evaluation"],
            "files": [str(output_path), str(summary_path)],
        },
    )


def _normalize_corpus(raw: str) -> str:
    norm = (raw or "raw").strip().lower()
    if norm in {"raw", "origin", "original", "patch"}:
        return "raw"
    if norm in {"summary", "cci"}:
        return "summary"
    raise ValueError("E_INPUT_INVALID: corpus must be one of raw/original/summary/cci")


def _load_backend(corpus: str) -> Any:
    return _SUMMARY_BACKEND if corpus == "summary" else _RAW_BACKEND


def _resolve_storage_path(project_root: Path, backend: _BackendConfig) -> Path:
    raw = Path(str(backend.DEFAULT_CONFIG["chroma_storage_path"]))
    if raw.is_absolute():
        return raw
    return (project_root / raw).resolve()


def _env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _env_int(*names: str) -> int | None:
    raw = _env_value(*names)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as err:
        joined = "/".join(names)
        raise ValueError(f"E_INPUT_INVALID: {joined} must be an integer") from err


def _embedding_model(backend: _BackendConfig, override: str | None = None) -> str:
    return (
        override
        or _env_value("RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL")
        or str(backend.DEFAULT_CONFIG["embedding_model"])
    ).strip()


def _embedding_dim(backend: _BackendConfig, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return _env_int("RAG_EMBEDDING_DIM", "EMBEDDING_DIM") or int(
        backend.DEFAULT_CONFIG["embedding_dim"]
    )


def _get_embeddings(
    texts: list[str],
    *,
    backend: _BackendConfig,
    model: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
) -> list[list[float] | None]:
    config = backend.DEFAULT_CONFIG
    token = (
        api_key
        or _env_value("SILICONFLOW_API_KEY", "EMBEDDING_API_KEY")
        or str(config["siliconflow_token"]).strip()
    )
    if not token:
        raise RuntimeError("E_RAG_EMBEDDING_FAILED: set SILICONFLOW_API_KEY in .env")

    payload_model = _embedding_model(backend, model)
    payload_dimensions = _embedding_dim(backend, dimensions)
    url = _env_value("SILICONFLOW_EMBEDDING_URL", "RAG_EMBEDDING_URL") or str(
        config["siliconflow_url"]
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    results: list[list[float] | None] = []
    for text in texts:
        payload = {
            "model": payload_model,
            "input": str(text).replace("\n", " "),
            "encoding_format": "float",
            "dimensions": payload_dimensions,
        }
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json().get("data")
            if not data:
                raise RuntimeError("embedding response did not contain data")
            embedding = data[0].get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("embedding response did not contain a numeric vector")
            results.append(embedding)
        except Exception:
            results.append(None)
    return results


def _run_batch(
    *,
    storage_path: Path,
    backend: Any,
    query_path: Path,
    output_path: Path,
    summary_path: Path,
    collection_name: str,
    top_k: int,
    recall_k: int,
    alpha: float,
    corpus: str,
) -> dict[str, Any]:
    total = 0
    with_gt_sha = 0
    retrieval_hits = {1: 0, 3: 0, 5: 0}
    valid_predictions = 0
    no_predictions = 0
    correct_predictions = 0

    with query_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"E_INPUT_INVALID: invalid JSON on line {line_no}: {err}") from err

            commit_data = _commit_data_from_query_item(item, line_no=line_no)
            result = _retrieve_commit(
                storage_path=storage_path,
                backend=backend,
                commit_data=commit_data,
                collection_name=collection_name,
                top_k=top_k,
                recall_k=recall_k,
                alpha=alpha,
                corpus=corpus,
            )

            predicted_tags = result["predicted_tags"]
            gt_tag = str(item.get("gt_tag", "")).strip().lower()
            gt_sha = str(item.get("gt_sha", "")).strip()
            retrieved_ids = result["retrieve_results"]["ids"][0]
            is_correct = bool(gt_tag and predicted_tags and gt_tag in predicted_tags)

            total += 1
            if predicted_tags:
                valid_predictions += 1
                if is_correct:
                    correct_predictions += 1
            else:
                no_predictions += 1

            if gt_sha:
                with_gt_sha += 1
                for k in (1, 3, 5):
                    if gt_sha in retrieved_ids[: min(k, len(retrieved_ids))]:
                        retrieval_hits[k] += 1

            record = dict(item)
            record["retrieve_results"] = result["retrieve_results"]
            record["context"] = result["context"]
            record["generated_intent"] = result["generated_intent"]
            record["predicted_tags"] = predicted_tags
            record["predicted_primary_tag"] = predicted_tags[0] if predicted_tags else ""
            record["classification_match"] = is_correct if gt_tag else None
            record["rag_metadata"] = {
                "corpus": corpus,
                "collection_name": collection_name,
                "top_k": top_k,
                "recall_k": recall_k,
                "alpha": alpha,
            }
            dst.write(json.dumps(record, ensure_ascii=False))
            dst.write("\n")

    retrieval_eval = {
        "total_queries": total,
        "queries_with_gt_sha": with_gt_sha,
        "top_1_accuracy": _accuracy_entry(retrieval_hits[1], with_gt_sha),
        "top_3_accuracy": _accuracy_entry(retrieval_hits[3], with_gt_sha),
        "top_5_accuracy": _accuracy_entry(retrieval_hits[5], with_gt_sha),
    }
    classification_eval = {
        "total_queries": total,
        "valid_predictions": valid_predictions,
        "no_predictions": no_predictions,
        "correct_predictions": correct_predictions,
        "overall_accuracy": round((correct_predictions / total * 100), 2) if total else 0.0,
        "valid_accuracy": round((correct_predictions / valid_predictions * 100), 2) if valid_predictions else 0.0,
    }
    summary = {
        "experiment_name": f"RAG Retrieval - {'Summary Corpus' if corpus == 'summary' else 'Raw Corpus'}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "corpus": corpus,
        "collection_name": collection_name,
        "query_file": str(query_path),
        "result_file": str(output_path),
        "evaluation": {
            "retrieval": retrieval_eval,
            "classification": classification_eval,
        },
        "configuration": {
            "chroma_storage_path": str(storage_path),
            "embedding_model": _embedding_model(backend),
            "embedding_dim": _embedding_dim(backend),
            "llm_model": str(backend.DEFAULT_CONFIG.get("llm_model", "")),
            "retrieval": {
                "top_k": top_k,
                "recall_k": recall_k,
                "alpha": alpha,
                "method": "hybrid (vector + keyword)",
            },
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _retrieve_commit(
    *,
    storage_path: Path,
    backend: Any,
    commit_data: dict[str, Any],
    collection_name: str,
    top_k: int,
    recall_k: int,
    alpha: float,
    corpus: str,
) -> dict[str, Any]:
    try:
        import chromadb
    except Exception as err:
        raise RuntimeError(f"E_RAG_UNAVAILABLE: chromadb is required for rag tool: {err}") from err

    query_text = _build_query_text(commit_data)
    query_embedding = _get_embeddings(
        [query_text],
        backend=backend,
    )[0]
    if query_embedding is None:
        raise RuntimeError("E_RAG_EMBEDDING_FAILED: failed to generate embedding for query commit")

    client = chromadb.PersistentClient(path=str(storage_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception as err:
        raise RuntimeError(f"E_RAG_COLLECTION_NOT_FOUND: failed to open collection {collection_name}: {err}") from err

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=recall_k,
        include=["documents", "metadatas", "distances"],
    )
    retrieve_results = _rerank_results(query_text, raw_results, top_k=top_k, alpha=alpha)
    similar_commits = _similar_commits_from_results(retrieve_results)
    predicted_tags = _predicted_tags(similar_commits)
    generated_intent = _build_generated_intent(predicted_tags, similar_commits, collection_name=collection_name, corpus=corpus)
    return {
        "success": True,
        "retrieve_results": retrieve_results,
        "similar_commits": similar_commits,
        "predicted_tags": predicted_tags,
        "generated_intent": generated_intent,
        "query": query_text,
        "context": _build_context(commit_data, similar_commits),
    }


def _rerank_results(query_text: str, raw_results: dict[str, Any], *, top_k: int, alpha: float) -> dict[str, Any]:
    ids = list((raw_results.get("ids") or [[]])[0])
    docs = list((raw_results.get("documents") or [[]])[0])
    metas = list((raw_results.get("metadatas") or [[]])[0])
    dists = list((raw_results.get("distances") or [[]])[0])

    query_counter = Counter(_tokenize(query_text))
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw_id in enumerate(ids):
        item_id = str(raw_id)
        if item_id in seen:
            continue
        seen.add(item_id)
        doc = str(docs[idx] if idx < len(docs) else "")
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        dist_raw = dists[idx] if idx < len(dists) else None
        dist = float(dist_raw) if dist_raw is not None else 1.0
        vector_sim = 1.0 - dist
        keyword_sim = _cosine_sim(query_counter, Counter(_tokenize(doc)))
        hybrid_score = alpha * vector_sim + (1.0 - alpha) * keyword_sim
        ranked.append(
            {
                "id": item_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "vector_sim": vector_sim,
                "keyword_sim": keyword_sim,
                "hybrid_score": hybrid_score,
            }
        )
    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    top = ranked[:top_k]
    return {
        "ids": [[item["id"] for item in top]],
        "documents": [[item["document"] for item in top]],
        "metadatas": [[item["metadata"] for item in top]],
        "distances": [[item["distance"] for item in top]],
    }


def _similar_commits_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    ids = list((results.get("ids") or [[]])[0])
    docs = list((results.get("documents") or [[]])[0])
    metas = list((results.get("metadatas") or [[]])[0])
    dists = list((results.get("distances") or [[]])[0])
    items = []
    for idx, item_id in enumerate(ids):
        items.append(
            {
                "id": str(item_id),
                "commit": str(docs[idx] if idx < len(docs) else ""),
                "distance": float(dists[idx]) if idx < len(dists) and dists[idx] is not None else None,
                "metadata": metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {},
            }
        )
    return items


def _predicted_tags(similar_commits: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    order: dict[str, int] = {}
    for idx, item in enumerate(similar_commits):
        meta = item.get("metadata") or {}
        raw = str(meta.get("tag", "")).strip().lower()
        if not raw:
            continue
        counts[raw] += 1
        order.setdefault(raw, idx)
    return sorted(counts, key=lambda tag: (-counts[tag], order[tag]))[:3]


def _build_generated_intent(
    predicted_tags: list[str],
    similar_commits: list[dict[str, Any]],
    *,
    collection_name: str,
    corpus: str,
) -> str:
    tag_counts = Counter(
        str((item.get("metadata") or {}).get("tag", "")).strip().lower()
        for item in similar_commits
        if str((item.get("metadata") or {}).get("tag", "")).strip()
    )
    summary = (
        f"Retrieved {len(similar_commits)} similar commits from collection {collection_name} "
        f"using the {corpus} corpus."
    )
    if predicted_tags:
        reasoning = "Top retrieved labels by frequency were: " + ", ".join(
            f"{tag}({tag_counts[tag]})" for tag in predicted_tags
        )
    else:
        reasoning = "Retrieved commits did not expose label metadata, so no label suggestion was produced."
    payload = {
        "summary": summary,
        "classification": predicted_tags,
        "reasoning": reasoning,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _build_context(commit_data: dict[str, Any], similar_commits: list[dict[str, Any]]) -> str:
    lines = [
        "### [QUERY COMMIT]",
        f"Commit Message:\n{str(commit_data.get('commit_message', '')).strip()}",
        "",
        f"Patch:\n{str(commit_data.get('patch', '')).strip()}",
    ]
    if similar_commits:
        lines.extend(["", "### [HISTORICAL SIMILAR COMMITS]"])
    for idx, item in enumerate(similar_commits, start=1):
        meta = item.get("metadata") or {}
        dist = item.get("distance")
        lines.extend(
            [
                "",
                f"--- Commit #{idx} ---",
                f"SHA: {item.get('id', '')}",
                f"Title: {meta.get('title', '')}",
                f"Tags: {meta.get('tag', '')}",
                f"Repo: {meta.get('repo', '')}",
                f"Distance: {dist:.4f}" if isinstance(dist, float) else "Distance: -",
                "Patch Context:",
                str(item.get("commit", "")).strip(),
            ]
        )
    return "\n".join(lines).strip()


def _commit_data_from_query_item(item: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    if isinstance(item.get("commit_data"), dict):
        return _sanitize_value(dict(item["commit_data"]))

    query_commit = item.get("query_commit")
    if isinstance(query_commit, dict):
        commit_message = str(
            query_commit.get("message")
            or query_commit.get("body")
            or query_commit.get("title")
            or item.get("commit_message")
            or ""
        )
        patch = str(query_commit.get("patch") or item.get("patch") or "")
    else:
        commit_message = str(item.get("commit_message") or item.get("message") or item.get("title") or "")
        patch = str(item.get("patch") or "")

    if not patch.strip():
        raise ValueError(f"E_INPUT_INVALID: query item on line {line_no} does not contain patch text")
    data = {
        "commit_message": commit_message,
        "patch": patch,
    }
    gt_tag = str(item.get("gt_tag", "")).strip()
    if gt_tag:
        data["oracle_label"] = gt_tag
    return _sanitize_value(data)


def _build_query_text(commit_data: dict[str, Any]) -> str:
    message = str(commit_data.get("commit_message", "") or "").strip()
    patch = str(commit_data.get("patch", "") or "").strip()
    return f"Commit message: {message}\n\nPatch: {patch}"


def _resolve_path(ctx: ToolCtx, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    base = Path.cwd()
    if getattr(ctx, "project", None) is not None and getattr(ctx.project, "cwd", None) is not None:
        base = Path(ctx.project.cwd)
    return (base / path).resolve()


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(ch for ch in value if ord(ch) >= 32 or ch in "\n\t")
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _tokenize(text: str) -> list[str]:
    return _TOKEN_SPLIT.sub(" ", text or "").split()


def _cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)
    norm_a = math.sqrt(sum(val * val for val in a.values()))
    norm_b = math.sqrt(sum(val * val for val in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _accuracy_entry(hits: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"percentage": 0.0, "hits": 0, "total": 0}
    return {"percentage": round(hits / total * 100, 2), "hits": hits, "total": total}
