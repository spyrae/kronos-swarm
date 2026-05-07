"""Mem0 memory store — self-hosted, no paid APIs required.

Uses DeepSeek for fact extraction + HuggingFace for embeddings (local).
Qdrant in local mode (in-process, no external server).
FTS5 keyword index runs in parallel for hybrid search.
"""

import logging
from functools import lru_cache

from kronos.config import settings
from kronos.memory import fts
from kronos.memory.hybrid import merge_hybrid_results
from kronos.security.pii import mask_pii

log = logging.getLogger("kronos.memory")


@lru_cache(maxsize=1)
def get_memory():
    """Get or create Mem0 Memory instance (singleton)."""
    from mem0 import Memory

    config = {
        "version": "v1.1",
    }

    # LLM for fact extraction — DeepSeek (cheap: $0.27/1M tokens)
    if settings.deepseek_api_key:
        config["llm"] = {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "api_key": settings.deepseek_api_key,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        }

    # Embeddings — HuggingFace local model (free, no API key)
    config["embedder"] = {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1",
            # 384 dimensions, good for semantic search
        },
    }

    # Vector store — Qdrant local mode, per-agent collection to prevent
    # cross-agent contamination. Historical default was a single shared
    # "kronos_memories" collection which corrupted on concurrent writes.
    collection_name = f"{settings.agent_name}_memories"
    config["vector_store"] = {
        "provider": "qdrant",
        "config": {
            "collection_name": collection_name,
            "path": settings.mem0_qdrant_path,
            "embedding_model_dims": 384,
        },
    }

    log.info(
        "Initializing Mem0: qdrant=%s, collection=%s, llm=deepseek, embedder=huggingface/multi-qa-MiniLM-L6-cos-v1",
        settings.mem0_qdrant_path, collection_name,
    )

    return Memory.from_config(config)



def search_memories(query: str, user_id: str, limit: int = 5) -> list[str]:
    """Hybrid search: vector (Mem0/Qdrant) + keyword (FTS5) with merge.

    Both searches run, results are merged with score normalization,
    temporal decay, and MMR re-ranking for diversity.
    """
    vector_results = []
    fts_results = []

    # 1. Vector search via Mem0
    try:
        mem = get_memory()
        results = mem.search(query, user_id=user_id, limit=limit * 3)
        vector_results = results.get("results", [])
    except Exception as e:
        log.error("Vector search failed: %s", e)

    # 2. Keyword search via FTS5
    try:
        fts_results = fts.search(query, user_id=user_id, limit=limit * 3)
    except Exception as e:
        log.error("FTS5 search failed: %s", e)

    if not vector_results and not fts_results:
        return []

    # 3. Merge with hybrid scoring + MMR
    memories = merge_hybrid_results(
        vector_results=vector_results,
        fts_results=fts_results,
        limit=limit,
    )

    if memories:
        log.info(
            "Hybrid search for user %s: %d vector + %d fts → %d merged",
            user_id, len(vector_results), len(fts_results), len(memories),
        )
        # Touch accessed facts (Ebbinghaus: boost relevance on access)
        fts.touch_facts(memories, user_id)

    return memories


def add_memories(
    messages: list[dict],
    user_id: str,
    session_id: str | None = None,
) -> list[str]:
    """Store conversation turn in memory (extracts facts via LLM).

    Facts are stored in both Mem0 (vector) and FTS5 (keyword) indexes.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
        user_id: User identifier for scoping
        session_id: Optional session identifier

    Returns:
        List of extracted fact strings (possibly empty). Callers use this
        to mirror user-sourced facts into shared cross-agent storage.
    """
    facts: list[str] = []
    try:
        mem = get_memory()
        kwargs = {"user_id": user_id}
        if session_id:
            kwargs["metadata"] = {"session_id": mask_pii(session_id)}

        result = mem.add(messages, **kwargs)
        extracted = result.get("results", [])
        added = len(extracted)

        if added:
            log.info("Stored %d memories for user %s", added, user_id)

            # Index extracted facts into FTS5 in parallel
            facts = [item.get("memory", "") for item in extracted if item.get("memory")]
            if facts:
                indexed = fts.index_facts_batch(facts, user_id)
                log.info("FTS5 indexed %d facts for user %s", indexed, user_id)

    except Exception as e:
        log.error("Memory add failed: %s", e)
    return facts


def get_all_memories(user_id: str) -> list[str]:
    """Get all memories for a user."""
    try:
        mem = get_memory()
        result = mem.get_all(user_id=user_id)
        return [item.get("memory", "") for item in result.get("results", []) if item.get("memory")]
    except Exception as e:
        log.error("Memory get_all failed: %s", e)
        return []
