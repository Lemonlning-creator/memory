from __future__ import annotations

import configparser
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI

from .prompts.templates import MID_TERM_MEMORY_SYSTEM_PROMPT, MID_TERM_MEMORY_USER_PROMPT_TEMPLATE, LONG_TERM_MEMORY_SYSTEM_PROMPT, LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE
from .utils import parse_json

class MemoryOSLocal:

    def __init__(
        self,
        collection_name: str = "memoryos_local",
        persist_path: str = "./data/chroma_memory_data",
        config_path: str = "config.ini",
        embedding_model_name: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_path = Path(persist_path)
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        api_config = config["API"]
        self.embedding_model_name = embedding_model_name or api_config.get("embedding_model", fallback="text-embedding-v4")
        self.embedding_client = OpenAI(
            api_key=api_config.get("api_key"),
            base_url=api_config.get("base_url"),
        )
        self.short_term_memory: List[Dict[str, Any]] = []
        self.summary_prune_messages = 10
        self.long_term_trigger_summaries = 3
        self.long_term_source_summaries = 5
        
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Local memory store for dialogue memories",
                "created_at": datetime.now().isoformat(),
            },
        )
        self.last_long_term_count = self._get_last_long_term_count()

    # ---------- API 嵌入模型，文本向量化 ----------
    def _embed_text(self, text: str) -> List[float]:
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model_name,
            input=text,
        )
        return list(response.data[0].embedding)

    def _upsert_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding_text: Optional[str] = None,
    ) -> None:
        existing = self.collection.get(ids=[doc_id], include=[])
        if existing.get("ids"):
            self.collection.delete(ids=[doc_id])
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[self._embed_text(embedding_text or content)],
        )

    # ---------- 短期记忆：仅存内存 ----------
    def append_stm(self, role: str, content: str) -> Dict[str, Any]:
        entry = {
            "id": f"stm_{len(self.short_term_memory) + 1}_{role}_{datetime.now().timestamp()}",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.short_term_memory.append(entry)
        return entry

    def get_recent_messages(self, limit: int = 14) -> List[Dict[str, Any]]:
        return self.short_term_memory[-limit:]

    def clear_stm(self) -> None:
        self.short_term_memory.clear()

    # ---------- 提炼中期记忆、长期记忆到向量数据库 ----------    
    def add_mid_term_summary(
        self,
        topic: str,
        summary: str,
        related_states: Optional[List[str]] = None,
        related_messages: Optional[List[str]] = None,
        importance: str = ""
    ) -> str:
        timestamp = datetime.now().isoformat()
        memory_id = f"mtm_{datetime.now().timestamp()}"
        summary_record = json.dumps(
            {
                "summary": summary,
                "related_states": list(related_states or []),
                "related_messages": list(related_messages or []),
            },
            ensure_ascii=False,
        )
        self._upsert_document(
            memory_id,
            summary_record,
            {
                "memory_type": "mid_term",
                "topic": topic,
                "importance": importance,
                "created_at": timestamp,
            },
            embedding_text=summary,
        )
        return memory_id

    def build_mid_term_summary(self, llm, summary_source_messages: int) -> str:
        source_messages = self.short_term_memory[: summary_source_messages]
        source_message_map = {message["id"]: f'{message["role"]}: {message["content"]}' for message in source_messages}

        summary_result = parse_json(
            llm.chat(
                MID_TERM_MEMORY_SYSTEM_PROMPT,
                MID_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(source_message_map=source_message_map),
            )
        )
        related_message_ids = summary_result.get("related_message_ids", [])
        related_messages = [
            source_message_map[message_id]
            for message_id in related_message_ids
            if message_id in source_message_map
        ]
        memory_id = self.add_mid_term_summary(
            topic = summary_result.get("topic", ""),
            summary = summary_result.get("summary", ""),
            related_states = summary_result.get("related_states", []),
            related_messages = related_messages,
            importance = summary_result.get("importance", "medium")
        )
        self.short_term_memory = self.short_term_memory[self.summary_prune_messages:]
        return memory_id

    def add_long_term_memory(
        self,
        content: str,
        memory_kind: str,
        confidence: float = 0.0,
        source_summary_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        timestamp = datetime.now().isoformat()
        memory_id = f"ltm_{datetime.now().timestamp()}"
        long_term_record = json.dumps(
            {
                "type": memory_kind,
                "content": content,
                "source_summary_ids": list(source_summary_ids or []),
            },
            ensure_ascii=False
        )
        self._upsert_document(
            memory_id,
            long_term_record,
            {
                "memory_type": "long_term",
                "kind": memory_kind,
                "confidence": float(confidence),
                "created_at": timestamp,
                **dict(metadata or {}),
            },
        )
        return memory_id

    def extract_long_term_memory(self, llm) -> Optional[str]:
        mid_terms = self._get_memories_by_type("mid_term")
        new_count = len(mid_terms) - self.last_long_term_count
        if len(mid_terms) < 3 or new_count < self.long_term_trigger_summaries:
            return None

        source_mid_terms = mid_terms[-self.long_term_source_summaries:]
        print("采用当前中期记忆去生成长期记忆" + source_mid_terms)
        result = parse_json(llm.chat(
            LONG_TERM_MEMORY_SYSTEM_PROMPT,
            LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                mid_term_summaries=json.dumps(source_mid_terms, ensure_ascii=False, indent=2)
            ),
        ))
        print("生成的长期记忆结果" + str(result))

        self.last_long_term_count = len(mid_terms)
        if not result.get("content"):
            return None

        return self.add_long_term_memory(
            content=result.get("content", ""),
            memory_kind=result.get("type", ""),
            confidence=result.get("confidence", 0.0),
            source_summary_ids=[m["id"] for m in source_mid_terms],
            metadata={"mid_term_count": self.last_long_term_count},
        )
    
    # ---------- 智能检索相关记忆 ----------
    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where = {"memory_type": memory_type} if memory_type else None
        results = self.collection.query(
            query_embeddings=[self._embed_text(query)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        memories: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])
        if ids and ids[0]:
            for i, memory_id in enumerate(ids[0]):
                distance = results["distances"][0][i]
                memories.append(
                    {
                        "id": memory_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - distance,
                    }
                )
        return memories

    def retrieve_relevant_memory(
        self,
        user_input: str,
        recent_limit: int = 6,
        mid_term_limit: int = 3,
    ) -> Dict[str, Any]:
        return {
            "recent_messages": self.get_recent_messages(limit=recent_limit),
            "mid_term_summaries": self.search_memories(user_input, top_k=mid_term_limit, memory_type="mid_term"),
        }

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        if not memory_ids:
            return []

        seen_ids = set()
        results: List[Dict[str, Any]] = []

        stm_map = {message["id"]: message for message in self.short_term_memory}
        for memory_id in memory_ids:
            if memory_id in stm_map and memory_id not in seen_ids:
                message = stm_map[memory_id]
                results.append(
                    {
                        "id": memory_id,
                        "content": message.get("content", ""),
                        "metadata": {
                            "memory_type": "short_term",
                            "role": message.get("role", ""),
                            "timestamp": message.get("timestamp", ""),
                            **message.get("metadata", {}),
                        },
                    }
                )
                seen_ids.add(memory_id)

        vector_ids = [memory_id for memory_id in memory_ids if memory_id not in seen_ids]
        if vector_ids:
            query_result = self.collection.get(
                ids=vector_ids,
                include=["documents", "metadatas"],
            )
            for i, memory_id in enumerate(query_result.get("ids", [])):
                if memory_id in seen_ids:
                    continue
                results.append(
                    {
                        "id": memory_id,
                        "content": query_result["documents"][i],
                        "metadata": query_result["metadatas"][i],
                    }
                )
                seen_ids.add(memory_id)

        result_map = {item["id"]: item for item in results}
        return [result_map[memory_id] for memory_id in memory_ids if memory_id in result_map]

    def _get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        query_result = self.collection.get(
            where={"memory_type": memory_type},
            include=["documents", "metadatas"],
        )
        memories = [
            {
                "id": memory_id,
                "content": query_result["documents"][i],
                "metadata": query_result["metadatas"][i],
            }
            for i, memory_id in enumerate(query_result.get("ids", []))
        ]
        return sorted(memories, key=lambda item: item["metadata"].get("created_at", ""))
    
    def _get_last_long_term_count(self) -> int:
        long_term_memories = self._get_memories_by_type("long_term")
        counts = [
            int(memory["metadata"].get("mid_term_count", 0))
            for memory in long_term_memories
            if memory["metadata"].get("mid_term_count") is not None
        ]
        return max(counts, default=0)
