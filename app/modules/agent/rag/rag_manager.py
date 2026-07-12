from __future__ import annotations

import argparse
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import EmbeddingSettings, embedding_settings

logger = logging.getLogger(__name__)

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = RAG_DIR / "chroma_db"
COLLECTION_NAME = "rag_policy_items"
TOP_K = 5

ROAD_TRAFFIC_SAFETY_LAW = "中华人民共和国道路交通安全法"
ROAD_TRAFFIC_SAFETY_REGULATION = "中华人民共和国道路交通安全法实施条例"

_CHINESE_NUMBER = "零〇一二三四五六七八九十百千万两0-9"
_CHAPTER_RE = re.compile(rf"^\s*(第[{_CHINESE_NUMBER}]+章)(?:[\s\u3000]+.*)?$")
_ARTICLE_RE = re.compile(rf"^\s*(第[{_CHINESE_NUMBER}]+条)[\s\u3000]*(.*)$")
_SECTION_RE = re.compile(rf"^\s*第[{_CHINESE_NUMBER}]+节(?:[\s\u3000]+.*)?$")
_CHAPTER_FROM_REFERENCE_RE = re.compile(rf"^(第[{_CHINESE_NUMBER}]+章)")
_ARTICLE_FROM_REFERENCE_RE = re.compile(rf"第[{_CHINESE_NUMBER}]+条$")
_INDEX_LOCK = threading.RLock()


class rag_policy_item(BaseModel):
    """一条法规及其检索向量。"""

    describe: str
    des_vector: list[float] = Field(default_factory=list, exclude=True)
    chapter: str


# 同时提供符合 Python 类命名习惯的别名。
RagPolicyItem = rag_policy_item


@dataclass(frozen=True)
class _PolicyDatabase:
    title: str
    source_file: Path
    database_dir: Path


POLICY_DATABASES = (
    _PolicyDatabase(
        title=ROAD_TRAFFIC_SAFETY_LAW,
        source_file=RAG_DIR / f"{ROAD_TRAFFIC_SAFETY_LAW}.txt",
        database_dir=CHROMA_DB_DIR / "road_traffic_safety_law",
    ),
    _PolicyDatabase(
        title=ROAD_TRAFFIC_SAFETY_REGULATION,
        source_file=RAG_DIR / f"{ROAD_TRAFFIC_SAFETY_REGULATION}.txt",
        database_dir=CHROMA_DB_DIR / "road_traffic_safety_regulation",
    ),
)


class rag_manager:
    """管理两部交通法规的 ChromaDB 向量索引。"""

    def __init__(
        self,
        config: EmbeddingSettings | None = None,
        embedding_client: Any | None = None,
    ) -> None:
        self._config = config or embedding_settings
        self._embedding_client = embedding_client
        self._collections: dict[str, Any] = {}

    def search_rag(self, search_str: str) -> dict[str, list[dict[str, str]]]:
        """分别返回两部法规中与输入语义最相近的五条记录。

        返回值是可直接由 FastAPI/JSON 序列化的字典，不包含向量字段。
        """

        prompt = search_str.strip()
        if not prompt:
            raise ValueError("search_str 不能为空")

        self._ensure_databases()
        query_vector = self._embed_texts([prompt])[0]

        return {
            policy.title: self._query_collection(
                collection=self._collections[policy.title],
                query_vector=query_vector,
            )
            for policy in POLICY_DATABASES
        }

    def initialize_databases(self, force_rebuild: bool = False) -> dict[str, int]:
        """显式初始化法规库，返回每个数据库中的法条数量。"""

        self._ensure_databases(force_rebuild=force_rebuild)
        return {
            title: int(collection.count())
            for title, collection in self._collections.items()
        }

    @classmethod
    def clean_source_files(cls) -> dict[str, int]:
        """将原始法规文本规范成一条法条一行的 UTF-8 文本。"""

        counts: dict[str, int] = {}
        for policy in POLICY_DATABASES:
            source_text = policy.source_file.read_text(encoding="utf-8-sig")
            items = cls.parse_policy_items(source_text)
            policy.source_file.write_text(
                cls._render_clean_text(items),
                encoding="utf-8",
                newline="\n",
            )
            counts[policy.title] = len(items)
        return counts

    @classmethod
    def parse_policy_items(cls, text: str) -> list[rag_policy_item]:
        """把法规正文解析成 `rag_policy_item` 列表。"""

        items: list[rag_policy_item] = []
        current_chapter: str | None = None
        current_article: str | None = None
        description_parts: list[str] = []

        def flush_article() -> None:
            nonlocal current_article, description_parts
            if current_chapter and current_article:
                description = " ".join(description_parts).strip()
                if not description:
                    raise ValueError(f"{current_chapter}{current_article} 没有正文")
                items.append(
                    rag_policy_item(
                        describe=description,
                        chapter=f"{current_chapter}{current_article}",
                    )
                )
            current_article = None
            description_parts = []

        for raw_line in text.splitlines():
            line = cls._normalize_line(raw_line)
            if not line or line.replace(" ", "") == "目录":
                continue

            chapter_match = _CHAPTER_RE.fullmatch(line)
            if chapter_match:
                flush_article()
                current_chapter = chapter_match.group(1)
                continue

            article_match = _ARTICLE_RE.match(line)
            if article_match:
                flush_article()
                if current_chapter is None:
                    raise ValueError(f"法条缺少所属章: {line}")
                current_article = article_match.group(1)
                first_paragraph = article_match.group(2).strip()
                if first_paragraph:
                    description_parts.append(first_paragraph)
                continue

            if _SECTION_RE.fullmatch(line):
                continue

            if current_article:
                description_parts.append(line)

        flush_article()

        if not items:
            raise ValueError("法规文本中没有解析到法条")

        references = [item.chapter for item in items]
        if len(references) != len(set(references)):
            raise ValueError("法规文本中存在重复的章条编号")
        return items

    def _ensure_databases(self, force_rebuild: bool = False) -> None:
        chromadb = self._load_chromadb()
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        with _INDEX_LOCK:
            for policy in POLICY_DATABASES:
                policy.database_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(policy.database_dir))
                source_text = policy.source_file.read_text(encoding="utf-8-sig")
                items = self.parse_policy_items(source_text)
                source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                expected_metadata = {
                    "policy_title": policy.title,
                    "source_sha256": source_hash,
                    "embedding_model": self._config.model_name,
                    "embedding_url": self._config.url.rstrip("/"),
                    "embedding_dimensions": str(self._config.dimensions),
                }

                collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata=expected_metadata,
                    configuration={"hnsw": {"space": "cosine"}},
                )
                stored_metadata = collection.metadata or {}
                index_is_current = (
                    not force_rebuild
                    and collection.count() == len(items)
                    and stored_metadata.get("source_sha256") == source_hash
                    and stored_metadata.get("embedding_model") == self._config.model_name
                    and stored_metadata.get("embedding_url")
                    == self._config.url.rstrip("/")
                    and stored_metadata.get("embedding_dimensions")
                    == str(self._config.dimensions)
                )

                if not index_is_current:
                    client.delete_collection(name=COLLECTION_NAME)
                    collection = client.create_collection(
                        name=COLLECTION_NAME,
                        metadata=expected_metadata,
                        configuration={"hnsw": {"space": "cosine"}},
                    )
                    self._populate_collection(collection, items)
                    logger.info("已重建法规向量库: %s，共 %d 条", policy.title, len(items))

                self._collections[policy.title] = collection

    def _populate_collection(
        self,
        collection: Any,
        items: Sequence[rag_policy_item],
    ) -> None:
        vectors = self._embed_texts_concurrently(
            [item.describe for item in items]
        )
        for item, vector in zip(items, vectors, strict=True):
            item.des_vector = vector

        collection.upsert(
            ids=[f"article-{index:04d}" for index in range(len(items))],
            embeddings=[item.des_vector for item in items],
            documents=[item.describe for item in items],
            metadatas=[{"chapter": item.chapter} for item in items],
        )

    def _embed_texts_concurrently(self, texts: Sequence[str]) -> list[list[float]]:
        """以配置的并发数逐条生成向量，并保持输入顺序。"""

        with ThreadPoolExecutor(max_workers=self._config.concurrency) as executor:
            return list(executor.map(self._embed_one_text, texts))

    def _embed_one_text(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_embedding_client()
        request: dict[str, Any] = {
            "model": self._config.model_name,
            "input": list(texts),
        }
        request["dimensions"] = self._config.dimensions

        response = client.embeddings.create(**request)
        ordered_data = sorted(response.data, key=lambda item: item.index)
        vectors = [[float(value) for value in item.embedding] for item in ordered_data]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding 接口返回 {len(vectors)} 条向量，预期 {len(texts)} 条"
            )
        invalid_dimensions = [
            len(vector)
            for vector in vectors
            if len(vector) != self._config.dimensions
        ]
        if invalid_dimensions:
            raise RuntimeError(
                "Embedding 向量维度不正确："
                f"期望 {self._config.dimensions}，实际 {invalid_dimensions[0]}"
            )
        return vectors

    def _get_embedding_client(self) -> Any:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                base_url=self._config.url,
                api_key=self._config.api_key,
                timeout=self._config.timeout,
            )
        return self._embedding_client

    @staticmethod
    def _query_collection(
        collection: Any,
        query_vector: list[float],
    ) -> list[dict[str, str]]:
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        output: list[dict[str, str]] = []
        for document, metadata in zip(documents, metadatas, strict=True):
            item = rag_policy_item(
                describe=str(document),
                chapter=str((metadata or {}).get("chapter", "")),
            )
            output.append(item.model_dump())
        return output

    @staticmethod
    def _normalize_line(line: str) -> str:
        normalized = line.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
        return re.sub(r"[\t ]+", " ", normalized).strip()

    @staticmethod
    def _render_clean_text(items: Iterable[rag_policy_item]) -> str:
        lines: list[str] = []
        rendered_chapter: str | None = None

        for item in items:
            chapter_match = _CHAPTER_FROM_REFERENCE_RE.match(item.chapter)
            article_match = _ARTICLE_FROM_REFERENCE_RE.search(item.chapter)
            if not chapter_match or not article_match:
                raise ValueError(f"非法章条格式: {item.chapter}")

            chapter = chapter_match.group(1)
            article = article_match.group(0)
            if chapter != rendered_chapter:
                if lines:
                    lines.append("")
                lines.extend([chapter, ""])
                rendered_chapter = chapter
            lines.append(f"{article} {item.describe}")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _load_chromadb() -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 chromadb 依赖，请先执行 pip install -r requirements.txt"
            ) from exc
        return chromadb


RagManager = rag_manager


def _main() -> None:
    parser = argparse.ArgumentParser(description="清洗法规文本并初始化 RAG 向量库")
    parser.add_argument("--clean", action="store_true", help="清洗两份法规源文本")
    parser.add_argument("--rebuild", action="store_true", help="强制重建两个 ChromaDB")
    args = parser.parse_args()

    if not args.clean and not args.rebuild:
        parser.error("请至少指定 --clean 或 --rebuild")

    if args.clean:
        print(rag_manager.clean_source_files())
    if args.rebuild:
        print(rag_manager().initialize_databases(force_rebuild=True))


if __name__ == "__main__":
    _main()
