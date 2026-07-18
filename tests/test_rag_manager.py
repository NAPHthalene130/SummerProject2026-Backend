import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import EmbeddingSettings
from app.modules.agent.rag.rag_manager import (
    POLICY_DATABASES,
    ROAD_TRAFFIC_SAFETY_LAW,
    ROAD_TRAFFIC_SAFETY_REGULATION,
    rag_manager,
    rag_policy_item,
)


class _FakeCollection:
    def __init__(self, title: str) -> None:
        self.title = title

    def query(self, **kwargs):
        assert kwargs["query_embeddings"] == [[0.1, 0.2]]
        assert kwargs["n_results"] == 5
        assert "embeddings" not in kwargs["include"]
        return {
            "documents": [[f"{self.title}描述{i}" for i in range(5)]],
            "metadatas": [[{"chapter": f"第一章第{i + 1}条"} for i in range(5)]],
        }


class _FakeEmbeddings:
    @staticmethod
    def create(**kwargs):
        assert kwargs["input"] == ["酒后驾驶如何处罚"]
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        )


class _DeterministicEmbeddings:
    def __init__(self) -> None:
        self.call_count = 0
        self.active_count = 0
        self.max_active_count = 0
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.call_count += 1
            self.active_count += 1
            self.max_active_count = max(self.max_active_count, self.active_count)
        try:
            time.sleep(0.01)
            data = []
            for index, text in enumerate(kwargs["input"]):
                checksum = sum(ord(character) for character in text)
                data.append(
                    SimpleNamespace(
                        index=index,
                        embedding=[
                            1.0,
                            float(checksum % 17 + 1),
                            float(checksum % 31 + 1),
                        ],
                    )
                )
            return SimpleNamespace(data=data)
        finally:
            with self.lock:
                self.active_count -= 1


def test_cleaned_policy_files_parse_to_individual_articles() -> None:
    expected = {
        ROAD_TRAFFIC_SAFETY_LAW: (124, "第一章第一条", "第八章第一百二十四条"),
        ROAD_TRAFFIC_SAFETY_REGULATION: (115, "第一章第一条", "第八章第一百一十五条"),
    }

    for policy in POLICY_DATABASES:
        text = Path(policy.source_file).read_text(encoding="utf-8")
        items = rag_manager.parse_policy_items(text)
        count, first_reference, last_reference = expected[policy.title]

        assert len(items) == count
        assert items[0].chapter == first_reference
        assert items[-1].chapter == last_reference
        assert "\u3000" not in text
        assert "目 录" not in text


def test_policy_item_does_not_serialize_vector() -> None:
    item = rag_policy_item(
        describe="法规正文",
        des_vector=[0.1, 0.2],
        chapter="第一章第一条",
    )

    assert item.model_dump() == {
        "describe": "法规正文",
        "chapter": "第一章第一条",
    }


def test_search_rag_returns_five_items_from_each_database() -> None:
    config = EmbeddingSettings(
        api_key="test-key",
        model_name="test-embedding",
        dimensions=2,
    )
    embedding_client = SimpleNamespace(embeddings=_FakeEmbeddings())
    manager = rag_manager(config=config, embedding_client=embedding_client)
    manager._collections = {
        ROAD_TRAFFIC_SAFETY_LAW: _FakeCollection(ROAD_TRAFFIC_SAFETY_LAW),
        ROAD_TRAFFIC_SAFETY_REGULATION: _FakeCollection(
            ROAD_TRAFFIC_SAFETY_REGULATION
        ),
    }
    with patch.object(manager, "_ensure_databases", return_value=None):
        result = manager.search_rag(" 酒后驾驶如何处罚 ")

    assert list(result) == [
        ROAD_TRAFFIC_SAFETY_LAW,
        ROAD_TRAFFIC_SAFETY_REGULATION,
    ]
    assert all(len(items) == 5 for items in result.values())
    assert all(
        "des_vector" not in item
        for items in result.values()
        for item in items
    )


def test_search_rag_rejects_empty_prompt() -> None:
    try:
        rag_manager().search_rag("  ")
    except ValueError as exc:
        assert "不能为空" in str(exc)
    else:
        raise AssertionError("空检索词应触发 ValueError")


def test_real_chromadb_persists_and_reuses_two_databases(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("chromadb", reason="chromadb 未安装,跳过真实向量库集成测试")
    rag_module = importlib.import_module("app.modules.agent.rag.rag_manager")
    database_root = tmp_path / "chroma_db"
    policies = []

    for policy_index in range(2):
        title = f"测试法规{policy_index + 1}"
        source_file = tmp_path / f"{title}.txt"
        source_file.write_text(
            "第一章\n\n"
            + "\n".join(
                f"第{article_index}条 {title}的第{article_index}条正文"
                for article_index in range(1, 7)
            )
            + "\n",
            encoding="utf-8",
        )
        policies.append(
            rag_module._PolicyDatabase(
                title=title,
                source_file=source_file,
                database_dir=database_root / f"policy-{policy_index + 1}",
            )
        )

    monkeypatch.setattr(rag_module, "CHROMA_DB_DIR", database_root)
    monkeypatch.setattr(rag_module, "POLICY_DATABASES", tuple(policies))

    embeddings = _DeterministicEmbeddings()
    embedding_client = SimpleNamespace(embeddings=embeddings)
    config = EmbeddingSettings(
        url="https://embedding.test/v1",
        api_key="test-key",
        model_name="test-embedding",
        dimensions=3,
        concurrency=3,
    )

    first_manager = rag_module.rag_manager(
        config=config,
        embedding_client=embedding_client,
    )
    assert first_manager.initialize_databases() == {
        "测试法规1": 6,
        "测试法规2": 6,
    }
    first_result = first_manager.search_rag("测试检索")
    assert all(len(items) == 5 for items in first_result.values())
    assert embeddings.call_count == 13  # 逐条生成 12 个法条向量，加一次查询。
    assert embeddings.max_active_count == 3

    second_manager = rag_module.rag_manager(
        config=config,
        embedding_client=embedding_client,
    )
    second_result = second_manager.search_rag("再次检索")

    assert all(len(items) == 5 for items in second_result.values())
    assert embeddings.call_count == 14  # 已持久化的数据库不会重复生成法条向量。
    assert all(policy.database_dir.exists() for policy in policies)
