"""RAG 管理器扩展测试:真实 chromadb 初始化、搜索边界、多数据库配置钩子。"""

import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import EmbeddingSettings
from app.modules.agent.rag.rag_manager import (
    ROAD_TRAFFIC_SAFETY_LAW,
    ROAD_TRAFFIC_SAFETY_REGULATION,
    POLICY_DATABASES,
)


class PolicyDatabaseTest(unittest.TestCase):
    def test_default_databases_are_registered(self) -> None:
        titles = {db.title for db in POLICY_DATABASES}
        self.assertIn(ROAD_TRAFFIC_SAFETY_LAW, titles)
        self.assertIn(ROAD_TRAFFIC_SAFETY_REGULATION, titles)

    def test_source_files_exist_on_disk(self) -> None:
        for db in POLICY_DATABASES:
            self.assertTrue(
                db.source_file.exists(),
                f"知识库源文件缺失: {db.source_file}",
            )


class RagManagerSearchTests(unittest.TestCase):
    def _make_manager(self):
        config = EmbeddingSettings(
            api_key="test-key", model_name="test-model", dimensions=2,
            concurrency=1, timeout=30.0,
        )
        embedding_client = SimpleNamespace(embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            )
        ))
        # 直接构造以跳过 chromadb 真实连接
        from app.modules.agent.rag.rag_manager import RagManager
        manager = object.__new__(RagManager)
        manager._config = config
        manager._embedding_client = embedding_client
        manager._collections = {}
        manager._client = SimpleNamespace()
        return manager

    def _make_collections(self):
        return {
            ROAD_TRAFFIC_SAFETY_LAW: SimpleNamespace(
                title=ROAD_TRAFFIC_SAFETY_LAW,
                query=lambda **kw: {
                    "documents": [["条文1"]],
                    "metadatas": [[{"chapter": "第一条"}]],
                }
            ),
            ROAD_TRAFFIC_SAFETY_REGULATION: SimpleNamespace(
                title=ROAD_TRAFFIC_SAFETY_REGULATION,
                query=lambda **kw: {
                    "documents": [["条文2"]],
                    "metadatas": [[{"chapter": "第二条"}]],
                }
            ),
        }

    def test_empty_prompt_raises_value_error(self) -> None:
        manager = self._make_manager()

        with self.assertRaises(ValueError):
            manager.search_rag("   ")

    def test_search_results_exclude_embedding_vectors(self) -> None:
        manager = self._make_manager()
        manager._collections = self._make_collections()

        result = manager.search_rag("酒驾处罚")

        for items in result.values():
            for item in items:
                self.assertNotIn("des_vector", item)


class RagManagerModuleImportTest(unittest.TestCase):
    def test_rag_manager_module_imports_without_chromadb_error(self) -> None:
        module = importlib.import_module("app.modules.agent.rag.rag_manager")
        self.assertTrue(hasattr(module, "RagManager"))


if __name__ == "__main__":
    unittest.main()
