"""配置加载与设置模型测试:YAML 解析、环境变量覆盖、类型强制转换。"""

import os
import unittest
from unittest.mock import patch

from app.config import (
    AppSettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    _bool_coerce,
    _float_coerce,
    _int_coerce,
    _list_coerce,
    load_yaml_config,
)


class AppSettingsDefaultsTest(unittest.TestCase):
    def test_all_defaults_with_empty_yaml(self) -> None:
        settings = AppSettings({"server": {}, "streaming": {}, "rtsp": {}, "yolo": {}, "risk": {},
                               "lane_segmentation": {}, "bev_calibration": {}})

        self.assertEqual(settings.HOST, "0.0.0.0")
        self.assertEqual(settings.PORT, 8000)
        self.assertTrue(settings.DEBUG)
        self.assertFalse(settings.RELOAD)
        self.assertEqual(settings.CORS_ORIGINS, ["*"])
        self.assertTrue(settings.ENABLE_STREAMING)
        self.assertTrue(settings.ENABLE_TRAFFIC_ANALYST)
        self.assertEqual(settings.STREAM_FPS, 15)
        self.assertFalse(settings.LANE_SEG_ENABLED)
        self.assertEqual(settings.LANE_SEG_INTERVAL, 5.0)
        self.assertEqual(settings.BEV_CALIBRATION, {})
        self.assertEqual(settings.BEV_LANE_WIDTH_M, 3.75)

    def test_yaml_override_and_coercion(self) -> None:
        settings = AppSettings({
            "server": {"port": "9090", "debug": "false", "reload": "true",
                       "cors_origins": '["https://app.test","http://localhost"]'},
            "streaming": {"enabled": "no", "traffic_analyst_enabled": False, "fps": "30"},
            "yolo": {"half": "false"},
            "lane_segmentation": {"enabled": "yes"},
        })

        self.assertEqual(settings.PORT, 9090)
        self.assertFalse(settings.DEBUG)
        self.assertTrue(settings.RELOAD)
        self.assertEqual(settings.CORS_ORIGINS, ["https://app.test", "http://localhost"])
        self.assertFalse(settings.ENABLE_STREAMING)
        self.assertFalse(settings.ENABLE_TRAFFIC_ANALYST)
        self.assertEqual(settings.STREAM_FPS, 30)
        self.assertFalse(settings.YOLO_HALF)
        self.assertTrue(settings.LANE_SEG_ENABLED)

    def test_env_var_overrides_yaml(self) -> None:
        with patch.dict(os.environ, {"SP2026_PORT": "7070", "SP2026_RELOAD": "true"}, clear=False):
            settings = AppSettings({"server": {"port": 8000, "reload": False}})

            self.assertEqual(settings.PORT, 7070)
            self.assertTrue(settings.RELOAD)

    def test_invalid_coercion_falls_back_to_default(self) -> None:
        settings = AppSettings({"server": {"port": "not-a-number"}})

        self.assertEqual(settings.PORT, 8000)

    def test_bev_calibration_non_dict_is_normalized_to_empty(self) -> None:
        settings = AppSettings({"bev_calibration": ["not", "a", "map"]})

        self.assertEqual(settings.BEV_CALIBRATION, {})


class CoerceFunctionsTest(unittest.TestCase):
    def test_bool_coerce(self) -> None:
        for value in (True, "true", "True", "1", "yes"):
            self.assertTrue(_bool_coerce(value))
        for value in (False, "false", "False", "0", "no", ""):
            self.assertFalse(_bool_coerce(value))

    def test_int_and_float_coerce(self) -> None:
        self.assertEqual(_int_coerce("42"), 42)
        self.assertEqual(_float_coerce("3.5"), 3.5)

    def test_list_coerce(self) -> None:
        self.assertEqual(_list_coerce(["a", "b"]), ["a", "b"])
        self.assertEqual(_list_coerce('["x","y"]'), ["x", "y"])
        self.assertEqual(_list_coerce("plain"), ["plain"])


class DatabaseSettingsTest(unittest.TestCase):
    def test_from_yaml_defaults(self) -> None:
        db = DatabaseSettings.from_yaml({})

        self.assertEqual(db.host, "127.0.0.1")
        self.assertEqual(db.port, 3306)
        self.assertEqual(db.name, "summer_project_2026")

    def test_sqlalchemy_url_encodes_special_characters(self) -> None:
        db = DatabaseSettings(
            host="127.0.0.1", port=3306, username="root",
            password="p@ss!w0rd", name="test_db", charset="utf8mb4",
        )

        url = db.sqlalchemy_url

        self.assertIn("p%40ss%21w0rd", url)
        self.assertIn("mysql+pymysql://", url)


class LLMSettingsTest(unittest.TestCase):
    def test_overrides_from_yaml(self) -> None:
        llm = LLMSettings.from_yaml({
            "llm": {"url": "http://local:6006/v1", "api_key": "sk-key", "model_name": "gemma"}
        })

        self.assertEqual(llm.url, "http://local:6006/v1")
        self.assertEqual(llm.api_key, "sk-key")
        self.assertEqual(llm.model_name, "gemma")


class EmbeddingSettingsTest(unittest.TestCase):
    def test_defaults_and_dimensions_validation(self) -> None:
        from pydantic import ValidationError

        emb = EmbeddingSettings.from_yaml({})
        self.assertEqual(emb.dimensions, 1024)
        self.assertEqual(emb.concurrency, 10)

        with self.assertRaises(ValidationError):
            EmbeddingSettings(dimensions=0)

    def test_overrides_from_yaml(self) -> None:
        emb = EmbeddingSettings.from_yaml({
            "embedding": {"timeout": 30.0, "concurrency": 5}
        })

        self.assertEqual(emb.timeout, 30.0)
        self.assertEqual(emb.concurrency, 5)


class LoadYamlConfigTest(unittest.TestCase):
    def test_missing_config_file_raises(self) -> None:
        with patch("app.config.CONFIG_PATH") as fake_path:
            fake_path.exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                load_yaml_config()

    def test_non_dict_yaml_raises(self) -> None:
        import yaml

        with (
            patch("app.config.CONFIG_PATH") as fake_path,
            patch("builtins.open"),
        ):
            fake_path.exists.return_value = True
            with patch("app.config.yaml.safe_load", return_value=["list"]):
                with self.assertRaises(ValueError):
                    load_yaml_config()


if __name__ == "__main__":
    unittest.main()
