"""CameraManager 单例与摄像头配置管理测试:基于项目内 config.yaml 的稳定数据。"""

import unittest

from pydantic import ValidationError

from app.utils.camera_manager import CameraConfig, CameraInfo, CameraManager


class CameraConfigModelTest(unittest.TestCase):
    def test_valid_config(self) -> None:
        config = CameraConfig(id="cam-1", name="测试", url="rtsp://x/y", longitude=0, latitude=0)

        self.assertEqual(config.id, "cam-1")

    def test_missing_field_raises(self) -> None:
        with self.assertRaises(ValidationError):
            CameraConfig(name="测试")


class CameraManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = CameraManager()

    def test_config_yaml_has_thirty_configured_cameras(self) -> None:
        cameras = self.manager.get_all()

        self.assertEqual(len(cameras), 30)
        self.assertIsInstance(cameras[0], CameraConfig)

    def test_get_by_id_returns_camera_config(self) -> None:
        cam = self.manager.get_by_id("cam-01")

        self.assertIsNotNone(cam)
        self.assertEqual(cam.name, "苏州街-海淀南路")
        self.assertTrue(cam.url.startswith("rtsp://"))
        self.assertAlmostEqual(cam.longitude, 116.30705)
        self.assertAlmostEqual(cam.latitude, 39.972601)

    def test_get_by_id_missing_returns_none(self) -> None:
        self.assertIsNone(self.manager.get_by_id("cam-999"))

    def test_camera_info_map_roundtrip(self) -> None:
        info = CameraInfo(car_info=[], weather="晴天")
        self.manager.update_camera_info("cam-01", info)

        result = self.manager.get_camera_info("cam-01")

        self.assertIs(result, info)

    def test_all_camera_info_is_empty_initially(self) -> None:
        self.assertEqual(self.manager.get_all_camera_info(), {})

    def test_reload_re_reads_config_yaml(self) -> None:
        self.manager.reload()

        self.assertEqual(len(self.manager.get_all()), 30)


if __name__ == "__main__":
    unittest.main()
