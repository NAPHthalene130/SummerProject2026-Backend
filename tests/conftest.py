"""pytest 共享夹具:每个用例前后重置进程级单例,避免跨用例状态污染。"""

import pytest

from app.modules.camera_data import CameraDataStore


@pytest.fixture(autouse=True)
def reset_camera_data_store():
    store = CameraDataStore()
    yield store
    store._data.clear()
    store._incident_data.clear()
    store._traffic_metrics.clear()
    store._prediction_tracks.clear()
    store._prediction_speed_samples.clear()
    store._last_prediction_sample.clear()
    store._active_incidents.clear()
    store._consecutive_normal_count.clear()
    store._lane_count_cache.clear()
