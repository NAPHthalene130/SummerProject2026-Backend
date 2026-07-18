"""图片上传 API 测试:类型校验、大小限制、空文件与落盘清理。"""

import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.api.v1 import uploads


def _upload_file(content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


class UploadImageApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_unsupported_content_type(self) -> None:
        file = _upload_file(b"data", "text/plain")

        with self.assertRaises(HTTPException) as caught:
            await uploads.upload_image(file)

        self.assertEqual(caught.exception.status_code, 415)

    async def test_rejects_empty_file(self) -> None:
        file = _upload_file(b"", "image/png")

        with self.assertRaises(HTTPException) as caught:
            await uploads.upload_image(file)

        self.assertEqual(caught.exception.status_code, 400)

    async def test_success_writes_file_and_returns_url(self) -> None:
        file = _upload_file(b"\x89PNG\r\n\x1a\nfake", "image/png")

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(uploads, "_IMAGE_DIR", Path(tmp_dir)):
                result = await uploads.upload_image(file)

                self.assertTrue(result["url"].startswith("/orderImg/mobile-"))
                self.assertTrue(result["url"].endswith(".png"))
                written = list(Path(tmp_dir).iterdir())
                self.assertEqual(len(written), 1)
                self.assertEqual(written[0].read_bytes(), b"\x89PNG\r\n\x1a\nfake")

    async def test_oversize_file_is_rejected_and_cleaned_up(self) -> None:
        file = _upload_file(b"x" * 64, "image/jpeg")

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(uploads, "_IMAGE_DIR", Path(tmp_dir)),
                patch.object(uploads, "_MAX_IMAGE_SIZE", 8),
            ):
                with self.assertRaises(HTTPException) as caught:
                    await uploads.upload_image(file)

                self.assertEqual(caught.exception.status_code, 413)
                # 超限文件必须删除,不留残留
                self.assertEqual(list(Path(tmp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
