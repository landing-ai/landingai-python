from pathlib import Path

import pytest
import responses

from landingai.storage.data_access import read_file
from landingai.pipeline.image_source import probe_video, sample_images_from_video

# TODO: solve the problem of exit code 134 when running the following test in GitHub Actions
# @patch("landingai.io.cv2.waitKey")
# @patch("landingai.io.cv2.VideoCapture")
# def test_read_from_notebook_webcam(mock_video_capture, mock_wait_key):
#     mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3)))
#     mock_wait_key.return_value = 288
#     take_photo_func = read_from_notebook_webcam()
#     filepath = take_photo_func()
#     image = PIL.Image.open(filepath)
#     assert image.size == (640, 480)


def test_sample_images_from_video(tmp_path: Path):
    test_video_file_path = "tests/data/videos/test.mp4"
    result = sample_images_from_video(test_video_file_path, tmp_path)
    assert len(result) == 2
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_probe():
    test_video_file_path = "tests/data/videos/test.mp4"
    total_frames, sample_size, video_length_seconds = probe_video(
        test_video_file_path, 1.0
    )
    assert total_frames == 48
    assert sample_size == 2
    assert video_length_seconds == 2.0


def test_probe_file_not_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        non_exist_file = str(tmp_path / "non_exist.mp4")
        probe_video(non_exist_file, 1.0)


# from responses import _recorder
# @_recorder.record(file_path="tests/data/responses/test_read_stream.yaml")
@responses.activate
def test_read_file():
    responses._add_from_file(file_path="tests/data/responses/test_read_stream.yaml")

    url = "https://landing-aviinference-benchmarking-dev.s3.us-east-2.amazonaws.com/images/test.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMSJHMEUCIFrGe9p9sBKng7GnrdeTPETmaXUz2RItwW9DtpyBXxkEAiEAomX3OFUwJUduZIJ5ujvONJUYK3qj9kOhHlZ7WvUuRAMq9gMI3P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw5NzAwNzMwNDE5OTMiDFFuNFyCyCXH%2BAiTkCrKA8DfON81axWS6frqZ0soRRkJIFlaVJyGkTlWkchYZqb5hi%2BqnIYX6sxKwtM07QOQ%2FQoAgHmGxIsLQA84YwKq2Zty7RI%2Fsxtq%2BdMEe3oeZjX%2FO%2F%2Fp%2BvYuoQXbzRDHZz%2BNL6sJaySsNzsh5lRU5qjGowff7dBBdLPswhWYlZxnng5YPSJjlvZodABudG8S88B%2BV3Ml%2BC%2Fd2Q%2Bf0FCaiyMHfMCECGlBMIXRatEmuMJksEu%2Bfhrz5IoypolbBWwCsBZOeloRcz50L5%2FlBwqyUkSD7KliJel1rN2Qoq8mCLXgY9ySHBl%2BKDgrR1n8Nh0eR99t2BQ57EcOcswSoQeqAVehFPuLCRBLpVuiP7BG4h%2Fqdi%2FoQnr5t1wrSP1T7DWzhH4uCZDNdERrYGG9RoDaxdvMJl05xpt7%2B6d1E%2BFD7hRzLRW1q9Rg7pdKmOtPE4XlPBC4MYMp7lTgXvI4QGA1nB6rTsqTwie%2Fm1q6g9%2FpRXbffuNu3pt%2FN7Vf8bRl%2B1dx%2F7CKzsgTvRDildpuYYCzHIyszQGlLFSEhtId7%2BOrCdIymmW7FcC9Adt0g31oij2FTrhLUVkf1DGzWVmWOE2A6el%2FG7IXDZJawEpNhOuCMKWUtKMGOpQCCeSRlMNN9jETYk%2B0JhcI3zmvvBTqsjGt4jrZ9FjJ2Dq2JhYvoCmBf%2F%2BrrhulxH2bYs740CdLfuDnTK8VSeJTjTHSOeRlb2r5%2Bx4DYIS3CDlj2clxWll0Vyl8Vzl8M7h%2BPwViB3zPbuB%2BsMMqPkh9uoc8uLie1aaGpQ57vqfHuJtZWLIzEptogTG2I92WCAPsiAECJkTPfNlnrtP%2F6VCzTlI1OrAKoonDuGIdp5UeA107oEaXjWo7WLQTXULRHY8q16HU86K0pkwqREK%2FPpNgZSBRQ7RymMgxJI%2F5eYWfT2JHayND6PIgOSDjtIHbqKRPVwgQOdni3vypgBWt043Koh4zLbu7bh6W%2Fm9gBRlRLvY%2BW2C1&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230523T185235Z&X-Amz-SignedHeaders=host&X-Amz-Expires=18000&X-Amz-Credential=ASIA6DXG35RE4HTVSB3E%2F20230523%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=debb56818b5126fec2ffbc06630bb94940f5f30fb2add48e67f262226df78c82"
    data = read_file(url)[
        "content"
    ]  # working url, expecting a response of status code 200
    assert type(data) == bytes
    assert len(data) == 84
    with pytest.raises(ValueError):
        url = "https://landing-aviinference-benchmarking-dev.s3.us-east-2.amazonaws.com/images/1mp_cereal_1.jpeg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMSJHMEUCIFrGe9p9sBKng7GnrdeTPETmaXUz2RItwW9DtpyBXxkEAiEAomX3OFUwJUduZIJ5ujvONJUYK3qj9kOhHlZ7WvUuRAMq9gMI3P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw5NzAwNzMwNDE5OTMiDFFuNFyCyCXH%2BAiTkCrKA8DfON81axWS6frqZ0soRRkJIFlaVJyGkTlWkchYZqb5hi%2BqnIYX6sxKwtM07QOQ%2FQoAgHmGxIsLQA84YwKq2Zty7RI%2Fsxtq%2BdMEe3oeZjX%2FO%2F%2Fp%2BvYuoQXbzRDHZz%2BNL6sJaySsNzsh5lRU5qjGowff7dBBdLPswhWYlZxnng5YPSJjlvZodABudG8S88B%2BV3Ml%2BC%2Fd2Q%2Bf0FCaiyMHfMCECGlBMIXRatEmuMJksEu%2Bfhrz5IoypolbBWwCsBZOeloRcz50L5%2FlBwqyUkSD7KliJel1rN2Qoq8mCLXgY9ySHBl%2BKDgrR1n8Nh0eR99t2BQ57EcOcswSoQeqAVehFPuLCRBLpVuiP7BG4h%2Fqdi%2FoQnr5t1wrSP1T7DWzhH4uCZDNdERrYGG9RoDaxdvMJl05xpt7%2B6d1E%2BFD7hRzLRW1q9Rg7pdKmOtPE4XlPBC4MYMp7lTgXvI4QGA1nB6rTsqTwie%2Fm1q6g9%2FpRXbffuNu3pt%2FN7Vf8bRl%2B1dx%2F7CKzsgTvRDildpuYYCzHIyszQGlLFSEhtId7%2BOrCdIymmW7FcC9Adt0g31oij2FTrhLUVkf1DGzWVmWOE2A6el%2FG7IXDZJawEpNhOuCMKWUtKMGOpQCCeSRlMNN9jETYk%2B0JhcI3zmvvBTqsjGt4jrZ9FjJ2Dq2JhYvoCmBf%2F%2BrrhulxH2bYs740CdLfuDnTK8VSeJTjTHSOeRlb2r5%2Bx4DYIS3CDlj2clxWll0Vyl8Vzl8M7h%2BPwViB3zPbuB%2BsMMqPkh9uoc8uLie1aaGpQ57vqfHuJtZWLIzEptogTG2I92WCAPsiAECJkTPfNlnrtP%2F6VCzTlI1OrAKoonDuGIdp5UeA107oEaXjWo7WLQTXULRHY8q16HU86K0pkwqREK%2FPpNgZSBRQ7RymMgxJI%2F5eYWfT2JHayND6PIgOSDjtIHbqKRPVwgQOdni3vypgBWt043Koh4zLbu7bh6W%2Fm9gBRlRLvY%2BW2C1&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230523T184836Z&X-Amz-SignedHeaders=host&X-Amz-Expires=120&X-Amz-Credential=ASIA6DXG35RE4HTVSB3E%2F20230523%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=9d62d37e678546692ad1a33091f668fe8d2a2248d867a5b96628ef68a68cc00c"
        read_file(url)  # expired

    with pytest.raises(ValueError):
        url = (
            "https://test.s3.us-west-2.amazonaws.com/img.png?X-Amz-Security-Token=12345"
        )
        read_file(url)  # 301, redirect
