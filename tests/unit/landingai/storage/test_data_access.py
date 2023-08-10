import os

import pytest
import responses

from landingai.storage.data_access import download_file, fetch_from_uri, read_file


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


def test_download_file():
    url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_light_color_272x92dp.png"
    file = download_file(url=url)
    assert os.path.exists(file) == 1
    assert os.path.getsize(file) > 5000


def test_fetch_from_uri():
    # Test the local access case
    uri = "tests/data/images/cereal1.jpeg"
    local_file = fetch_from_uri(uri=uri)
    assert os.path.getsize(local_file) > 5000
