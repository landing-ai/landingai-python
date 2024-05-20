import io
import logging
import os
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import responses
from PIL import Image
from responses.matchers import multipart_matcher

from landingai.common import (
    APIKey,
    ClassificationPrediction,
    InferenceMetadata,
    OcrPrediction,
)
from landingai.exceptions import (
    BadRequestError,
    ClientError,
    InternalServerError,
    InvalidApiKeyError,
    PermissionDeniedError,
    ServiceUnavailableError,
    UnauthorizedError,
    UnexpectedRedirectError,
)
from landingai.predict import EdgePredictor, Predictor, OcrPredictor
from landingai.predict.snowflake import SnowflakeNativeAppPredictor
from landingai.visualize import overlay_predictions
from landingai.pipeline.frameset import FrameSet, Frame


def test_predict_with_none():
    with pytest.raises(ValueError) as excinfo:
        Predictor("12345", api_key="land_sk_1111").predict(None)
    assert "Input image must be non-emtpy, but got: None" in str(excinfo.value)


def test_predict_with_empty_array():
    with pytest.raises(ValueError) as excinfo:
        Predictor("12345", api_key="land_sk_1111").predict(np.array([]))
    assert "Input image must be non-emtpy, but got: []" in str(excinfo.value)


@responses.activate
def test_predict_500():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_500.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(InternalServerError) as excinfo:
        Predictor(
            "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", api_key="land_sk_1111"
        ).predict(img)
    assert (
        "Internal server error. The model server encountered an unexpected condition and failed. Please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_504():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_504.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ServiceUnavailableError) as excinfo:
        Predictor(
            "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", api_key="land_sk_1111"
        ).predict(img)
    assert "Service temporarily unavailable. Please try again in a few minutes." in str(
        excinfo.value
    )


@responses.activate
def test_predict_503():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_503.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ServiceUnavailableError) as excinfo:
        Predictor(
            "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", api_key="land_sk_1111"
        ).predict(img)
    assert "Service temporarily unavailable. Please try again in a few minutes." in str(
        excinfo.value
    )


@responses.activate
def test_predict_404():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_404.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(BadRequestError) as excinfo:
        with patch.object(Predictor, "_url", "https://predict.app.landing.ai/v0/foo"):
            Predictor(
                "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", api_key="land_sk_1111"
            ).predict(img)
    assert (
        "Endpoint doesn't exist. Please check the inference url path and other configuration is correct and try again. If this issue persists, please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_400():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_400.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ClientError) as excinfo:
        Predictor(
            "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", api_key="land_sk_1111"
        ).predict(img)
    assert (
        "Client error. Please check your configuration and inference request is well-formed and try again. If this issue persists, please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_300():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_300.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(UnexpectedRedirectError) as excinfo:
        Predictor(
            "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", api_key="land_sk_1111"
        ).predict(img)
    assert (
        "Unexpected redirect. Please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@mock.patch("tenacity.nap.time.sleep")
@responses.activate
def test_predict_429_rate_limited(mocked_sleep):
    mocked_sleep.return_value = 0
    # This response will be retried
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_429.yaml"
    )
    # This response will be used to finish the test
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_300.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(UnexpectedRedirectError) as excinfo:
        Predictor._num_retry = 1
        Predictor(
            "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", api_key="land_sk_1111"
        ).predict(img)
    assert (
        "Unexpected redirect. Please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_403_run_out_of_credits():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_403.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(PermissionDeniedError) as excinfo:
        Predictor(
            "dfa79692-75eb-4a48-b02e-b273751adbae",
            api_key="land_sk_1r1c7i0c1pnmp1oac1748fnrnerh7dg",
        ).predict(img)
    assert (
        "Permission denied. Please check your account has enough credits or your enterprise contract is not expired or if you have access to this endpoint. Contact your account admin for more information."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_401_with_wrong_api_key():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_401.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(UnauthorizedError) as excinfo:
        Predictor(
            "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43",
            api_key="land_sk_wrong_key",
        ).predict(img)
    assert "Unauthorized. Please check your API key and API secret is correct." in str(
        excinfo.value
    )


@responses.activate
def test_predict_422_with_wrong_endpoint_id():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_422.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(BadRequestError) as excinfo:
        Predictor("12345", api_key="land_sk_correct_key").predict(img)
    assert "Bad request. Please check your Endpoint ID is correct." in str(
        excinfo.value
    )


@responses.activate
def test_predict_matching_expected_request_body():
    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    expected_request_files = {"file": _read_image_as_jpeg_bytes(img_path)}
    responses.post(
        url="https://predict.app.landing.ai/inference/v1/predict?endpoint_id=8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43",
        match=[multipart_matcher(expected_request_files)],
        json={
            "backbonetype": None,
            "backbonepredictions": None,
            "predictions": {
                "score": 0.9951885938644409,
                "labelIndex": 0,
                "labelName": "Fire",
            },
            "type": "ClassificationPrediction",
        },
    )
    predictor = Predictor(
        "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43",
        api_key="land_sk_1111",
    )
    predictor.predict(img)


@responses.activate
@mock.patch("snowflake.connector.connect")
def test_snowflake_nativeapp_predict_matching_expected_request_body(
    snowflake_connect_mock,
):
    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    expected_request_files = {"file": _read_image_as_jpeg_bytes(img_path)}
    responses.post(
        url=(
            "https://my-nativeapp.snowflakeapp.com/inference"
            "/v1/predict?endpoint_id=8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43"
        ),
        match=[multipart_matcher(expected_request_files)],
        json={
            "backbonetype": None,
            "backbonepredictions": None,
            "predictions": {
                "score": 0.9951885938644409,
                "labelIndex": 0,
                "labelName": "Fire",
            },
            "type": "ClassificationPrediction",
        },
    )

    # Mock snowflake connection
    snowflake_ctx = snowflake_connect_mock.return_value
    snowflake_ctx._rest._token_request.return_value = {
        "data": {"sessionToken": "fake_token"}
    }
    predictor = SnowflakeNativeAppPredictor(
        "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43",
        snowflake_account="ABCD1234",
        snowflake_user="my-user",
        snowflake_password="my-password",
        native_app_url="https://my-nativeapp.snowflakeapp.com",
    )
    response = predictor.predict(img)
    assert response == [
        ClassificationPrediction(
            score=0.9951885938644409, label_name="Fire", label_index=0
        )
    ]
    assert snowflake_connect_mock.call_count == 1
    snowflake_connect_mock.assert_called_with(
        user="my-user",
        password="my-password",
        account="ABCD1234",
        session_parameters={"PYTHON_CONNECTOR_QUERY_RESULT_FORMAT": "json"},
    )

    # Make sure that subsequent calls to predict() will not call snowflake.connect again,
    # and will instead reuse the token
    response = predictor.predict(img)
    assert response == [
        ClassificationPrediction(
            score=0.9951885938644409, label_name="Fire", label_index=0
        )
    ]
    assert snowflake_connect_mock.call_count == 1  # No new calls to snowflake.connect

    # And after 5 minutes, it should call snowflake.connect again
    predictor._last_auth_token_fetch = (
        predictor._last_auth_token_fetch
        - SnowflakeNativeAppPredictor.AUTH_TOKEN_MAX_AGE
    )
    response = predictor.predict(img)
    assert response == [
        ClassificationPrediction(
            score=0.9951885938644409, label_name="Fire", label_index=0
        )
    ]
    assert (
        snowflake_connect_mock.call_count == 2
    )  # New call to snowflake.connect as made


@patch("socket.socket")
@responses.activate
def test_edge_class_predict(connect_mock):
    # Fake a succesfull connection
    sock_instance = connect_mock.return_value
    sock_instance.connect_ex.return_value = 0
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_class_predict.yaml"
    )
    # Project: https://app.landing.ai/app/376/pr/26119078438913/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 26119078438913 \
    #                                            -m "59eff733-1dcd-4ace-b104-9041c745f1da" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    img = Image.open("tests/data/images/wildfire1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    assert preds[0].label_name == "HasFire"
    assert preds[0].label_index == 0
    np.testing.assert_almost_equal(
        preds[0].score, 0.99565023, decimal=3, err_msg="class score mismatch"
    )
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_edge_class.jpg")


@patch("socket.socket")
@responses.activate
def test_edge_batch_predict(connect_mock):
    # Fake a successful connection
    sock_instance = connect_mock.return_value
    sock_instance.connect_ex.return_value = 0
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_class_predict.yaml"
    )
    # Project: https://app.landing.ai/app/376/pr/26119078438913/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 26119078438913 \
    #                                            -m "59eff733-1dcd-4ace-b104-9041c745f1da" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    test_image = "tests/data/images/wildfire1.jpeg"
    frs = FrameSet.from_image(test_image)
    for _ in range(9):
        frs.append(Frame.from_image(test_image))
    frs.run_predict(predictor=predictor, num_workers=5)

    for frame in frs:
        assert len(frame.predictions) == 1, "Result should not be empty or None"
        assert frame.predictions[0].label_name == "HasFire"
        assert frame.predictions[0].label_index == 0
        np.testing.assert_almost_equal(
            frame.predictions[0].score,
            0.99565023,
            decimal=3,
            err_msg="class score mismatch",
        )


@responses.activate
def test_connection_check():
    with pytest.raises(ConnectionError):
        EdgePredictor("localhost", 51203, check_server_ready=True)  # Non existing port
    # There should not be any ConnectionError for Cloud inference endpoint
    Predictor(
        endpoint_id="123",
        api_key="land_sk_1234",
        check_server_ready=True,
    )


@responses.activate
@patch("socket.socket")
def test_edge_od_predict(connect_mock):
    # Fake a succesfull connection
    sock_instance = connect_mock.return_value
    sock_instance.connect_ex.return_value = 0
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Endpoint: https://app.landing.ai/app/376/pr/11165/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 11165 \
    #                                    -m "5d0b04ce-8327-465c-b270-6913d92f5936" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    responses._add_from_file(file_path="tests/data/responses/test_edge_od_predict.yaml")
    img = np.asarray(Image.open("tests/data/images/cereal1.jpeg"))
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    preds = predictor.predict(img)
    assert len(preds) == 3, "Result should not be empty or None"
    expected_scores = [0.993014, 0.992160, 0.954738]
    expected_bboxes = [
        (945, 1603, 1118, 1795),
        (436, 1037, 640, 1203),
        (1515, 1419, 1977, 1787),
    ]
    for i, pred in enumerate(preds):
        assert pred.label_name == "Screw"
        assert pred.label_index == 1
        np.testing.assert_almost_equal(
            pred.score, expected_scores[i], decimal=3, err_msg="OD score mismatch"
        )
        assert pred.bboxes == expected_bboxes[i]
    logging.info(preds)
    img_with_preds = overlay_predictions(predictions=preds, image=img)
    img_with_preds.save("tests/output/test_edge_od.jpg")


@responses.activate
@patch("socket.socket")
def test_edge_seg_predict(connect_mock, seg_mask_validator):
    # Fake a succesfull connection
    sock_instance = connect_mock.return_value
    sock_instance.connect_ex.return_value = 0

    expected_seg_prediction = {
        "label_name": "screw",
        "label_index": 1,
        "score": 0.99487104554061,
        "encoded_mask": "2134595Z28N2020Z28N2020Z28N2015Z36N2012Z36N2012Z36N2007Z43N2005Z43N2002Z49N1999Z49N1999Z49N1997Z51N1997Z51N1992Z58N1990Z58N1990Z58N1982Z66N1982Z66N1974Z74N1974Z74N1974Z74N1969Z79N1969Z79N1967Z81N1967Z81N1967Z81N1964Z84N1964Z84N1964Z84N1964Z87N1961Z87N1958Z90N1958Z90N1958Z90N1956Z92N1956Z92N1953Z95N1953Z95N1953Z95N1951Z97N1951Z97N1946Z99N1949Z99N1949Z99N1905Z10N28Z105N1905Z10N28Z105N1895Z31N12Z110N1895Z31N12Z110N1895Z31N12Z110N1892Z156N1892Z156N1892Z156N1890Z158N1890Z158N1890Z158N1890Z158N1890Z158N1890Z156N1892Z156N1889Z156N1892Z156N1892Z156N1892Z154N1894Z154N1894Z151N1897Z151N1897Z151N1897Z146N1902Z146N1902Z146N1902Z144N1904Z144N1904Z141N1907Z141N1907Z141N1907Z138N1910Z138N1910Z133N1915Z133N1915Z133N1915Z128N1920Z128N1920Z120N1928Z120N1928Z120N1928Z118N1930Z118N1933Z110N1938Z110N1938Z110N1938Z107N1941Z107N1941Z107N1941Z102N1946Z102N1948Z98N1950Z98N1950Z98N1950Z92N1956Z92N1959Z82N1966Z82N1966Z82N1968Z75N1973Z75N1976Z69N1979Z69N1979Z69N1979Z67N1981Z67N1984Z64N1984Z64N1984Z64N1984Z61N1987Z61N1987Z61N1987Z61N1987Z61N1989Z59N1989Z59N1989Z59N1992Z53N1995Z53N1997Z51N1997Z51N1997Z51N2000Z48N2000Z48N2005Z43N2005Z43N2005Z43N2007Z41N2007Z41N2007Z41N2010Z38N2010Z38N2013Z35N2013Z35N2013Z35N2018Z28N2020Z28N2030Z15N2033Z15N2033Z15N531789Z18N2030Z18N2030Z18N2020Z33N2015Z33N2010Z43N2005Z43N2005Z43N2000Z53N1995Z53N1995Z53N1992Z61N1987Z61N1984Z67N1981Z67N1981Z67N1976Z75N1973Z75N1966Z82N1966Z82N1966Z82N1955Z95N1953Z95N1948Z105N1943Z105N1943Z105N1938Z113N1935Z113N1930Z123N1925Z123N1925Z123N1920Z131N1917Z131N1917Z131N1912Z138N1910Z138N1905Z143N1905Z143N1905Z143N1902Z149N1899Z149N1894Z154N1894Z154N1894Z154N1891Z159N1889Z159N1887Z164N1884Z164N1884Z164N1879Z171N1877Z171N1877Z171N1872Z179N1869Z179N1861Z187N1861Z187N1861Z187N1856Z194N1854Z194N1849Z202N1846Z202N1846Z202N1841Z207N1841Z207N1838Z210N1838Z210N1838Z210N1833Z218N1830Z218N1825Z223N1825Z223N1825Z223N1822Z226N1822Z226N1822Z226N1817Z231N1817Z231N1815Z235N1813Z235N1813Z235N1805Z243N1805Z243N1797Z251N1797Z251N1797Z251N1792Z256N1792Z256N1784Z264N1784Z264N1784Z264N1779Z269N1779Z269N1777Z271N1777Z271N1777Z271N1772Z276N1772Z276N1772Z276N1769Z279N1769Z279N1764Z284N1764Z284N1764Z284N1754Z292N1756Z292N1738Z310N1738Z310N1738Z310N1725Z320N1728Z320N1718Z330N1718Z330N1718Z330N1713Z332N1716Z332N1716Z332N1713Z333N1715Z333N1710Z338N1710Z338N1710Z338N1707Z338N1710Z338N1672Z23N8Z343N1674Z23N8Z343N1674Z23N8Z343N1669Z376N1672Z376N1667Z381N1667Z381N1667Z381N1659Z387N1661Z387N1653Z392N1656Z392N1656Z392N1654Z392N1656Z392N1656Z392N1653Z392N1656Z392N1656Z389N1659Z389N1659Z389N1659Z387N1661Z387N1661Z384N1664Z384N1664Z384N1664Z382N1666Z382N1666Z377N1671Z377N1671Z377N1671Z371N1677Z371N1677Z366N1682Z366N1682Z366N1682Z364N1684Z364N1684Z364N1684Z359N1689Z359N1687Z358N1690Z358N1690Z358N1690Z356N1692Z356N1692Z353N1695Z353N1695Z353N1695Z348N1700Z348N1697Z346N1702Z346N1702Z346N1152Z20N530Z341N1157Z20N530Z341N1157Z20N530Z341N1154Z31N522Z336N1159Z31N522Z336N1157Z38N517Z333N1160Z38N517Z333N1160Z38N517Z333N1157Z44N514Z328N1162Z44N514Z328N1162Z51N507Z307N1183Z51N507Z307N1183Z51N507Z307N1181Z66N497Z299N1186Z66N497Z299N1186Z71N492Z294N1191Z71N492Z294N1191Z71N492Z294N1188Z79N489Z287N1193Z79N489Z287N1193Z84N489Z277N1198Z84N489Z277N1198Z84N489Z277N1195Z93N486Z269N1200Z93N486Z269N1200Z93N486Z269N1198Z97N489Z261N1201Z97N489Z261N1201Z97N492Z251N1208Z97N492Z251N1208Z97N492Z251N1208Z97N494Z238N1219Z97N494Z238N1219Z97N499Z226N1226Z97N499Z226N1226Z97N499Z226N1226Z97N499Z220N1232Z97N499Z220N1232Z97N502Z212N1237Z97N502Z212N1237Z97N502Z212N1239Z98N501Z205N1244Z98N501Z205N1244Z98N504Z200N1246Z98N504Z200N1246Z98N504Z200N1246Z100N502Z197N1249Z100N502Z197N1249Z100N502Z197N1252Z97N505Z189N1257Z97N505Z189N1262Z95N502Z184N1267Z95N502Z184N1267Z95N502Z184N1270Z94N500Z176N1278Z94N500Z176N1280Z92N500Z174N1282Z92N500Z174N1282Z92N500Z174N1287Z87N500Z169N1292Z87N500Z169N1295Z84N500Z166N1298Z84N500Z166N1298Z84N500Z166N1300Z82N500Z158N1308Z82N500Z158N1308Z82N500Z158N1311Z79N500Z153N1316Z79N500Z153N1316Z79N502Z146N1321Z79N502Z146N1321Z79N502Z146N1318Z82N502Z141N1323Z82N502Z141N1323Z80N504Z141N1323Z80N504Z141N1323Z80N504Z141N1323Z77N507Z138N1326Z77N507Z138N1324Z77N509Z136N1326Z77N509Z136N1326Z77N509Z136N1326Z74N515Z133N1326Z74N515Z133N1326Z74N515Z130N1329Z74N515Z130N1329Z74N515Z130N1329Z72N517Z130N1329Z72N517Z130N1329Z72N517Z130N1326Z75N519Z126N1328Z75N519Z126N1328Z75N522Z123N1328Z75N522Z123N1328Z75N522Z123N1326Z77N524Z121N1326Z77N524Z121N1323Z80N524Z121N1323Z80N524Z121N1323Z80N524Z121N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z82N535Z113N1318Z82N535Z113N1318Z82N535Z113N1318Z82N535Z110N1321Z82N535Z110N1321Z82N538Z107N1321Z82N538Z107N1321Z82N538Z107N1321Z79N543Z103N1323Z79N543Z103N1323Z77N548Z97N1326Z77N548Z97N1326Z77N548Z97N1324Z76N553Z95N1324Z76N553Z95N1324Z74N560Z87N1327Z74N560Z87N1327Z74N560Z87N1327Z71N566Z82N1329Z71N566Z82N1329Z71N566Z82N1329Z69N571Z79N1329Z69N571Z79N1329Z69N576Z71N1332Z69N576Z71N1332Z69N576Z71N1334Z67N578Z67N1336Z67N578Z67N1336Z64N584Z61N1339Z64N584Z61N1339Z64N584Z61N1339Z64N586Z54N1344Z64N586Z54N1344Z64N586Z49N1349Z64N586Z49N1349Z64N586Z49N1349Z62N591Z43N1352Z62N591Z43N1352Z62N593Z39N1354Z62N593Z39N1354Z62N593Z39N1354Z62N596Z33N1357Z62N596Z33N1357Z62N596Z33N1360Z56N604Z26N1362Z56N604Z26N1364Z51N625Z3N1369Z51N625Z3N1369Z51N625Z3N1372Z46N2002Z46N2007Z36N2012Z36N2012Z36N2014Z29N2019Z29N2025Z17N2031Z17N2031Z17N2033Z13N2035Z13N2038Z5N2043Z5N2043Z5N519175Z",
        "num_predicted_pixels": 94553,
        "percentage_predicted_pixels": 0.02254319190979004,
        "mask_shape": (2048, 2048),
    }
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_seg_predict.yaml"
    )
    # Project: https://app.landing.ai/app/376/pr/26113016987660/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 26113016987660 \
    #                                          -m "1a2b49dd-25c0-45fa-9f1c-4433acee80cd" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    img = Image.open("tests/data/images/cereal1.jpeg")
    assert img is not None
    preds = predictor.predict(
        img,
        metadata=InferenceMetadata(
            imageId="test-img-id-1",
            inspectionStationId="camera-station-1",
            locationId="factory-floor-1",
        ),
    )
    assert len(preds) == 1, "Result should not be empty or None"
    seg_mask_validator(preds[0], expected_seg_prediction)
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_edge_seg.jpg")


def _read_image_as_jpeg_bytes(file_path: str) -> bytes:
    img_buffer = io.BytesIO()
    image = Image.open(file_path)
    image.save(img_buffer, format="JPEG")
    data = img_buffer.getvalue()
    assert data is not None
    return data


def test_load_api_credential_invalid_key():
    with pytest.raises(InvalidApiKeyError):
        Predictor("endpoint_1234")
    with pytest.raises(InvalidApiKeyError):
        Predictor("endpoint_1234", api_key="fake_key")
    with pytest.raises(InvalidApiKeyError):
        os.environ["landingai_api_key"] = "1234"
        Predictor("endpoint_1234")


def test_load_api_credential_from_constructor():
    predictor = Predictor("endpoint_1234", api_key="land_sk_1234")
    assert predictor._api_credential.api_key == "land_sk_1234"


def test_load_api_credential_from_env_var():
    os.environ["landingai_api_key"] = "land_sk_123"
    predictor = Predictor("endpoint_1234")
    assert predictor._api_credential.api_key == "land_sk_123"
    del os.environ["landingai_api_key"]


def test_load_api_credential_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        """
                        LANDINGAI_API_KEY="land_sk_12345"
                        """
    )
    # Overwrite the default env_prefix to avoid conflict with the real .env
    APIKey.__config__.env_file = str(env_file)
    predictor = Predictor("endpoint_1234")
    assert predictor._api_credential.api_key == "land_sk_12345"
    # reset back to the default config
    APIKey.__config__.env_file = ".env"
    env_file.unlink()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "weird mode"},
        {"mode": "single-text"},
    ],
)
def test_OcrPredictor_kwargs(kwargs):
    predictor = OcrPredictor(api_key="land_sk_something")
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        predictor.predict(image, **kwargs)


@patch("socket.socket")
@responses.activate
def test_predict_ocr_successful_expected_request_body(connect_mock):
    # Fake a succesfull connection
    sock_instance = connect_mock.return_value
    sock_instance.connect_ex.return_value = 0

    responses._add_from_file(file_path="tests/data/responses/test_ocr_predict.yaml")

    img_path = "tests/data/images/ocr_test.png"
    img = Image.open(img_path)

    expected_predictions = [
        OcrPrediction(
            text="test",
            location=[(598, 248), (818, 250), (818, 303), (598, 301)],
            score=0.6326617002487183,
        )
    ]

    predictor = OcrPredictor(api_key="land_sk_something")
    result = predictor.predict(img)

    assert result == expected_predictions
