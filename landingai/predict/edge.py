from landingai.common import (
    ClassificationPrediction,
    InferenceMetadata,
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)
from landingai.predict.cloud import Predictor
from landingai.predict.utils import (
    PredictionExtractor,
    get_cloudinference_prediction,
    create_requests_session,
)
from landingai.timer import Timer
from landingai.utils import serialize_image


import PIL.Image
import numpy as np


from typing import Any, Dict, List, Optional, Union


class EdgePredictor(Predictor):
    """`EdgePredictor` runs local inference by connecting to an edge inference service (e.g. LandingEdge)"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        check_server_ready: bool = True,
    ) -> None:
        """By default the inference service runs on `localhost:8000`

        Parameters
        ----------
        host : str, optional
            Hostname or IP, by default "localhost"
        port : int, optional
            Port, by default 8000
        check_server_ready : bool, optional
            Check if the inference server is running, by default True
        """
        self._url = f"http://{host}:{port}/images"
        # Check if the inference server is reachable
        if check_server_ready and not self._check_connectivity(host=(host, port)):
            raise ConnectionError(
                f"Failed to connect to the model server. Please check if the server is running and the connection url ({self._url})."
            )
        self._session = create_requests_session(
            self._url,
            0,
            {
                "contentType": "multipart/form-data"
            },  # No retries for the inference service
        )

    @Timer(name="EdgePredictor.predict")
    def predict(
        self,
        image: Union[np.ndarray, PIL.Image.Image],
        metadata: Optional[InferenceMetadata] = None,
        reuse_session: bool = True,
        **kwargs: Any,
    ) -> List[Prediction]:
        """Run Edge inference on the input image and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted
        metadata
            The (optional) metadata associated with this inference/image.
            Metadata is helpful for attaching additional information to the inference result so you can later filter the historical inference results by your custom values in LandingLens.
            Note: The metadata is not reported back to LandingLens by default unless the edge inference server (i.e. ModelRunner) enables the feature of reporting historical inference results.

            See `landingai.common.InferenceMetadata` for more details.
        reuse_session
            Whether to reuse the HTTPS session for sending multiple inference requests. By default, the session is reused to improve the performance on high latency networks (e.g. fewer SSL negotiations). If you are sending requests from multiple threads, set this to False.
        Returns
        -------
        List[Prediction]
            A list of prediction result.
        """
        buffer_bytes = serialize_image(image)
        files = {"file": buffer_bytes}
        data = {"metadata": metadata.json()} if metadata else None
        if reuse_session:
            session = self._session
        else:
            session = create_requests_session(
                self._url,
                0,
                {
                    "contentType": "multipart/form-data"
                },  # No retries for the inference service
            )
        (preds, self._performance_metrics) = get_cloudinference_prediction(
            session, self._url, files, {}, _EdgeExtractor, data=data
        )
        return preds


class _EdgeExtractor(PredictionExtractor):
    """A class that extract the raw Edge JSON inference result to Predict Results instances."""

    @staticmethod
    def _extract_edge_class_prediction(
        response: Dict[str, Any],
    ) -> List[ClassificationPrediction]:
        """Extract Edge Inference Classification prediction result from response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.
        Example input:
        {
            "type": "ClassificationPrediction",
            "predictions":
            {
                "score": 0.9951885938644409,
                "labelIndex": 0,
                "labelName": "Fire"
            }
        }
        """
        return [
            ClassificationPrediction(
                score=response["predictions"]["score"],
                label_index=response["predictions"]["labelIndex"],
                label_name=response["predictions"]["labelName"],
            )
        ]

    @staticmethod
    def _extract_edge_od_prediction(
        response: Dict[str, Any],
    ) -> List[ObjectDetectionPrediction]:
        """Extract Object Detection prediction result from edge inference response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.
        Example input:
        {
            "type": "ObjectDetectionPrediction",
            "predictions":
            {
                "196f6705-4486-42e4-8b48-325bba622f18":
                {
                    "score": 0.9939298629760742,
                    "defect_id": 72558,
                    "coordinates":
                    {
                        "xmin": 1523,
                        "ymin": 1417,
                        "xmax": 1982,
                        "ymax": 1801
                    },
                    "labelIndex": 1,
                    "labelName": "Screw"
                },
                "ca576098-5d97-4d6e-87e5-e3c93e0187d3":
                {
                    "score": 0.9845893383026123,
                    "defect_id": 72558,
                    "coordinates":
                    {
                        "xmin": 429,
                        "ymin": 1035,
                        "xmax": 651,
                        "ymax": 1207
                    },
                    "labelIndex": 1,
                    "labelName": "Screw"
                },
                "cdb4941b-db69-4952-8769-e057a470a94a":
                {
                    "score": 0.9666865468025208,
                    "defect_id": 72558,
                    "coordinates":
                    {
                        "xmin": 948,
                        "ymin": 1595,
                        "xmax": 1123,
                        "ymax": 1804
                    },
                    "labelIndex": 1,
                    "labelName": "Screw"
                }
            }
        }

        Returns
        -------
        A list of Object Detection predictions, each prediction is a dictionary with the following keys:
            id: str
            label: str
            score: float
            bboxes: Dict[str, int]
            E.g. {
                "id": "cdb4941b-db69-4952-8769-e057a470a94a",
                "label": "Screw",
                "score": 0.9666865468025208,
                "bboxes": {
                    "xmin": 948,
                    "ymin": 1595,
                    "xmax": 1123,
                    "ymax": 1804,
                }
            }
        """
        predictions = response["predictions"]
        return [
            ObjectDetectionPrediction(
                id=id,
                label_name=pred["labelName"],
                label_index=pred["labelIndex"],
                score=pred["score"],
                bboxes=(
                    pred["coordinates"]["xmin"],
                    pred["coordinates"]["ymin"],
                    pred["coordinates"]["xmax"],
                    pred["coordinates"]["ymax"],
                ),
            )
            for id, pred in predictions.items()
        ]

    @staticmethod
    def _extract_edge_seg_prediction(
        response: Dict[str, Any],
    ) -> List[SegmentationPrediction]:
        """Extract Segmentation prediction result from response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.
        Example input:
        {
            "type": "SegmentationPrediction",
            "model_id": "9315c71e-31af-451f-9b38-120e035e6240",
            "predictions": {
                "bitmaps": {
                    "1855c44a-215f-40d0-b627-9c4c83641df2": {
                        "bitmap": "84480Z",
                        "defectId": 74026,
                        "labelIndex": 2,
                        "labelName": "Cloud",
                        "score": 0
                    },
                    "c2e7372c-4d64-4078-a6ee-09bf4ef5084a": {
                        "bitmap": "84480Z",
                        "defectId": 74025,
                        "labelIndex": 1,
                        "labelName": "Sky",
                        "score": 0
                    }
                },
                "encoding": {
                    "algorithm": "rle",
                    "options": {
                        "map": {
                            "N": 1,
                            "Z": 0
                        }
                    }
                },
                "imageHeight": 240,
                "imageWidth": 352,
                "numClasses": 2
            },
            "latency": {
                "decoding_s": 0.0084266,
                "infer_s": 3.3537345,
                "postprocess_s": 0.0255059,
                "preprocess_s": 0.0124037,
                "waiting_s": 0.0001487
            }
        }
        """
        encoded_predictions = response["predictions"]["bitmaps"]
        encoding_map = response["predictions"]["encoding"]["options"]["map"]
        mask_shape = (
            response["predictions"]["imageHeight"],
            response["predictions"]["imageWidth"],
        )
        return [
            SegmentationPrediction(
                id=id,
                label_name=bitmap_info["labelName"],
                label_index=bitmap_info["labelIndex"],
                score=bitmap_info["score"],
                encoded_mask=bitmap_info["bitmap"],
                encoding_map=encoding_map,
                mask_shape=mask_shape,
            )
            for id, bitmap_info in encoded_predictions.items()
        ]

    @staticmethod
    def extract_prediction(response: Dict[str, Any]) -> List[Prediction]:
        response_type = response.get("type")
        if response_type is None:
            response_type = response["type"]
        predictions: List[Prediction]
        if response_type == "ObjectDetectionPrediction":
            predictions = _EdgeExtractor._extract_edge_od_prediction(response)  # type: ignore
        elif response_type == "SegmentationPrediction":
            predictions = _EdgeExtractor._extract_edge_seg_prediction(response)  # type: ignore
        elif response_type == "ClassificationPrediction":
            predictions = _EdgeExtractor._extract_edge_class_prediction(response)  # type: ignore
        else:
            raise NotImplementedError(
                f"{response_type} is not implemented in EdgeExtractor"
            )
        return predictions
