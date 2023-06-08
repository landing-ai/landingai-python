from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from paddleocr import PaddleOCR

from landingai.common import OcrPrediction


class DetModel(int, Enum):
    AUTO_DETECT = 1
    MANUAL_ROI = 0


class OCRModel:
    def __init__(self, device: Optional[str] = "cpu") -> None:
        # Cache all the models during startup
        if device == "cpu":
            self.pp_inferencer = PaddleOCR(
                use_angle_cls=True, lang="en", page_num=10, use_gpu=False
            )
        else:
            self.pp_inferencer = PaddleOCR(
                use_angle_cls=True, lang="en", page_num=10, use_gpu=True
            )

    def __call__(self, img: List[np.ndarray], det_mode: str) -> Dict:
        result_list = self.pp_inferencer.ocr(img, det=bool(DetModel[det_mode].value))
        return result_list


# Pre-cache the models
if "ocr_model" not in st.session_state:
    model = OCRModel()
    st.session_state.ocr_model = model
else:
    model = st.session_state.ocr_model


class OcrPredictor:
    def __init__(self, det_mode: str, threshold: float) -> None:
        self._det_mode = det_mode
        self._threshold = threshold

    def predict(self, image, roi_boxes) -> List[OcrPrediction]:
        # TODO: crop image to ROI
        ocr_result = model(image, self._det_mode)
        if self._det_mode == DetModel.AUTO_DETECT.name:
            ocr_lines = ocr_result[0]
        else:
            assert roi_boxes, "ROI boxes are required for manual ROI detection"
            ocr_lines = [
                (box, (line[0][0], line[0][1]))
                for (box, line) in zip(roi_boxes, ocr_result)
            ]
        preds = []
        for i, (box, (text, score)) in enumerate(ocr_lines):
            if score < self._threshold:
                continue
            preds.append(
                OcrPrediction(
                    score=score,
                    text=text,
                    text_location=box,
                )
            )
        return preds
