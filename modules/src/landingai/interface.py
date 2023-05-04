class Predictor:
    import numpy as np
    from typing import Any

    def __init__(self, endpoint_id: str, api_key: str, api_secret: str):
        self._endpoint_id = endpoint_id
        self._api_key = api_key
        self._api_secret = api_secret

    def predict(self, image: np.ndarray) -> Any:
        import cv2
        import requests

        img = cv2.imencode(".jpg", image)[1]
        files = [("file", ("image.jpg", img, "image/jpg"))]
        headers = {
            'apikey': self._api_key,
            'apisecret': self._api_secret,
        }
        url = 'https://predict.app.landing.ai/inference/v1/predict'
        payload = {'endpoint_id': self._endpoint_id}
        response = requests.post(url, headers=headers, files=files, params=payload)
        return response.json()
