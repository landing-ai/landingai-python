import os
import cv2

from landingai import Predictor

api_key = os.getenv("LANDING_API_KEY")
api_secret = os.getenv("LANDING_API_SECRET")

predictor = Predictor("fb3b4ff0-99ff-4a4c-b17c-76bbc72dcc70", api_key, api_secret)

frame = cv2.imread(r"/Users/william/Sync/Landing/Datasets/debris/cereal_1.jpg")

print(predictor.predict(frame))
