import time
import cv2

from landingai import Predictor

api_key = "LANDING_API_KEY"
api_secret = "LANDING_API_SECRET"
predictor = Predictor("fb3b4ff0-99ff-4a4c-b17c-76bbc72dcc70", api_key, api_secret)


def run():
    # connect to camera
    vid = cv2.VideoCapture(0)  # change to cv2.VideoCapture("rtsp://xxxxxx") as needed

    # create a window for visualization, need to figure this out
    #cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    # loop forever until ctrl+c pressed
    while True:
        try:
            # capture frame
            _, frame = vid.read()

            # run inference
            result = predictor.predict(frame)

            # overlay the predictions on the original image
            visualized = result.visualize(frame)

            # show the predictions
            # this doesn't work very well:
            #cv2.imshow("window", frame)
            #cv2.waitKey(1)

            # delay slightly before running again
            time.sleep(1)

        except KeyboardInterrupt:
            print("exiting...")
            break
        except Exception as err:
            print(f"unexpected error: {err}")

    # cleanup
    # remove the visualization window
    cv2.destroyWindow("window")
    # close the video stream
    vid.release()


if __name__ == "__main__":
    run()
