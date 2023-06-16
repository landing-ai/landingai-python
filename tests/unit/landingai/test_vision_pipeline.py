import cv2
import numpy as np

from landingai.vision_pipeline import NetworkedCamera


def test_networked_camera():
    # Use a video to simulate a live camera and test motion detection
    test_video_file_path = "tests/data/videos/countdown.mp4"
    Camera = NetworkedCamera(
        stream_url=test_video_file_path, motion_detection_threshold=50
    )

    # Get the first frame and the next frame where motion is detected. I keep the detection threshold low to make the test fast
    i = iter(Camera)
    print(f"FPS {Camera._inter_frame_interval}")
    frame1 = next(i)
    while True:
        # if we cannot get any motion detection (e.g. Threshold 100%), next() will throw an exception and fail the test
        frame2 = next(i)
        if not frame2.is_empty():
            break
    # frame1.show_image()
    # frame2.show_image()
    image_distance = np.sum(
        cv2.absdiff(src1=frame1[0].to_numpy_array(), src2=frame2[0].to_numpy_array())
    )

    # Compute the diff by summing the delta between each pixel across the two images
    del Camera
    assert (
        image_distance > 100000
    )  # Even with little motion this number should exceed 100k
