Just like [local webcams](image-acquisition/video-file.md), you can extract frames from IP cameras.

The example below iterates over the first 100 frames captured by the network camera, resizes it and saves it to
a new video file.

```py
from landingai.pipeline.image_source import NetworkedCamera
from landingai.pipeline.frameset import FrameSet

frameset = FrameSet() # (1)!
with NetworkedCamera(stream_url="rtsp://192.168.0.77:8080/h264_opus.sdp", fps=1) as camera: # (2)!
    for i, frame in enumerate(camera): # (3)!
        if i >= 100:
            break
        frame.resize(width=256) # (4)!
        frameset.append(frame) # (5)!
frameset.save_video("/tmp/resized-video.mp4") # (6)!
```

1. Creates an empty `FrameSet`, where we will store the modified frames
2. Connects to the IP camera using RTSP protocol, and captures frames at 1 frame per second
3. Iterate over each frame of the video file
4. Resize the frame to `width=256px` (keeping aspect ratio)
5. Append the resized frame to the `FrameSet`
6. Save the resized video to `/tmp/resized-video.mp4`.

It is also possible to only yield frames if the camera detects motion. This is useful for reducing the number of frames
processed when running inferences, for example:

```py
# motion_detection_threshold is a value between 0 and 100, as a percent of pixels changed from one frame to the next.
cam = NetworkedCamera(
    stream_url="rtsp://192.168.0.77:8080/h264_opus.sdp",
    fps=1,
    motion_detection_threshold=15,
)
with cam:
    for frame in cam:
        ...
```