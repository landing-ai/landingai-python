Just like [video files](image-acquisition/video-file.md), you can extract frames from your local webcam using `landingai.image_source.Webcam`.

The example below iterates over the first 100 frames captured by the webcam, resizes it and saves it to
a new video file.

```py
from landingai.pipeline.image_source import Webcam
from landingai.pipeline.frameset import FrameSet

frameset = FrameSet() # (1)!
with Webcam(fps=1) as camera: # (2)!
    for i, frame in enumerate(camera): # (3)!
        if i >= 100:
            break
        frame.resize(width=256) # (4)!
        frameset.append(frame) # (5)!
frameset.save_video("/tmp/resized-video.mp4") # (6)!
```

1. Creates an empty `FrameSet`, where we will store the modified frames
2. Capture frames from the webcam at 1 frame per second
3. Iterate over each frame of the video file
4. Resize the frame to `width=256px` (keeping aspect ratio)
5. Append the resized frame to the `FrameSet`
6. Save the resized video to `/tmp/resized-video.mp4`.

## Webcam on Collab and Remote Jupyter Notebooks

If you want to use your webcam while running a remote Jupyter Notebook or Google Collab, you might want to
capture images from your local webcam (the one in the computer running the web UI, not the
machine running the kernel).

In this case, `landingai.pipeline.image_source.Webcam` will not work. Instead, you should
capture the images from your webcam using `take_photo_from_webcam()` function, from `landingai.image_source_ops` package:

```py
from landingai.image_source_ops import take_photo_from_webcam
from landingai.pipeline.frameset import Frame

frame = Frame(image=take_photo_from_webcam())
frame.resize(width=256)
```