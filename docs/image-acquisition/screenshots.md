In case you want to track what is happening in your desktop, you can use `landingai.image_source.Screenshot` to capture screenshots of your desktop.

In the example below, we capture the first 20 frames, and save them to a video file.

```py
from landingai.pipeline.image_source import Screenshot
from landingai.pipeline.frameset import FrameSet
import time

frameset = FrameSet() # (1)!
with Screenshot() as screenshots: # (2)!
    for i, frame in enumerate(screenshots): # (3)!
        if i >= 20:
            break
        frame.resize(width=512) # (4)!
        frameset.append(frame) # (5)!
        time.sleep(0.5) # (6)!
frameset.save_video("/tmp/resized-video.mp4") # (7)!
```

1. Creates an empty `FrameSet`, where we will store the modified frames
2. Build the screenshot capture object
3. Iterate over each frame captured from the desktop
4. Resize the frame to `width=512px` (keeping aspect ratio)
5. Append the resized frame to the `FrameSet`
6. Wait for 0.5 seconds before capturing the next frame
7. Save the resized video to `/tmp/resized-video.mp4`.
