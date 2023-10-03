You can extract frames out of video files using `landingai.image_source.VideoFile`.

The example below iterates over each frame of a video, resizes it and saves it to
a new video file.

```py
from landingai.pipeline.image_source import VideoFile
from landingai.pipeline.frameset import FrameSet

frameset = FrameSet() # (1)!
with VideoFile("/path/to/your/file.mp4", samples_per_second=5) as video_file: # (2)!
    for frame in video_file: # (3)!
        frame.resize(width=256) # (4)!
        frameset.append(frame) # (5)!
frameset.save_video("/tmp/resized-video.mp4") # (6)!
```

1. Creates an empty `FrameSet`, where we will store the modified frames
2. Open `/path/to/your/file.mp4` video file, at a rate of 5 frames per second.
3. Iterate over each frame of the video file
4. Resize the frame to `width=256px` (keeping aspect ratio)
5. Append the resized frame to the `FrameSet`
6. Save the resized video to `/tmp/resized-video.mp4`.

