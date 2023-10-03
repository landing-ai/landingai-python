You can extract frames out of video files using `landingai.image_source.VideoFile`:

```py
from landingai.pipeline.image_source import VideoFile, FrameSet

frameset = FrameSet()
with VideoFile("/path/to/your/file.mp4", samples_per_second=5) as video_file:
    for frame in video_file:
        frame.resize(width=256)
        frameset.append(frame)
frameset.save_video("/tmp/resized-video.mp4")
```

1. Open `/path/to/your/image.jpg` image file.
2. Resize the frame to `width=512px` (keeping aspect ratio)
3. Save the resized image to `/tmp/resized-image-<index>.png`.

