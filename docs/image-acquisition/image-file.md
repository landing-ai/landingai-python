Creating a frame out of an image file is pretty straightforward:
```py
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("/path/to/your/image.jpg")
frame.resize(width=512, height=512)
frame.save_image("/tmp/resized-image.png")
```

1. Open `/path/to/your/image.jpg` image file.
2. Resize the frame to `width=512px` (keeping aspect ratio)
3. Save the resized image to `/tmp/resized-image-<index>.png`.


Alternatively, if you have a folder with multiple images, you can iterate over that
using `landingai.image_source.ImageFolder`:

```py
from landingai.pipeline.image_source import ImageFolder

with ImageFolder("/path/to/your/images-dir/*.png") as image_folder:
    for i, frame in enumerate(image_folder): # (1)!
        frame.resize(width=512)  # (2)!
        frame.save_image(f"/tmp/resized-image-{i}.png") # (3)!
```

1. Iterate over all png images in `/path/to/your/image-dir`
2. Resize the frame to `width=512px` (keeping aspect ratio)
3. Save the resized image to `/tmp/resized-image-<index>.png`.