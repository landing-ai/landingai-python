After [acquiring your image](image-acquisition/image-acquisition.md) as `Frame` objects in the previous section, you can perform a variety of operations on it.

Those operations are specially useful for pre-processing before [running inferences](inferences/getting-started.md) on them, specially if you are using a model that was trained with images acquired in slightly different conditions from what you have when running inferences.

For example, if your train dataset was taken on a lighter room, you might want to adjust a bit brightness and contrast before running inferences on your frames.

Below, you can see the list of useful operations that can be performed on a `Frame` object.

## Cropping

Cropping is the operation to extract regions of interest from an image.

```python
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.crop((16, 16, 112, 112))  # (1)!
frame.save_image("docs/images/cereal-ops/cereal-cropped.jpeg")
```

1. `(16, 16, 112, 112)` is the bounding-box coordinates to crop, meaning that the cropped image will start at `(16, 16)` and end at `(112, 112)`.

![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal cropped image](../images/cereal-ops/cereal-cropped.jpeg)


## Resizing

You can resize your image to any aspect ratio by passing both width and height:


```python
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.resize(width=64, height=128)
frame.save_image("docs/images/cereal-ops/cereal-resized-both.jpeg")
```

![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal resized both dimentions](../images/cereal-ops/cereal-resized-both.jpeg)

Or keeping the original aspect ratio by passing only one of the dimensions.

```python
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.resize(width=64)
frame.save_image("docs/images/cereal-ops/cereal-resized-width.jpeg")
```

![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal resized only one dimention](../images/cereal-ops/cereal-resized-width.jpeg)

When running inferences, resizing images can help reduce the traffic and achieve faster inferences. But don't downscale it too much, or you might lose important image details.

## Color

You can adjust color intensity. The factor parameter must be 1.0 for no change, less than 1.0 for less intensity and greater than 1.0 for more intensity.

```python
from landingai.pipeline.frameset import Frame

# Adjust color to 0.1
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_color(0.1)
frame.save_image("docs/images/cereal-ops/cereal-color-0.1.jpeg")

# Adjust color to 2.0
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_color(2.0)
frame.save_image("docs/images/cereal-ops/cereal-color-2.0.jpeg")
```

![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal low color adjustment](../images/cereal-ops/cereal-color-0.1.jpeg)
![Cereal high color adjustment](../images/cereal-ops/cereal-color-2.0.jpeg)

## Contrast

Similar to color, you can adjust the contrast. The factor parameter must be 1.0 for no change, less than 1.0 for less intensity and greater than 1.0 for more intensity.

```python
from landingai.pipeline.frameset import Frame

# Adjust contrast to 0.1
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_contrast(0.1)
frame.save_image("docs/images/cereal-ops/cereal-contrast-0.1.jpeg")

# Adjust contrast to 2.0
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_contrast(2.0)
frame.save_image("docs/images/cereal-ops/cereal-contrast-2.0.jpeg")
```


![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal low contrast adjustment](../images/cereal-ops/cereal-contrast-0.1.jpeg)
![Cereal high contrast adjustment](../images/cereal-ops/cereal-contrast-2.0.jpeg)

## Brightness

To adjust brightness, use the same factor parameter: 1.0 for no change, less than 1.0 for less intensity and greater than 1.0 for more intensity.

```python
from landingai.pipeline.frameset import Frame

# Adjust brightness to 0.1
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_brightness(0.1)
frame.save_image("docs/images/cereal-ops/cereal-brightness-0.1.jpeg")

# Adjust brightness to 2.0
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_brightness(2.0)
frame.save_image("docs/images/cereal-ops/cereal-brightness-2.0.jpeg")
```


![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal low brightness adjustment](../images/cereal-ops/cereal-brightness-0.1.jpeg)
![Cereal high brightness adjustment](../images/cereal-ops/cereal-brightness-2.0.jpeg)

## Sharpness

Adjusting sharpness follows the same pattern as color, brightness and contrast: 1.0 for no change, less than 1.0 for less intensity and greater than 1.0 for more intensity.

```python
from landingai.pipeline.frameset import Frame

# Adjust sharpness to 0.1
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_sharpness(0.1)
frame.save_image("docs/images/cereal-ops/cereal-sharpness-0.1.jpeg")

# Adjust sharpness to 2.0
frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.adjust_sharpness(2.0)
frame.save_image("docs/images/cereal-ops/cereal-sharpness-2.0.jpeg")
```


![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal low sharpness adjustment](../images/cereal-ops/cereal-sharpness-0.1.jpeg)
![Cereal high sharpness adjustment](../images/cereal-ops/cereal-sharpness-2.0.jpeg)