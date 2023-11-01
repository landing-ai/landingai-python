After [acquiring your images](image-acquisition/image-acquisition.md) as `Frame` objects in the [previous section](image-acquisition/image-acquisition.md), you can perform a variety of operations on them.

These operations are useful for pre-processing the images before [running inferences](inferences/getting-started.md) on them, especially if your model was trained with images acquired in slightly different conditions from what you have when running inference.

For example, if your train dataset images were taken in a room with more light, you might want to adjust the brightness and contrast before running inference on your frames.

Below, you can see the list of useful operations that can be performed on a `Frame` object.

## Cropping

Cropping is the operation of extracting regions of interest from an image.

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

You can also keep the original aspect ratio by passing only one of the dimensions:

```python
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("docs/images/cereal-ops/cereal.jpeg")
frame.resize(width=64)
frame.save_image("docs/images/cereal-ops/cereal-resized-width.jpeg")
```

![Cereal image](../images/cereal-ops/cereal.jpeg)
![Cereal resized only one dimention](../images/cereal-ops/cereal-resized-width.jpeg)

Resizing images before running inference can speed up the inference process. But don't downscale too much, or you might lose important image details.

## Color

You can adjust color intensity:

- No change: 1.0
- Less intensity: Less than 1.0
- More intensity: Greater than 1.0

See the following examples.

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

You can adjust the contrast intensity:

- No change: 1.0
- Less intensity: Less than 1.0
- More intensity: Greater than 1.0

See the following examples.

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

You can adjust the brightness intensity:

- No change: 1.0
- Less intensity: Less than 1.0
- More intensity: Greater than 1.0

See the following examples.

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

You can adjust the sharpness intensity:

- No change: 1.0
- Less intensity: Less than 1.0
- More intensity: Greater than 1.0

See the following examples.

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