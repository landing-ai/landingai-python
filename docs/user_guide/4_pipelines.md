## Vision Pipelines

Pipelines can simplify complex vision tasks by breaking it into a sequence of operations that are applied to every image. Images are modeled as `landingai.vision_pipeline.Frame` and sequences of images as `landingai.vision_pipeline.FrameSet`. Most image sources will produce a `FrameSet` even when it contains a single image (e.g. `landingai.vision_pipeline.NetworkedCamera`). This is convenient as it allows subsequent stages of the pipelines to introduce new derived frames as part of the processing (e.g. when extracting faces from an initial Frame into new Frames).

The following example shows a basic pipeline starting from a single image:
```python
FrameSet.fromImage("sample_images/1196.png")
    .run_predict(predictor=...)
    .overlay_predictions()
    .show_image() 
    .show_image(image_src="overlay") 
```

For more details on the operations supported on pipelines check `landingai.vision_pipeline.FrameSet`
