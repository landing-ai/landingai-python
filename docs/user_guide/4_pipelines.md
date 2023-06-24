## Vision Pipelines

Pipelines can simplify complex vision tasks by breaking them into a sequence of operations that are applied to every image. Images are modeled as `landingai.vision_pipeline.Frame` and sequences of images are modeled as `landingai.vision_pipeline.FrameSet`. Most image sources will produce a `FrameSet` even when it contains a single image. 

For example, a `landingai.vision_pipeline.NetworkedCamera` can connect to a live video source and expose it as `FrameSet` iterator. This is convenient as it allows subsequent stages of the pipeline to introduce new derived frames as part of the processing. 

As an other example, if we are detecting faces from a live stream, we may want to first use an object detection model to identify the regions of interest and then break the initial `Frame` into multiple frames (one per face). Subsequent stages of the pipeline may apply other models to individual faces.

The following example shows a basic pipeline that applies several processing layers starting from a single image. In this case after running inference (i.e. `run_predict`), the `overlay_predictions` function creates a copy of the original image (i.e. `frs[0].image`) and populates `frs[0].other_images["overlay"]` with the results. 

```python
frs= FrameSet.from_image("sample_images/1196.png")
frs.run_predict(predictor=...)
.overlay_predictions()
.show_image() 
.show_image(image_src="overlay") 

```

For more details on the operations supported on pipelines, go to `landingai.vision_pipeline.FrameSet`.
