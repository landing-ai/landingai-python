# User Guide

After you've developed a computer vision model, you can run inference with it and apply those results to serve your business goals. To help you make the most of your model, Landing AI offers the `landingai` Python library. This library enables you to use your LandingLens model and build your computer vision applications with ease.

Specifically, it provides Python APIs to allow you to:
- Get prediction results from your deployed model (without worrying about communicating with the model server and different deployment options)
- Post-process your prediction results into other formats
- Visualize your prediction results
- Chain multiple model inference and post-processing operations together
- Read image data from a variety of sources (RTSP streams, video file, Snowflake, etc.)
- Upload images to LandingLens from various sources (RTSP streams, video file, Snowflake, etc.)
- And more...

## High-Level Concepts

This section explains important high-level concepts that will help you better use this Python library.

### Model Deployment Options

Before using this library for inference, you need train your model in LandingLens and [deploy it](https://support.landing.ai/docs/deployment-options).

This library supports two deployment options:
- [Cloud Deployment](https://support.landing.ai/landinglens/docs/cloud-deployment)
- [Edge Deployment] (support coming soon...)

The easiest way to get started is using a Cloud Deployment.

### Inference and Predictor

You can use this `landingai` library to get inference results from a model you trained on LandingLens.
The library calls the endpoint of your deployed model for inference results. The endpoint could either be hosted on a LandingLens server (i.e. Cloud Deployment) or on your own server using LandingEdge.

The actual computation of model inference does not happen on the library side, it happens on the model server side.

This library abstracts out the complexity of server communication, error handling, and result parsing into a simple class called `landingai.predict.Predictor`. For more information, see `landingai.predict.Predictor` in the API doc.


#### Cloud Inference Limitations

The `Predictor` class uses Cloud Deployment by default.

1. **Rate Limit**

For non-enterprise users, you can make inference up to 40 inferences per minute per organization. If you exceed that limit, the cloud delpoyment server returns a 429 Too Many Requests response status code.

Reference: https://community.landing.ai/c/ask-the-community/friday-facts-inference-limits

If you need to make more frequent calls (i.e. more inference throughput), consider using the second deployment option (Edge Deployment), which is currently under alpha testing. Reach out to us at support@landing.ai for more information.

### Prediction Results

Once you get inference results, you can process them further based on your business needs.

Depending on the AI task the result schema will be different. Here are the schema definitions for each project type:

- Classification:`landingai.common.ClassificationPrediction`
- Object Detection:`landingai.common.ObjectDetectionPrediction`
- Segmentation: `landingai.common.SegmentationPrediction`
- Visual Prompting: `landingai.common.SegmentationPrediction`

This library also defines a parent class, `landingai.common.Prediction`, which represent a generic prediction that contains all the shared attributes across all of the subclasses.

### Build Sophisticated Vision Pipelines

Real world problems require a multi-step approach in order to get results. For example: if we want to produce statistics about the cars going over a bridge, we will need to take images from a live RTSP stream and act when motion is detected. We will also need to use object detection to identify individual cars and track them as they traverse the frame. We will then need to classify the cars and collect statistics to produce the desired report.

While this image processing pipeline may seem daunting, it has many commonalities with other vision problems. This is why we developed a vision pipeline concept where `Frames` produced by an image source can be passed across processing modules using method chaining. 

For more details about vision pipelines, go to the [*Vision Pipelines User Guide*](#vision-pipelines)
