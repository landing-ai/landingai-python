# User Guide

After you've developed a computer vision model, you can run inference with it and apply those results to serve your business goals. To help you make the most of your model, Landing AI offers the `landingai` Python library. This library enables you to use your LandingLens model and build your computer vision applications with ease.

Specifically, it provides Python APIs to allow you:
- Get prediction results from your deployed model (without worrying about communicating with the model server and different deployment options)
- Post-process your prediction results into other format
- Visualize your prediction results
- Chain multiple model inference and post-processing operations together
- Read image data from a variety of sources (RTSP streams, video file, Snowflake etc.)
- And more...

## High-level Concepts

This section explains important high-level concepts that will help you better use this Python library.

### Model Deployment Options

Before using this library for inference, you need train your model in LandingLens and [deploy it](https://support.landing.ai/docs/deployment-options).

This library supports two deployment options:
- [Cloud Deployment](https://support.landing.ai/landinglens/docs/cloud-deployment)
- LandingEdge (support coming soon...)

The easiest way to get started is using A Cloud Deployment.

### Inference and Predictor

You can use this `landingai` library to get inference results from a model you trained on LandingLens.
The library calls the endpoint of your deployed model for inference results. The endpoint could either be hosted on a LandingLens server (i.e. Cloud Deployment) or on your own server using LandingEdge.

The actual computation of model inference does not happen on the library side, it happens on the model server side.

This library abstracts out the complexity of server communication, error handling, and result parsing into a simple class called `landingai.predict.Predictor`. For more information, see `landingai.predict.Predictor` in the API doc.

### Prediction Results

Once you get inference results, you can process them further based on your business needs.

Depending on the AI task the result schema will be different. Here are the schema definitions for each project type:

- Classification:`landingai.common.ClassificationPrediction`
- Object Detection:`landingai.common.ObjectDetectionPrediction`
- Segmentation: `landingai.common.SegmentationPrediction`
- Visual Prompting: `landingai.common.SegmentationPrediction`

This library also defines a parent class, `landingai.common.Prediction`, which represent a generic prediction that contains all the shared attributes across all of the subclasses.

### Building sophisticated vision pipelines

Real world problems require a multi step approach in order to get results. For example: if we want to count the number and model of cars going over a bridge, we will need to take images from a live RTSP stream and act when motion is detected. We will also need to use object detection to identify individual cars and track them as they traverse the frame. We will then need to classify the cars and collect statistics to product the desired report.

While this image processing pipeline may seem daunting, it has a many commonalities with other vision problems. This is why we developed a vision pipeline concept where `Frames` produced by an image source can be passed across processing modules using method chaining. 

For more details about vision pipelines, look into [*Vision Pipelines user guide*](#vision-pipelines)
