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

Before using this library for inference, you need train your model in `LandingLens` and [deploy it](https://support.landing.ai/docs/deployment-options).

This library supports two deployment options:
- [Cloud Deployment](https://support.landing.ai/landinglens/docs/cloud-deployment)
- LandingEdge (support coming soon...)

The easiest way to get started is using `CloudDeployment`.

### Inference and Predictor

You can use this `landingai` library to get inference results from a model you trained on `LandingLens`.
The library calls the endpoint of your deployed model for inference results. The endpoint could either be hosted on a `LandingLens` server (`CloudDeployment`) or on your own server via `LandingEdge`.

The actual computation of model inference does not happen on the library side, it happens on the model server side.

This library has abstracted out the complexity of server communication, error handling, and result parsing into a simple class called `landingai.predict.Predictor`. For more information, see `landingai.predict.Predictor` in the API doc.

### Prediction Result

Once you get inference results, you can process them further based on your business needs.

It's helpful to know what the result schema will look like before you process the results. Here are the scema definitions for each project type:

- Classification:`landingai.common.ClassificationPrediction`
- Object Detection:`landingai.common.ObjectDetectionPrediction`
- Segmentation: `landingai.common.SegmentationPrediction`
- Visual Prompting: `landingai.common.SegmentationPrediction`

This library also defines a parent class, `landingai.common.Prediction`, which represent a generic prediction that contains all the shared attributes across all of the subclasses.
