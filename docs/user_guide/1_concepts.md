# User Guide

Turning your model inference result into data that serves your business goal often requires non-trivial effort. `landingai` is a Python library that enables you to use your LandingLens model and build your CV applications with ease.

Specifically, it provides Python APIs to allow you:
1. Get prediction results from your deployed model (without worrying about communication to the model server and different deployment options)
2. Post-process your prediction results into other format
3. Visualize your prediction results
4. Chain multiple model inference and post-processing operations together
5. Read image data from a variety of sources (RTSP stream, video file, snowflake etc.)
6. And more...

## High-level Concepts

This section explains important high-level concepts that helps you better use this Python library.

### Model Deployment Options

Before using this library for inference, you need to have a trained model in `LandingLens`, and have deployed the model via one of the [deployment options](https://support.landing.ai/docs/deployment-options).

This library supports two deployment options:
1. [CloudDeployment](https://support.landing.ai/landinglens/docs/cloud-deployment)
2. LandingEdge (coming soon...)

The easiest way to get started is using `CloudDeployment`.

### Inference and Predictor

You can use this `landingai` library to get inference results from a model you trained on `LandingLens`.
The library calls the endpoint of your deployed model for inference results. The endpoint could either be hosted on a `LandingLens`' server, i.e. `CloudDeployment` or on your own server via `LandingEdge`.

The actual computation of model inference does not happen on the library side, it happens on the model server side.

The library abstracted out the complexity of server communication, error handling, result parsing into a simple class called `landingai.predict.Predictor`. See the API doc of this class for more detail.

### Prediction Result

Once you get inference results, you can process them further based on your business needs.
It's helpful to know what result schema look like before processing them.

Before getting into the schema, it's helpful to know that this library supports 4 kinds of project types of LandingLens:
1. Object Detection
2. Segmentation
3. Classification
4. Visual Prompting

For each project type, the schema is defined as follow:
1. Classification:`landingai.common.ClassificationPrediction`
2. Object Detection:`landingai.common.ObjectDetectionPrediction`
3. Segmentation: `landingai.common.SegmentationPrediction`
4. Visual Prompting: `landingai.common.SegmentationPrediction`

The library also defined a parent class, `landingai.common.Prediction`, that represent a generic prediction that contains all the shared attributes across all the subclasses.
