Running inferences with the standard `landingai.predict.Predictor` will send your image to LandingLens cloud, which is ideal if you don't want to worry about backend scalability, hardware, etc. But this also adds some networking overhead that might limit how many inferences per second you can run.

If you need to run several inferences per second, and you have your own cloud service or local machine, you might want to run inference using your own resources.

For that, we provide *Model Runner*, a Docker image with your LandingLens trained model embeded that you can run anywhere. See below for a guide on how to use that:

- [Requirements](inferences/model-runner/requirements.md)
- [Spinning Up Model Runner](inferences/model-runner/spinning-up.md)
- [Running Inferences](infereces/model-runner/running-inferences.md)
- [GPU Support](inferences/model-runner/gpu-support.md)
