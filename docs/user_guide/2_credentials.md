## Manage API Credentials

If you send images to an endpoint through API (Cloud Deployment), you must add your API Key to the API call. You can generate the API Key in LandingLens. This API key is also known as API key v2. See [here](https://support.landing.ai/docs/api-key-and-api-secret) for more information.

Once you have generated the API key, here are three ways to configure your API Key, ordered by the priority in which they are loaded:

1. Pass it as function parameters. See `landingai.predict.Predictor`.
2. Set it as environment variables. For example: `export LANDINGAI_API_KEY=...`.
3. Store it in an `.env` file under your project root directory. For example, here is a set of credentials in an `.env` file:

```
   LANDINGAI_API_KEY=land_sk_v7b0hdyfj6271xy2o9lmiwkkcb12345
```

### Legacy API key and secret

In the past, LandingLens supports generating a key and secret pair, which is known as API key v1. This key is no longer supported in `landingai` Python package in version `0.1.0` and above.

See [here](https://support.landing.ai/docs/api-key) for how to generate API v2 key.

### FAQ

#### What's the difference between the v1 API key and v2 API key

Here are a few differences:
1. The v2 API key always starts with a prefix "land_sk_" whereas the v1 API key doesn't.
2. The v1 API key always comes with a secret string, and the SDK requires both whereas the v2 API key only has a single string.
3. Users can only generate v2 API keys after July 2023.
