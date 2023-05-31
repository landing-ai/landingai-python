## Manage API Credentials

Here are the three ways to configure your API Key and API Secret, ordered by the priority in which they are loaded:

1. Pass them as function parameters. See `landingai.predict.Predictor`.
2. Set them as environment variables. For example: `export LANDINGAI_API_KEY=...`, `export LANDINGAI_API_SECRET=...`.
3. Store them in an `.env` file under your project root directory. For example, here is a set of credentials in an `.env` file:

```
   LANDINGAI_API_KEY=v7b0hdyfj6271xy2o9lmiwkkcb12345
   LANDINGAI_API_SECRET=ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq312345
```