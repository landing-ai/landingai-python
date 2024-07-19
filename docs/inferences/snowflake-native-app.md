If you are hosting LandingLens as a Snowflake Native App, you need to use the `SnowflakeNativeAppPredictor` class to run inference.

In order to use this predictor class, you must first install the `snowflake` optionals when installing the `landingai` package. You can do this by running:

```sh
pip install "landingai[snowflake]"
```

Here is an example of how to use the `SnowflakeNativeAppPredictor` class:

```py
from landingai.pipeline.image_source import Webcam
from landingai.predict import SnowflakeNativeAppPredictor

endpoint_id = "c4344971-fc3c-4cb8-8fd5-5144d25cbd74"
url = "https://focq4dkf-rkwerpo-your-account.snowflakecomputing.app"

predictor = SnowflakeNativeAppPredictor(
    endpoint_id=endpoint_id, # (1)!
    native_app_url=url, # (2)!
    snowflake_account="your-snowflake-account-locator", # (3)!
    snowflake_user="your-snowflake-user",  # (4)!
    snowflake_private_key="your-snowflake-user-private-key",  # (5)!
)

frame = Frame.from_image("/home/user/dataset/some-class/image-1.png")
frame.run_predict(predictor=predictor)
print(frame.predictions)
# [
#   ClassificationPrediction(
#       score=0.9957893490791321,
#       label_name='some-class',
#       label_index=0
#   )
# ]
```

1. User the endpoint ID you created in LandingLens.
2. The URL you use to access the Snowflake Native App. Should start with `https://`, and have no trailing slash or path in the end.
3. Your Snowflake account locator, in the format `ABC01234`
4. Your Snowflake user name. Keep in mind that this Snowflake user must have access to the application.
5. Your Snowflake user private key. This key is used to authenticate the user. Alternativelly, you can pass the `snowflake_password` parameter with the user password instead of the private key, but this is not recommended in
production environment.

## Creating a user

When using the `SnowflakeNativeAppPredictor` class, the preferred authentication mechanism for production
environments is to use a private key. To create a user with a private key, you can run the following SQL
commands in your Snowflake account:

```sql
CREATE OR REPLACE USER LANDING_LIBRARY_USER
    LOGIN_NAME = 'LANDING_LIBRARY_USER'
    DISPLAY_NAME = 'LandingAI Library User'
    COMMENT = 'User for LandingAI library'
    RSA_PUBLIC_KEY = 'MIIBI...';  -- Replace this with your public key
CREATE ROLE LANDINGLENS_EXTERNAL_ACCESS;
GRANT ROLE LANDINGLENS_EXTERNAL_ACCESS
    TO USER LANDING_LIBRARY_USER;
GRANT APPLICATION ROLE llens_snw_production.LLENS_PUBLIC
    TO ROLE LANDINGLENS_EXTERNAL_ACCESS;
```

See [key-pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth) on Snowflake's documentation for more information on how to generate a key pair.
