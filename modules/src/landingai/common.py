from pydantic import BaseSettings


class Credential(BaseSettings):
    api_key: str
    api_secret: str
