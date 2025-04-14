from pydantic_settings import BaseSettings

OPENAI_API_KEY = "tvoj-api-kljuc-ovde"
class Settings(BaseSettings):
    app_name: str = "Bridge API."
    app_version: float = 1.0
    app_decription: str = "This is RestAPI application for creating and processing Conversations."



settings = Settings()
