from pydantic import BaseModel
from datetime import date


class User(BaseModel):
    userId: str
    accountId: str
    apiKey: str
    password: str
    username: str

class Message(BaseModel):
    userId: str
    text: str

class Conversation(BaseModel):
    conversation_id: str
    user_id : str
    account_id : str
    model: str
    created_at: date
    message: list[Message]

class LoginModel(BaseModel):
    username: str
    password: str

