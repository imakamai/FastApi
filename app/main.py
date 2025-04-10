from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from functools import lru_cache
import motor.motor_asyncio
import os
import hashlib
import pypdf
from  .config import settings
from passlib.context import CryptContext
import jwt
import uuid
from functools import lru_cache
from .models import User, Conversation, LoginModel
from .config import Settings
from datetime import datetime, timedelta, timezone
app = FastAPI()

# MongoDB connection
MONGO_URL = os.getenv("ME_CONFIG_MONGODB_URL", "mongodb://fastapi:fastapi@mongo:27017/")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.get_database("Bridge")
user_collection = db.get_collection("users")
conversation_collection = db.get_collection("conversations")

user = None

#@lru_cache
#def get_settings():
    #return Settings

#Configuration for login and jtw
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "tajni_kljuc"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def generate_api_key(userId: str, accountId: str) -> str:
    """Generates API key by hashing username and password."""
    combined = f"{userId}{accountId}"
    return hashlib.sha256(combined.encode()).hexdigest()

@app.post("/register")
async def register_user(user: User):
    existing_user = await user_collection.find_one({"userId": user.userId})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    api_key = generate_api_key(user.userId, user.accountId)
    new_user = {
        "userId": user.userId,
        "accountId": user.accountId,
        "apiKey": api_key,
        "password": pwd_context.hash(user.password),
        "username": user.username,
    }
    await user_collection.insert_one(new_user)
    return {"message": "User registered successfully", "apiKey": api_key}



@app.get("/about")
async def about(tsttoken:str):
           try:
            payload = jwt.decode(tsttoken, options={"verify_signature": False})
           except jwt.InvalidTokenError:
               print("Invalid token")
           return {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "app_decription": settings.app_decription,
        }



@app.get("/models")
async def get_models():
    return {"Model_one": "gpt-4o", "Model_two": "gpt-4o-mini"}

@app.post("/conversation")
async def create_conversation(user: User,request:Request, tsttoken:str, coverstaionId: str):
    try:
        payload = jwt.decode(tsttoken, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        print("Invalid token")

    conversatrion = {
            "conversation_id": coverstaionId,
            "user_id" :user.userId,
            "account_id": user.accountId,
            "model": "gpt-4o-mini",
            "created_at": datetime.now().date().strftime('%yyyy-MM-dd'),
            "messages": list()
    }

    await conversation_collection.insert_one(dict(conversatrion))
    return {"contextId": "123e4567-e89b-12d3-a456-426614174000","welcomePrompt": "Hello! How can I assist you today?"}

@app.post("/get")
async def get_conversation( request:Request, tsttoken:str, coverstaion: Conversation):
    try:
        payload = jwt.decode(tsttoken, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        print("Invalid token")

    # user = await user_collection.find_one({"userId": userId})
    # if not user:
    #     raise HTTPException(status_code=404, detail="User not found, coverastion does not exist")
    # else:
        conversation = await conversation_collection.find_one({"conversation_id": coverstaion.conversation_id})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation does not exist")
        else:
            return dict(conversation)

@app.post("/add-message")
async def add_message(userId: str, conversationId: str, requste: Request, tsttoken:str):
    try:
        payload = jwt.decode(tsttoken, options={"verify_signature": False})
    except jwt.InvalidTokenError:

        print("Invalid token")

    user = await user_collection.find_one({"userId": userId})
    if not user:
        raise HTTPException(status_code=404, detail="User not found, coverastion does not exist")
    else:
        conversation = await conversation_collection.find_one({"userId": userId})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation does not exist")
        else:
            conversation.update({"messages": ["DDATATAT"]})



#login user
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@app.post("/login")
async def login(user: LoginModel):
    user_data = await user_collection.find_one({"username": user.username})

    if not user_data or not verify_password(user.password, user_data["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password.")

    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}