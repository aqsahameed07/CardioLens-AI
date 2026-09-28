import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "cardiolens")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not configured in .env")

client = MongoClient(MONGO_URI)

db = client[MONGO_DB]

users_collection = db["users"]
assessments_collection = db["assessments"]


def check_database_connection():
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False