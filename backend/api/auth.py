import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pwdlib import PasswordHash
from pymongo.errors import DuplicateKeyError

from api.database import users_collection
from api.email import send_verification_code

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not configured in .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
CODE_EXPIRY_MINUTES = 10
LINK_EXPIRY_HOURS = 24
RESEND_COOLDOWN_SECONDS = 60
MAX_CODE_ATTEMPTS = 5
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")

password_hash = PasswordHash.recommended()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "exp": expire,
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def _normalize_email(email: str) -> str:
    return email.lower().strip()


def _hash_code(code: str) -> str:
    return hmac.new(
        JWT_SECRET.encode("utf-8"),
        code.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _codes_match(code: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False

    return hmac.compare_digest(_hash_code(code), stored_hash)


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _generate_verification_token() -> str:
    return secrets.token_urlsafe(32)


def _is_verified(user: dict) -> bool:
    if "email_verified" not in user:
        return True

    return bool(user.get("email_verified"))


def _assert_can_send(user: dict | None, sent_at_field: str):
    if not user:
        return

    last_sent = user.get(sent_at_field)

    if not last_sent:
        return

    if last_sent.tzinfo is None:
        last_sent = last_sent.replace(tzinfo=timezone.utc)

    elapsed = (datetime.now(timezone.utc) - last_sent).total_seconds()

    if elapsed < RESEND_COOLDOWN_SECONDS:
        wait = int(RESEND_COOLDOWN_SECONDS - elapsed) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Please wait {wait} seconds before requesting another code",
        )


def _public_user(user: dict, user_id: str | None = None):
    return {
        "id": user_id or str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
    }


def _finish_email_verification(user: dict):
    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "email_verified": True,
                "email_verified_at": datetime.now(timezone.utc),
            },
            "$unset": {
                "verification_code_hash": "",
                "verification_expires": "",
                "verification_attempts": "",
                "verification_token_hash": "",
                "verification_link_expires": "",
            },
        },
    )

    user_data = _public_user(user)
    return {
        "access_token": create_access_token(user_data["id"]),
        "token_type": "bearer",
        "user": user_data,
    }


def start_signup(name: str, email: str, password: str):
    email = _normalize_email(email)
    now = datetime.now(timezone.utc)

    existing_user = users_collection.find_one({"email": email})

    if existing_user and _is_verified(existing_user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists",
        )

    _assert_can_send(existing_user, "verification_sent_at")

    code = _generate_code()
    token = _generate_verification_token()
    code_fields = {
        "verification_code_hash": _hash_code(code),
        "verification_expires": now + timedelta(minutes=CODE_EXPIRY_MINUTES),
        "verification_token_hash": _hash_code(token),
        "verification_link_expires": now + timedelta(hours=LINK_EXPIRY_HOURS),
        "verification_attempts": 0,
        "verification_sent_at": now,
        "email_verified": False,
        "name": name.strip(),
        "password": hash_password(password),
    }

    if existing_user:
        users_collection.update_one(
            {"_id": existing_user["_id"]},
            {"$set": code_fields},
        )
    else:
        user = {
            **code_fields,
            "email": email,
            "created_at": now,
        }

        try:
            users_collection.insert_one(user)
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists",
            )

    send_verification_code(
        email,
        code,
        "signup",
        f"{FRONTEND_URL}/signup?verification_token={token}",
    )

    return {
        "message": "Verification code sent to your email",
        "email": email,
    }


def resend_verification(email: str):
    email = _normalize_email(email)
    user = users_collection.find_one({"email": email})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this email",
        )

    if _is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already verified",
        )

    _assert_can_send(user, "verification_sent_at")

    code = _generate_code()
    token = _generate_verification_token()
    now = datetime.now(timezone.utc)

    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "verification_code_hash": _hash_code(code),
                "verification_expires": now + timedelta(minutes=CODE_EXPIRY_MINUTES),
                "verification_token_hash": _hash_code(token),
                "verification_link_expires": now + timedelta(hours=LINK_EXPIRY_HOURS),
                "verification_attempts": 0,
                "verification_sent_at": now,
            }
        },
    )

    send_verification_code(
        email,
        code,
        "signup",
        f"{FRONTEND_URL}/signup?verification_token={token}",
    )

    return {
        "message": "Verification code sent to your email",
        "email": email,
    }


def verify_email(email: str, code: str):
    email = _normalize_email(email)
    user = users_collection.find_one({"email": email})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code",
        )

    if _is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already verified",
        )

    expires = user.get("verification_expires")

    if expires and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)

    if not expires or expires < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code has expired. Request a new one.",
        )

    attempts = user.get("verification_attempts", 0)

    if attempts >= MAX_CODE_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many attempts. Request a new verification code.",
        )

    if not _codes_match(code.strip(), user.get("verification_code_hash")):
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$inc": {"verification_attempts": 1}},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code",
        )

    return _finish_email_verification(user)


def verify_email_link(token: str):
    user = users_collection.find_one(
        {"verification_token_hash": _hash_code(token)}
    )

    if not user or _is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This verification link is invalid or has already been used",
        )

    expires = user.get("verification_link_expires")
    if expires and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)

    if not expires or expires < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification link has expired. Request a new one.",
        )

    return _finish_email_verification(user)


def start_password_reset(email: str):
    email = _normalize_email(email)
    user = users_collection.find_one({"email": email})

    generic = {
        "message": "If an account exists for this email, a reset code has been sent",
        "email": email,
    }

    if not user:
        return generic

    _assert_can_send(user, "reset_sent_at")

    code = _generate_code()
    now = datetime.now(timezone.utc)

    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "reset_code_hash": _hash_code(code),
                "reset_expires": now + timedelta(minutes=CODE_EXPIRY_MINUTES),
                "reset_attempts": 0,
                "reset_sent_at": now,
            }
        },
    )

    send_verification_code(email, code, "reset")

    return generic


def verify_reset_code(email: str, code: str):
    _load_valid_reset_user(email, code)

    return {
        "message": "Reset code verified",
        "email": _normalize_email(email),
    }


def reset_password(email: str, code: str, password: str):
    user = _load_valid_reset_user(email, code)

    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "password": hash_password(password),
                "email_verified": True,
                "email_verified_at": datetime.now(timezone.utc),
            },
            "$unset": {
                "reset_code_hash": "",
                "reset_expires": "",
                "reset_attempts": "",
                "verification_code_hash": "",
                "verification_expires": "",
                "verification_attempts": "",
            },
        },
    )

    return {
        "message": "Password updated successfully",
        "email": user["email"],
    }


def _load_valid_reset_user(email: str, code: str):
    email = _normalize_email(email)
    user = users_collection.find_one({"email": email})

    invalid = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid or expired reset code",
    )

    if not user:
        raise invalid

    expires = user.get("reset_expires")

    if expires and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)

    if not expires or expires < datetime.now(timezone.utc):
        raise invalid

    attempts = user.get("reset_attempts", 0)

    if attempts >= MAX_CODE_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many attempts. Request a new reset code.",
        )

    if not _codes_match(code.strip(), user.get("reset_code_hash")):
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$inc": {"reset_attempts": 1}},
        )
        raise invalid

    return user


def authenticate_user(email: str, password: str):
    email = _normalize_email(email)

    user = users_collection.find_one({"email": email})

    if not user:
        return None

    if not verify_password(password, user["password"]):
        return None

    if not _is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before signing in",
        )

    return user


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[ALGORITHM],
        )

        user_id = payload.get("sub")

        if not user_id:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    from bson import ObjectId

    try:
        user = users_collection.find_one(
            {"_id": ObjectId(user_id)}
        )
    except Exception:
        raise credentials_exception

    if not user:
        raise credentials_exception

    if not _is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before continuing",
        )

    return user
