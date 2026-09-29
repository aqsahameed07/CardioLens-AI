from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING

from api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    resend_verification,
    reset_password,
    start_password_reset,
    start_signup,
    verify_email,
    verify_email_link,
    verify_reset_code,
)
from api.database import (
    assessments_collection,
    check_database_connection,
)
from api.predictor import predict_heart_disease
from api.schemas import (
    AuthResponse,
    EmailRequest,
    LoginRequest,
    MessageResponse,
    PatientData,
    PredictionResponse,
    ResetPasswordRequest,
    SignupRequest,
    UserResponse,
    VerifyCodeRequest,
    VerifyLinkRequest,
)


app = FastAPI(
    title="CardioLens API",
    description="Heart disease prediction API using Machine Learning",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "CardioLens API is running"
    }


@app.get("/health")
def health():
    database_status = check_database_connection()

    return {
        "status": "healthy",
        "database": "connected" if database_status else "disconnected",
    }


@app.post(
    "/auth/signup",
    response_model=MessageResponse,
)
def signup(data: SignupRequest):
    return start_signup(
        name=data.name,
        email=data.email,
        password=data.password,
    )


@app.post(
    "/auth/verify-email",
    response_model=AuthResponse,
)
def confirm_email(data: VerifyCodeRequest):
    return verify_email(
        email=data.email,
        code=data.code,
    )


@app.post(
    "/auth/verify-email-link",
    response_model=AuthResponse,
)
def confirm_email_link(data: VerifyLinkRequest):
    return verify_email_link(token=data.token)


@app.post(
    "/auth/resend-verification",
    response_model=MessageResponse,
)
def resend_email_verification(data: EmailRequest):
    return resend_verification(email=data.email)


@app.post(
    "/auth/forgot-password",
    response_model=MessageResponse,
)
def forgot_password(data: EmailRequest):
    return start_password_reset(email=data.email)


@app.post(
    "/auth/verify-reset-code",
    response_model=MessageResponse,
)
def confirm_reset_code(data: VerifyCodeRequest):
    return verify_reset_code(
        email=data.email,
        code=data.code,
    )


@app.post(
    "/auth/reset-password",
    response_model=MessageResponse,
)
def update_password(data: ResetPasswordRequest):
    return reset_password(
        email=data.email,
        code=data.code,
        password=data.password,
    )


@app.post(
    "/auth/login",
    response_model=AuthResponse,
)
def login(data: LoginRequest):

    user = authenticate_user(
        email=data.email,
        password=data.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user_data = {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
    }

    token = create_access_token(
        user_data["id"]
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data,
    }


@app.get(
    "/auth/me",
    response_model=UserResponse,
)
def get_me(
    current_user=Depends(get_current_user),
):

    return {
        "id": str(current_user["_id"]),
        "name": current_user["name"],
        "email": current_user["email"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    data: PatientData,
    current_user=Depends(get_current_user),
):

    return predict_heart_disease(
        data,
        str(current_user["_id"]),
    )


@app.get("/assessments")
def get_assessments(
    current_user=Depends(get_current_user),
):

    user_id = current_user["_id"]

    assessments = list(
        assessments_collection.find(
            {"user_id": user_id}
        ).sort(
            "created_at",
            DESCENDING,
        )
    )

    results = []

    for assessment in assessments:
        results.append(
            {
                "id": str(assessment["_id"]),
                "patient_data": assessment["patient_data"],
                "prediction": assessment["prediction"],
                "probability": assessment["probability"],
                "result": assessment["result"],
                "model": assessment["model"],
                "created_at": assessment["created_at"],
            }
        )

    return results


@app.get("/assessments/{assessment_id}")
def get_assessment(
    assessment_id: str,
    current_user=Depends(get_current_user),
):

    from bson import ObjectId

    try:
        object_id = ObjectId(assessment_id)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment ID",
        )

    assessment = assessments_collection.find_one(
        {
            "_id": object_id,
            "user_id": current_user["_id"],
        }
    )

    if not assessment:
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    return {
        "id": str(assessment["_id"]),
        "patient_data": assessment["patient_data"],
        "prediction": assessment["prediction"],
        "probability": assessment["probability"],
        "result": assessment["result"],
        "model": assessment["model"],
        "created_at": assessment["created_at"],
    }