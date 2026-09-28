from pydantic import BaseModel, EmailStr, Field


class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=70, le=250)
    chol: float = Field(..., ge=80, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

class ShapExplanation(BaseModel):
    feature: str
    value: float
    shap_value: float
    
class PredictionResponse(BaseModel):
    id: str
    prediction: int
    probability: float
    result: str
    explanation: list[ShapExplanation]


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse