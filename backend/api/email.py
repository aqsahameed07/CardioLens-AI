import os
import smtplib
from html import escape
from email.message import EmailMessage

from dotenv import load_dotenv
from fastapi import HTTPException, status

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_SECURE = os.getenv("SMTP_SECURE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "ssl",
    "tls",
}
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL") or SMTP_USER


def _require_smtp_config():
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, FROM_EMAIL]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email delivery is not configured",
        )


def send_email(to_email: str, subject: str, text_body: str, html_body: str):
    _require_smtp_config()

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = FROM_EMAIL
    message["To"] = to_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if SMTP_SECURE and SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(message)
            return

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            if SMTP_SECURE or SMTP_PORT == 587:
                server.starttls()
                server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(message)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not send verification email. Please try again.",
        )


def send_verification_code(
    to_email: str,
    code: str,
    purpose: str,
    verification_url: str | None = None,
):
    if purpose == "signup":
        subject = "Verify your CardioLens account"
        intro = "Verify your CardioLens account using the secure link below."
    else:
        subject = "Reset your CardioLens password"
        intro = "Use this code to reset your CardioLens password."

    text_body = f"{intro}\n\n"
    if verification_url:
        text_body += (
            f"Verify your email: {verification_url}\n"
            "This link expires in 24 hours.\n\n"
        )
    text_body += (
        f"Your verification code is: {code}\n"
        "This code expires in 10 minutes. If you did not request this, "
        "you can ignore this email."
    )

    verification_link = ""
    if verification_url:
        verification_link = (
            '<p><a href="'
            + escape(verification_url, quote=True)
            + '" style="display:inline-block;padding:12px 18px;background:#e06b5a;'
            'color:#fff;text-decoration:none;border-radius:6px;font-weight:700;">'
            "Verify email</a></p>"
            "<p>This link expires in 24 hours. Or use this backup code:</p>"
        )

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; color: #1f1f1f;">
      <h2 style="margin-bottom: 8px;">CardioLens</h2>
      <p>{intro}</p>
    {verification_link}
      <p style="font-size: 28px; letter-spacing: 6px; font-weight: 700; color: #e06b5a;">
        {code}
      </p>
      <p>This code expires in 10 minutes.</p>
      <p style="color: #6b6b6b; font-size: 13px;">
        If you did not request this, you can ignore this email.
      </p>
    </div>
    """

    send_email(to_email, subject, text_body, html_body)
