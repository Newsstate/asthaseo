import os, smtplib
from email.mime.text import MIMEText

def send_email(to_email: str, subject: str, body: str):
    host = os.getenv("SMTP_SERVER")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL", user or "no-reply@example.local")
    if not host or not user or not pwd:
        # Email not configured; silently ignore
        return False
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
    return True