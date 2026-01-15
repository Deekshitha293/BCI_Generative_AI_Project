import smtplib
from email.message import EmailMessage

def send_email_alert(command):
    sender_email = "deekshithadeekshithab293@gmail.com"
    sender_password = "iezb plkx euct ovbd"
    receiver_email = "aspiro527@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = f"BCI Alert: {command}"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(f"""
    🚨 Brain-Computer Interface Alert 🚨

    Patient has triggered the command: {command}

    Immediate attention may be required.
    """)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "📧 Email alert sent successfully!"
    except Exception as e:
        return f"❌ Email failed: {e}"
