import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email, subject, message):
    """
    Send an email using SMTP (e.g., Gmail).
    """
    try:
        # SMTP server details
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = "sohamsantra04@gmail.com"  # Replace with your Gmail address
        smtp_password = "bubd mmze jnav edro"  # Replace with your Gmail app password

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add HTML and plain text versions
        msg.attach(MIMEText(message, 'html'))
        msg.attach(MIMEText("Your vehicle has been detected. Please check your email for details.", 'plain'))

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to secure
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False