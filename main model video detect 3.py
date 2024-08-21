import os
import torch
from ultralytics import YOLO
import cv2
import math
import cvzone

print(torch.cuda.is_available())  # Should return True if CUDA is properly installed
print(torch.cuda.get_device_name(0))  # Should return the name of the GPU (e.g., NVIDIA GPU)

# Ensure the dedicated GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use "0" for the first dedicated GPU

# Load a model
email_sent = False


def sendmail(image):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email import encoders
    from PIL import Image
    import io

    sender_email = "pixashield@gmail.com"
    receiver_email = "kunal4103@gmail.com"

    password = "bjxc lysq nizf qmle"

    # Create a message object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "ANOMALY DETECTED!!!"

    # Convert the image to RGB if it isn't already
    if image.shape[-1] == 3:  # If image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to PIL format and save to a byte stream
    pil_img = Image.fromarray(image)
    image_byte_array = io.BytesIO()
    pil_img.save(image_byte_array, format="PNG")
    image_byte_array.seek(0)

    # Attach image to the email
    img = MIMEImage(image_byte_array.read())
    img.add_header("Content-Disposition", "attachment", filename="image.png")
    message.attach(img)

    # Establish a connection with the SMTP server (for Gmail, use 'smtp.gmail.com' and port 587)
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        # Start TLS for security
        server.starttls()

        # Login to your email account
        server.login(sender_email, password)

        # Send the email
        server.send_message(message)

    print("Email with tensor image sent successfully!")


# Load the model and ensure it uses GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(r"C:\Users\Kunal\Desktop\RJPOLICE_HACK_238_PIXASHIELD_3-main\gunkaro_Naik\best (2).pt").to(device)

cap = cv2.VideoCapture(
    r"C:\Users\Kunal\Desktop\RJPOLICE_HACK_238_PIXASHIELD_3-main\gunkaro_Naik\air_gun.mp4")  # For Video
classNames = [
    "Grenade",
    "Handgun",
    "Rifle",
    "Steel arms",
    "Climbing",
    "Fall",
    "Violence",
    "Fire",
]

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB (as OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to be divisible by 32
    img_rgb = cv2.resize(img_rgb, (640, 640))

    # Convert to tensor, add batch dimension, and move to GPU
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    results = model(img_tensor, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.2:
                if currentClass in classNames:
                    myColor = (0, 0, 255)
                    if email_sent == False:
                        sendmail(img)
                        email_sent = True

                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    colorT=(255, 255, 255),
                    offset=5,
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
