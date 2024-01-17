import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
import av
import cv2
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import time
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat

st.set_page_config(
    page_title="PixaShield",  # the title of the webpage
    page_icon="logo.png",  # the icon in the browser tab
    # layout="centered",  # controlled the page layout
    # initial_sidebar_state="auto"  # initial state of the sidebar
)

# Add the login page to your Streamlit app
def show_login_page():
    logo_image = "logo.png"

    # Create a layout with two columns for the logo and the title
    col1, col2 = st.columns([1, 4])

    # Add the logo to the first column
    with col1:
        st.image(logo_image)

    # Add the title to the second column
    with col2:
        st.title("PixaShield : AI Intelligent Camera")
    st.markdown('<hr style="border: 2px solid #f63366;"/>', unsafe_allow_html=True)

    st.subheader("Login")

    with st.form(key='login_form'):
        username = st.text_input("Username", key='username')
        password = st.text_input("Password", type="password", key='password')

        submitted = st.form_submit_button("Login")

        if submitted:
            if username == "admin" and password == "password":
                st.experimental_set_query_params(logged_in=True)  # Set a query parameter to simulate redirection
            else:
                st.error("Invalid username or password. Please try again.")

def main_content():
    # Main content of your app goes here

    st.title("PixaShield : AI Intelligent Camera")
    st.markdown('<hr style="border: 2px solid #f63366;"/>', unsafe_allow_html=True)

    p_time = 0

st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLO Model', 'YOLOv8', 'YOLOv7')
)

st.title(f'{model_type} Predictions')
# sample_img = cv2.imread('logo.png')
# FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None

if not model_type == 'YOLO Model':
    path_model_file = st.sidebar.text_input(
        f'path to {model_type} Model:',
        f'eg: dir/{model_type}.pt'
    )
    if st.sidebar.checkbox('Load Model'):
        
        # YOLOv7 Model
        if model_type == 'YOLOv7':
            # GPU
            gpu_option = st.sidebar.radio(
                'PU Options:', ('CPU', 'GPU'))

            if not torch.cuda.is_available():
                st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
            else:
                st.sidebar.success(
                    'GPU is Available on this Device, Choose GPU for the best performance',
                    icon="✅"
                )
            # Model
            if gpu_option == 'CPU':
                model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom(path_or_model=path_model_file, gpu=True)

        # YOLOv8 Model
        if model_type == 'YOLOv8':
            from ultralytics import YOLO
            model = YOLO(path_model_file)

        # Load Class names
        class_labels = model.names

        # Inference Mode
        options = st.sidebar.radio(
            'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        color_pick_list = []
        for i in range(len(class_labels)):
            classname = class_labels[i]
            color = color_picker_fn(classname, i)
            color_pick_list.append(color)

        # Image
        if options == 'Image':
            upload_img_file = st.sidebar.file_uploader(
                'Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')
                file_bytes = np.asarray(
                    bytearray(upload_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                FRAME_WINDOW.image(img, channels='BGR')

                if pred:
                    img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                    FRAME_WINDOW.image(img, channels='BGR')

                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with st.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq, use_container_width=True)
        
        # Video
        if options == 'Video':
            upload_video_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv'])
            if upload_video_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(upload_video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                # if pred:


        # Web-cam
        if options == 'Webcam':
            cam_options = st.sidebar.selectbox('Webcam Channel',
                                            ('Select Channel', '0', '1', '2', '3'))
        
            if not cam_options == 'Select Channel':
                pred = st.checkbox(f'Predict Using {model_type}')
                cap = cv2.VideoCapture(int(cam_options))


        # RTSP
        if options == 'RTSP':
            rtsp_url = st.sidebar.text_input(
                'RTSP URL:',
                'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
            )
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(rtsp_url)


if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)

# Show the login page when the app starts
query_params = st.experimental_get_query_params()
logged_in = query_params.get("logged_in")

if logged_in:
    main_content()
else:
    show_login_page()

st.markdown(
        """
        <p style='text-align: justify;'><b><span style='color: #FF4B4B;'>Holovision: Holo Conferencing</span></b> is an innovative project aimed at integrating holographic technology with live camera feeds, offering a futuristic visual experience in real time. By leveraging cutting-edge holographic displays and advanced image processing, the project seeks to revolutionize the way we interact with live video content, opening new possibilities for immersive communication and entertainment.</p>
        """,
        unsafe_allow_html=True,
    )

warnings.resetwarnings()