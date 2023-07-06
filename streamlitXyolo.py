import streamlit as st
import torch
from PIL import Image
import cv2

def frame_detect(img, size=None):
    model.conf = confidence
    product = model(img, size=size) if size else model(img)
    product.render()
    image = Image.fromarray(product.ims[0])
    return image


def image_processing(uploaded_image):
    _uploaded_image = uploaded_image
    if _uploaded_image:
        
        image = Image.open(_uploaded_image)
    
        original, detected = st.columns(2)
        with original: 
            st.image(image, caption = "Original")
        with detected:
            processed_img = frame_detect(image)
            st.image(processed_img, caption = "detected")
              
def video_processing(uploaded_video):
    _uv = uploaded_video
    if _uv:
        vid_path = "data/upload." + _uv.name.split('.')[-1]
        with open(vid_path, 'wb') as out:
            out.write(_uv.read())

    if vid_path:
        cv_vid = cv2.VideoCapture(vid_path)
        
        width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
             
        output = st.empty()
        while True:
            ret, frame = cv_vid.read()
            if not ret:
                st.write("Can't read frame")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = frame_detect(frame)
            output.image(output_img)

        cv_vid.release()
        
        

def main():
    # global variables
    global model, confidence
    
    confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=0.5)
    
    # pretrained YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Image and video download
    
    
    uploaded_image = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"])
    uploaded_video = st.file_uploader("Or video here", type=['mp4', 'mpv', 'avi'])
    
    if uploaded_image: 
        image_processing(uploaded_image)
    
    if uploaded_video:
        video_processing(uploaded_video)
           
    
        

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass