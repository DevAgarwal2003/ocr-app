import streamlit as st
import zipfile
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, TFAutoModelForSeq2SeqLM
import torch
import pyttsx3
import tensorflow as tf
from gtts import gTTS
import io
import pygame


# Preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.GaussianBlur(1))
    return image

# Segment lines from the preprocessed image
def segment_lines_opencv(image, min_line_height=10):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.width // 100, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    
    lines = []
    if not bounding_boxes:
        return lines
    
    current_box = bounding_boxes[0]
    for box in bounding_boxes[1:]:
        if box[1] <= current_box[1] + current_box[3] + min_line_height:
            current_box = (
                min(current_box[0], box[0]),
                min(current_box[1], box[1]),
                max(current_box[0]+current_box[2], box[0]+box[2]) - min(current_box[0], box[0]),
                max(current_box[1]+current_box[3], box[1]+box[3]) - min(current_box[1], box[1])
            )
        else:
            lines.append(current_box)
            current_box = box
    lines.append(current_box)
    
    cropped_lines = [image.crop((x, y, x + w, y + h)) for (x, y, w, h) in lines]
    return cropped_lines

# Preprocess individual line
def preprocess_line(line_image, target_size=(384, 32)):
    line_image = line_image.resize(target_size, Image.ANTIALIAS)
    return line_image

# Predict text from each line image
def predict_text(line_image, processor, model, device):
    if line_image.mode != 'RGB':
        line_image = line_image.convert('RGB')
    
    pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# OCR function that processes the uploaded image
def ocr_paragraph(image, processor, model, device):
    image = preprocess_image(image)
    lines = segment_lines_opencv(image, min_line_height=10)
    
    if not lines:
        return ""
    
    preprocessed_lines = [preprocess_line(line) for line in lines]
    predicted_texts = []
    
    for line in preprocessed_lines:
        predicted_text = predict_text(line, processor, model, device)
        predicted_texts.append(predicted_text)
    
    paragraph = "\n".join(predicted_texts)
    return paragraph

# Streamlit app
import streamlit as st

# Custom styled title with a background color and width
st.markdown("""
    <style>
    body {
        background-color: black;
    }
    .title-style {
        font-size: 48px;
        color: white;
        background-color: #D2B48C;  
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        width: 100%;
        margin: 0 auto;
    }
    </style>
    <div class="title-style">
        Handwriting OCR App
    </div>
    """, unsafe_allow_html=True)

st.markdown("Upload an image of handwritten text and convert it to text along with voice translation")

# Load the model and processor only once
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return processor, model, device

# Load the translation model only once
@st.cache_resource
# Function to unzip and load the translation model
def load_translation_model(zip_path="tf_model.zip", model_folder="tf_model"):
    if not os.path.exists(model_folder):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
    # Use the tf.device context to ensure model loads on CPU
    with tf.device('/CPU:0'):
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        translation_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_folder)
    return tokenizer, translation_model

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model.generate(inputs['input_ids'], max_length=40, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

processor, model, device = load_model()
tokenizer, translation_model = load_translation_model()

# Upload image using Streamlit
uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# Text-to-speech function
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
def speak_hindi(text):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Convert text to speech and store in memory buffer
    tts = gTTS(text=text, lang='hi')
    with io.BytesIO() as audio_buffer:
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Load and play the audio
        pygame.mixer.music.load(audio_buffer, 'mp3')
        pygame.mixer.music.play()

        # Wait for the sound to finish playing
        while pygame.mixer.music.get_busy():
            continue

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Load the image
    image = Image.open(uploaded_image)
    
    # Check if OCR result is already stored in session_state
    if 'ocr_result' not in st.session_state:
        # Run OCR (assuming the 'ocr_paragraph' function exists)
        with st.spinner("Processing..."):
            recognized_paragraph = ocr_paragraph(image, processor, model, device)
            st.session_state['ocr_result'] = recognized_paragraph  # Store in session state
    else:
        recognized_paragraph = st.session_state['ocr_result']
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recognized Text (English):")
        st.text(recognized_paragraph)
    
    with col2:
        # Button to trigger text-to-speech
        if st.button('🔊'):
            st.info("Reading aloud in english...")
            speak_text(recognized_paragraph)
        
    translated_paragraph = None
    
    if 'translated_paragraph' not in st.session_state:
        st.session_state['translated_paragraph'] = ""

    # Button to translate to Hindi
    if st.button('Translate to Hindi'):
        if 'translated_paragraph' not in st.session_state or not st.session_state['translated_paragraph']:
            with st.spinner("Translating..."):
                translated_paragraph = translate_text(recognized_paragraph, tokenizer, translation_model)
                st.session_state['translated_paragraph'] = translated_paragraph  # Store translation in session state
        else:
            translated_paragraph = st.session_state['translated_paragraph']
        
        st.subheader("Translated Text (Hindi):")
        st.text(translated_paragraph)
        speak_hindi(translated_paragraph)