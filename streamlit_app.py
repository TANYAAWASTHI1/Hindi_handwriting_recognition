import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

# Set the Page Title
st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon="🔎"
)

# Load the Model
def load_model():
    # Replace this with your model loading code if available
    return None

hindi_character = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'ब', 'भ', 'म',
                   'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'ॠ', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ',
                   'ज', 'झ', '0', '१', '२', '३', '४', '५', '६', '७', '८', '९']

# Side Navigation
with st.sidebar:
    sel = st.selectbox("Navigation", ["Home", "Prediction", "Get test images", "Code"])

# Read and resize the image
def load_and_prep(file):
    img = Image.open(file)
    img = np.array(img)
    img = cv2.resize(img, (32, 32))
    return img

# Get top n predictions
def get_n_predictions(pred_prob, n):
    pred_prob = np.squeeze(pred_prob)
    top_n_max_idx = np.argsort(pred_prob)[::-1][:n]
    top_n_max_val = list(pred_prob[top_n_max_idx])
    top_n_class_name = [hindi_character[i] for i in top_n_max_idx]
    return top_n_class_name, top_n_max_val

# Show Animated image
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

###################################### Home Page ######################################
if sel == 'Home':
    st.write("##  Hi, Welcome to my project")
    st.title("Hindi Character Recognition")
    st.info("The project's fundamental notion is that when it comes to constructing\
             OCR models for native languages, the accuracies achieved are rather low, and so this\
             is a sector that still needs research. This model (as implemented here) can be extended\
             to recognize complete words, phrases, or even entire paragraphs.")
    st.info("Handwritten character recognition is an important area in the study of image processing\
             and pattern recognition. It is a broad field that encompasses all types of character recognition\
             by machine in a variety of application fields. The purpose of this pattern recognition area is to\
             convert human-readable characters into machine-readable characters. We now have automatic character\
             recognizers that assist people in a wide range of practical and commercial applications.")

###################################### Prediction Page ######################################
elif sel == 'Prediction':
    file = st.file_uploader(" ")  # From here we can upload an image
    if file is None:
        st.write("### Please Upload an Image that contains a Hindi Character")
    else:
        img = load_and_prep(file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("### Select the top n predictions")
        n = st.slider('n', min_value=1, max_value=5, value=3, step=1)
        if st.button("Predict"):
            pred_prob = np.random.rand(len(hindi_character))  # Random probabilities for demonstration
            class_name, confidence = get_n_predictions(pred_prob, n)
            st.header(f"Top {n} Prediction for given image")
            fig = go.Figure(data=[go.Bar(x=confidence, y=class_name, orientation='h')])
            fig.update_layout(height=500, width=900, xaxis_title='Probability', yaxis_title=f'Top {n} Class Name')
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"The image is classified as \t  \'{class_name[0]}\' \t with {confidence[0] * 100:.1f} % probability")

###################################### Download Page ######################################
elif sel == 'Get test images':
    st.write('# Download the test images')
    st.write("Replace this section with your code for downloading test images.")

###################################### Code Page ######################################
elif sel == 'Code':
    with open('app.py', 'r', encoding="utf8") as f:
        code = f.read()
    st.code(code, 'python')

