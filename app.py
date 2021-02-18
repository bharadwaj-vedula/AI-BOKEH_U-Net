import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title(" AI BOKEH")
st.markdown("**upload an image**")
uploaded_image = st.file_uploader(label="",type=['jpg','jpeg','png','jfif'],accept_multiple_files=True)

"""def prediction(image):
    unet_model=load_model("best_unet.h5")
    mask=unet_model.predict(image)[0]
    mask=np.resize(mask,(256,256))
    return mask"""

@st.cache
def Load_Model():
    unet_model=load_model("best_unet.h5")
    return unet_model


if uploaded_image is not None:
    st.image(uploaded_image, caption="uploaded image", use_column_width=True)
    image=Image.open(uploaded_image)
    image = np.array(image)
    uploaded_image=cv2.resize(image,(256,256))
    uploaded_image=uploaded_image/255
    uploaded_image=np.expand_dims(uploaded_image,axis=0)
    model=Load_Model()
    mask=model.predict(uploaded_image)[0]
    mask=np.resize(mask,(256,256))
    mask=(mask>0.1)*255
    mask=np.full((256,256),[mask],np.uint8)
    mapping = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    image=cv2.resize(image,(256,256))
    blurred_original_image = cv2.GaussianBlur(image,(25,25),0)
    layered_image = np.where(mapping != (0,0,0),image,blurred_original_image)
    st.image(mask,caption="mask",use_column_width=True)
    st.image(layered_image,caption="potrait-image",use_column_width=True)
