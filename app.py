import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os

MODEL = tf.keras.models.load_model('models/model_1.keras')
CLASS_NAMES = ['Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_healthy']


def google_map(latitude, longitude, zoom):
    """Render a Google Map component."""
    map_html = f"""
    <iframe
        width="100%"
        height="500"
        src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d{zoom}!2d{longitude}!3d{latitude}!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2zMTPCsDE3JzM2LjIiTiA3NMKwNTInMjMuMiJX!5e0!3m2!1sen!2sus!4v1617128122641!5m2!1sen!2sus"

        frameborder="0" style="border:0;"
        allowfullscreen=""
        aria-hidden="false"
        tabindex="0"
    ></iframe>
    """
    st.components.v1.html(map_html, height=600)

def text_to_speech(text):
    # Generate speech
    tts = gTTS(text=text, lang="en", slow=False)
    
    # Save speech to a file
    tts.save("output.mp3")

def preprocess_image(image):

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title('Patato Disease Classification App')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            img_batch = np.expand_dims(image, 0)
            predictions = MODEL.predict(img_batch)
            
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            print(f"My Class - {predicted_class}")
            st.write('Prediction:', predictions)


            st.write('Prediction Class:', predicted_class)


            
            if predicted_class=="Late Blight":
                files="late"
            elif predicted_class=='early':
                files="early"
            else:
                files="health"
           
     
            google_map(28.368815442586538, 77.31872731252683, 15)
            with open(f"doc/{files}.txt","r") as f:
                mytext=f.read()
                

                print(mytext)
                st.write('', mytext)
                text_to_speech(mytext)
        
                # Display audio player to play the generated speech
                audio_file = open("output.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                audio_file.close()
                # os.remove("output.mp3")  # Remove the temporary audio file
            
if __name__ == '__main__':
    main()
