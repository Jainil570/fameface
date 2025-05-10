import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('celebrity_cnn_model.h5')

# Sorted class names
labels = sorted([
    "Aamir Khan", "Abhay Deol", "Abhishek Bachchan", "Aishwarya Rai", "Ajay Devgn", "Akshay Kumar", "Akshaye Khanna", "Alia Bhatt",
    "Amitabh Bachchan", "Amy Jackson", "Angelina Jolie", "Anil Kapoor", "Anushka Sharma", "Arjun Kapoor", "Arjun Rampal", "Arshad Warsi",
    "Ayushmann Khurrana", "Benedict Cumberbatch", "Bhumi Padnekar", "bill_gates", "Bipashu Basu", "Bobby Deol", "Brad Pitt", "Brie Larson",
    "Chris Evans", "Chris Hemsworth", "Deepika Padukone", "Denzel Washington", "Disha Patani", "Elizabeth Olsen", "Emraan Hashmi", "Hrithik Roshan",
    "Hugh Jackman", "Ileana D'Cruz", "Irrfan Khan", "Jacqueline Fernandez", "Jennifer Lawrence", "John Abraham", "Johnny Depp", "Josh Brolin",
    "Juhi Chawla", "Kane Williamson", "Kangana Ranaut", "Kareena Kapoor", "Karen Gillan", "Karisma Kapoor", "Kartik Aaryan", "Kate Winslet",
    "Katrina Kaif", "Kiara Advani", "Kobe Bryant", "Kriti Sanon", "Leonardo DiCaprio", "lionel_messi", "Madhuri Dixit", "Manoj Bajpayee",
    "Maria Sharapova", "Mark Ruffalo", "Megan Fox", "mithali_raj", "Mrunal Thakur", "ms_dhoni", "Nana Patekar", "Narendra Modi",
    "Natalie Portman", "Nicole Kidman", "Nora Fatehi", "Paresh Rawal", "Parineeti Chopra", "Paul Rudd", "Pooja Hegde", "Prabhas",
    "Prachi Desai", "Priyanka Chopra", "Rajkummar Rao", "Ranbir Kapoor", "Randeep Hooda", "Ranveer Singh", "Rashmika Mandanna",
    "Riteish Deshmukh", "Robert Downey Jr", "roger_federer", "ronaldo", "sachin_tendulkar", "Saif Ali Khan", "Salman Khan",
    "Sandra Bullock", "Sanjay Dutt", "Sara Ali Khan", "Scarlett Johansson", "serena_williams", "Shah Rukh Khan", "Shahid Kapoor",
    "Shilpa Shetty", "Shraddha Kapoor", "Shruti Haasan", "Sidharth Malhotra", "Sonakshi Sinha", "Sonam Kapoor", "steve_jobs",
    "Suniel Shetty", "Sunny Deol", "Sushant Singh Rajput", "Tom Cruise", "Tom Hanks", "Tom Holland", "Vin Diesel", "virat_kohli",
    "Will Smith", "Yami Gautam"
])

# App UI
st.set_page_config(page_title="celebrity recognizer", layout="centered")
st.title("✨ Who is that star?")
st.subheader("Who's in the frame? Upload a photo and find out!")

uploaded_file = st.file_uploader("📤 Upload an image of a celebrity", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='📸 Uploaded Image', use_column_width=False)

    if st.button("🔍 Identify Celebrity"):
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"🎯 Match Found: **{labels[class_index]}**")
        st.write(f"📊 Confidence: {confidence:.2%}")

        if confidence < 0.7:
            st.info("🤔 Not so sure? Try uploading a clearer image.")

else:
    st.info("👆 Upload a clear frontal face image to begin.")

st.markdown("---")