# fameface
# ✨ Celebrity Recognizer App

## Identify the Stars in Your Photos!

This web application allows you to upload an image of a celebrity and instantly identify who they are using a Convolutional Neural Network (CNN) model. Built with Streamlit for a user-friendly interface and powered by TensorFlow/Keras for the deep learning model, this app makes celebrity recognition accessible to everyone.

## 🎬 Demo Video

[![Watch the Celebrity Recognizer App Demo](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://screenrec.com/share/OY8iDJZng2)

**Click the image above to watch a short demonstration of the Celebrity Recognizer App in action!**

The video showcases the following:

* **User Interface:** A clean and intuitive interface powered by Streamlit.
* **Image Upload:** How easily a user can upload an image of a celebrity using the file uploader.
* **Celebrity Identification:** The process of clicking the "Identify Celebrity" button to trigger the prediction.
* **Real-time Prediction:** The app quickly analyzes the uploaded image using the trained CNN model.
* **Displaying Results:** Clearly presents the identified celebrity's name with a confidence score.
* **Handling Uncertainty:** Shows how the app provides a helpful message when the confidence level is low, suggesting a clearer image.
* **Variety of Celebrities:** Demonstrates the app's ability to recognize different celebrities from the trained dataset.

This demo provides a quick overview of the app's functionality and ease of use. Give it a try yourself!

## 🚀 Getting Started

1.  **Clone the repository** (if you have the code in one):
    ```bash
    git clone [your_repository_url]
    cd [your_app_directory]
    ```

2.  **Install the required libraries:**
    ```bash
    pip install streamlit tensorflow numpy Pillow
    ```

3.  **Ensure you have the model file (`celebrity_cnn_model.h5`) in the same directory as your Streamlit app script.**

4.  **Run the Streamlit app:**
    ```bash
    streamlit run your_app_script_name.py
    ```
    (Replace `your_app_script_name.py` with the actual name of your Python file.)

5.  **Open your web browser** to the address displayed in the terminal (usually `http://localhost:8501`).

## 📸 How to Use

1.  **Upload an Image:** Click on the "Browse files" button (or drag and drop an image) to upload a JPG, JPEG, or PNG image of a celebrity.
2.  **View Uploaded Image:** The app will display the image you've uploaded.
3.  **Identify Celebrity:** Click the "🔍 Identify Celebrity" button.
4.  **View Results:**
    * The app will display the predicted celebrity name with a "🎯 Match Found!" message.
    * The confidence score of the prediction will be shown as a percentage under "📊 Confidence:".
    * If the confidence is below 70%, a "🤔 Not so sure?" message will appear, suggesting a clearer image might improve accuracy.

## ⚙️ Technology Focus: Deep Learning with Convolutional Neural Networks (CNNs)

At the heart of this application lies a **Convolutional Neural Network (CNN)**, a powerful type of deep learning model particularly well-suited for image recognition tasks. Here's a breakdown of why CNNs are effective for this:

* **Hierarchical Feature Learning:** CNNs learn features in a hierarchical manner. In the early layers, they automatically learn to detect basic image elements like edges, corners, and textures. As the data flows through deeper layers, the network combines these simple features into more complex, high-level representations such as facial features (eyes, nose, mouth), and ultimately, entire faces and identities.

* **Convolutional Layers:** The core building blocks of CNNs are convolutional layers. These layers use **filters** (small matrices) that slide over the input image. During this process, the filter performs a dot product with the local region of the image, producing a **feature map**. Each filter is designed to detect a specific type of feature. By using multiple filters, the network can extract a rich set of features from the input image.

* **Pooling Layers:** Pooling layers are typically inserted after convolutional layers. Their role is to reduce the dimensionality of the feature maps, which helps to:
    * **Reduce computational cost:** Fewer parameters to process in subsequent layers.
    * **Increase robustness to variations:** Make the model more invariant to small shifts, rotations, and changes in scale of the features in the image. Common pooling operations include **max pooling** (selecting the maximum value in a local region) and **average pooling**.

* **Activation Functions:** After each convolutional and fully connected layer, an **activation function** is applied. This function introduces non-linearity into the model, allowing it to learn complex patterns. Common activation functions include ReLU (Rectified Linear Unit).

* **Fully Connected Layers:** The final layers of a CNN are usually fully connected layers, similar to those in traditional neural networks. The high-level features learned by the convolutional and pooling layers are flattened and fed into these fully connected layers. The output layer typically uses a **softmax activation function** to produce a probability distribution over the different celebrity classes. The class with the highest probability is the model's prediction.

* **Training Process:** The CNN model used in this app was trained on a large dataset of celebrity images. During training, the model learned to associate specific visual features with different celebrity identities by adjusting its internal weights and biases through a process called **backpropagation** and an optimization algorithm.

* **TensorFlow and Keras:** This app leverages the power of **TensorFlow**, an open-source machine learning framework, and **Keras**, a high-level API for building and training neural networks. TensorFlow provides the underlying computational infrastructure, while Keras simplifies the process of designing, building, and training the CNN model.

## 📚 Libraries Used

* **Streamlit:** For creating the interactive web application interface.
* **TensorFlow:** For building, loading, and running the CNN model.
* **NumPy:** For numerical operations, especially for handling image data as arrays.
* **Pillow (PIL):** For image manipulation, such as opening and resizing uploaded images.

## ⚠️ Model Accuracy and Limitations

The accuracy of this celebrity recognition app depends heavily on the quality and diversity of the training data used to build the CNN model. Factors that can affect the prediction accuracy include:

* **Image Quality:** Blurry, low-resolution, or poorly lit images may be difficult for the model to analyze.
* **Occlusion and Pose:** If the celebrity's face is partially obscured or in an unusual pose, the accuracy might decrease.
* **Similarity Between Celebrities:** Some celebrities may have similar facial features, which can lead to confusion for the model.
* **Training Data Coverage:** The model can only recognize celebrities that were included in its training dataset.
