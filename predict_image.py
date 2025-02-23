import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load Trained Model
classifier = tf.keras.models.load_model('dogcat_model.h5')

def predict_image(img_path):
    img1 = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img1)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model

    prediction = classifier.predict(img)

    if prediction[0, 0] > 0.5:
        label = f'Dog: {prediction[0,0]:.2f}'
    else:
        label = f'Cat: {1.0 - prediction[0,0]:.2f}'

    # Display the image with the prediction
    plt.imshow(img1)
    plt.text(20, 62, label, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

# Predict an image (change the path accordingly)
predict_image('dataset/test_organized/Cat/cat.1512.jpg')
