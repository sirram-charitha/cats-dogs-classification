import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Model
classifier = tf.keras.models.load_model('dogcat_model.h5')

# Data Preparation
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_organized', target_size=(64, 64), batch_size=32, class_mode='binary', shuffle=False)

# Evaluate Model
accuracy = classifier.evaluate(test_set)
print(f"Validation Accuracy: {accuracy[1] * 100:.2f}%")
