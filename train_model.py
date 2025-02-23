from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Define the CNN model
classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/train_organized', target_size=(64,64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_organized', target_size=(64,64), batch_size=32, class_mode='binary')

# Train the model
classifier.fit(train_set, epochs=10, validation_data=test_set)

# Save the trained model
classifier.save('dogcat_model.h5')
