import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# Load Model
classifier = load_model('dogcat_model.h5')

# Load Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_organized', target_size=(64, 64), batch_size=32, class_mode='binary', shuffle=False)

# Predict
test_set.reset()
predictions = classifier.predict(test_set)

# Create DataFrame
df = pd.DataFrame({
    'filename': test_set.filenames,
    'prediction': predictions[:, 0],
    'true_label': test_set.classes
})
df['predicted_label'] = (df['prediction'] > 0.5).astype(int)

# Find Misclassified Images
misclassified = df[df['true_label'] != df['predicted_label']]
print(f"Total misclassified images: {misclassified.shape[0]}")

# Plot Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(df['true_label'], df['predicted_label'])
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
