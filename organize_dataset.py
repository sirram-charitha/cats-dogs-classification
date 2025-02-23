import os
import shutil
import random

# Define paths
original_train_path = "dataset/train/"  # Your original train folder
output_train_path = "dataset/train_organized/"
output_test_path = "dataset/test_organized/"

# Create organized folders
for category in ["Cat", "Dog"]:
    os.makedirs(output_train_path + category, exist_ok=True)
    os.makedirs(output_test_path + category, exist_ok=True)

# Function to move images
def move_images(category, train_count=5000, test_count=1000):
    images = [img for img in os.listdir(original_train_path) if img.startswith(category.lower())]
    random.shuffle(images)  # Shuffle images for randomness

    # Move images to training set
    for img in images[:train_count]:
        shutil.move(original_train_path + img, output_train_path + category + "/" + img)

    # Move images to testing set
    for img in images[train_count:train_count + test_count]:
        shutil.move(original_train_path + img, output_test_path + category + "/" + img)

# Move Cat and Dog images
move_images("Cat")
move_images("Dog")

print("✅ Dataset Organized Successfully!")
