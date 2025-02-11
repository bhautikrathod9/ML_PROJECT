import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_df = pd.read_csv('/train.csv')

# Load the testing dataset
test_df = pd.read_csv('/test.csv')

# Display the first few rows of the training dataset
print(train_df.head())

# Select the first 10 images and labels
sample_images = train_df.iloc[:10, :-1].values  # All pixel values
sample_labels = train_df.iloc[:10, -1].values    # Labels

# Reshape images to 28x28
sample_images = sample_images.reshape(-1, 28, 28)

# Plot the images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i], cmap='gray')
    plt.title(f'Label: {sample_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()