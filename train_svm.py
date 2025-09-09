from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

dataset_path = "D:/SCT_ML_3/dogs-vs-cats"  # Change to your dataset path
categories = ["cat", "dog"]
image_size = 64

data = []
labels = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    files = os.listdir(folder)
    for file in files[:50]:  # For demo, first 50 images per class
        file_path = os.path.join(folder, file)
        try:
            img = Image.open(file_path).convert("L")
            img = img.resize((image_size, image_size))
            arr = np.array(img)
            data.append(arr)
            labels.append(category)
        except:
            print("Failed:", file_path)

data = np.array(data)
labels = np.array(labels)

n_samples = data.shape[0]
data_flatten = data.reshape(n_samples, -1)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data_flatten, labels_encoded, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("SVM Test Accuracy:", accuracy)

# Save model and label encoder
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
print("Model and Label Encoder saved!")
