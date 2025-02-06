from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.image import img_to_array

class ImageRetrievalModel:
    def __init__(self, input_shape=(128, 128, 3), embedding_dim=64):  # ✅ Fixed init method
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.autoencoder, self.encoder = self._build_autoencoder()  # ✅ Now initializes correctly
        self.neighbor_model = None
        self.image_paths = []
        self.images = []

    def _build_autoencoder(self):
        input_img = Input(shape=self.input_shape)
        x = Flatten()(input_img)
        encoded = Dense(256, activation='relu')(x)
        encoded = Dense(128, activation='relu')(encoded)
        
        x = Dense(256, activation='relu')(encoded)
        x = Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        decoded = Reshape(self.input_shape)(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)  
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')

        return autoencoder, encoder

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).resize(self.input_shape[:2]).convert('RGB')
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                self.image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        self.images = np.array(images)

    def fit(self, dataset_path, batch_size=32, epochs=5):
        self.load_images_from_folder(dataset_path)

        self.autoencoder.fit(
            self.images, self.images,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )

        encoded_features = self.encoder.predict(self.images)
        feature_vectors = encoded_features / np.linalg.norm(encoded_features, axis=1, keepdims=True)
        self.neighbor_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine').fit(feature_vectors)

    def retrieve_similar_images(self, image_path, top_n=5):
        if self.neighbor_model is None:
            raise Exception("Model has not been fitted yet.")

        query_image = Image.open(image_path).resize(self.input_shape[:2]).convert('RGB')
        query_image_array = img_to_array(query_image) / 255.0
        query_image_array = np.expand_dims(query_image_array, axis=0)
        query_features = self.encoder.predict(query_image_array)
        query_features_normalized = query_features / np.linalg.norm(query_features)

        distances, indices = self.neighbor_model.kneighbors(query_features_normalized)
        similar_image_paths = [self.image_paths[idx] for idx in indices[0]]

        return similar_image_paths

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = ImageRetrievalModel()

# ✅ Check if model weights exist before loading
if os.path.exists('autoencoder.weights.h5') and os.path.exists('encoder.weights.h5'):
    model.autoencoder.load_weights('autoencoder.weights.h5')
    model.encoder.load_weights('encoder.weights.h5')
else:
    print("❌ Model weights not found! Train the model first.")

# Load images and fit the NearestNeighbors model
dataset_path = "C:/Users/Rajesh R/Desktop/similarity/Apparel images dataset new/black_shirt"
model.load_images_from_folder(dataset_path)

if len(model.images) > 0:
    encoded_features = model.encoder.predict(model.images)
    feature_vectors = encoded_features / np.linalg.norm(encoded_features, axis=1, keepdims=True)
    model.neighbor_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine').fit(feature_vectors)
else:
    print("❌ No images found in dataset path!")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize(model.input_shape[:2]).convert('RGB')
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        query_features = model.encoder.predict(image_array)
        query_features_normalized = query_features / np.linalg.norm(query_features)

        distances, indices = model.neighbor_model.kneighbors(query_features_normalized)
        similar_image_paths = [model.image_paths[idx] for idx in indices[0]]

        return JSONResponse(content={"similar_images": similar_image_paths})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ Fixed `if __name__ == "__main__"` syntax
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)