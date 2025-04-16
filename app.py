import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

def setup_gpu():
    """Configure GPU memory growth to avoid OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

def clean_invalid_images(data_dir='data', image_exts=['jpeg', 'jpg', 'bmp', 'png']):
    """Remove invalid or corrupted images from the dataset"""
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))

def load_data(data_dir='data'):
    """Load and preprocess the image dataset"""
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data = data.map(lambda x,y: (x/255, y))
    return data

def create_model():
    """Create and compile the CNN model"""
    model = Sequential()
    
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

def train_model(model, data, epochs=20):
    """Train the model and return training history"""
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    
    hist = model.fit(train, epochs=epochs, validation_data=val)
    return hist, test

def evaluate_model(model, test_data):
    """Evaluate the model on test data"""
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    
    for batch in test_data.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    
    print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

def save_model(model, model_dir='models'):
    """Save the trained model"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, 'imageclassifier.h5'))

def predict_image(image_path, model_path='models/imageclassifier.h5'):
    """Predict whether an image is happy or sad"""
    # Load the model if it exists
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Model not found. Please train the model first.")
        return None, None
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None
    
    # Resize image to match model's expected sizing
    resize = tf.image.resize(img, (256, 256))
    
    # Normalize the image
    normalized = resize / 255.0
    
    # Add batch dimension
    input_image = np.expand_dims(normalized, 0)
    
    # Make prediction
    prediction = model.predict(input_image)
    
    # Convert image to base64 for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    
    # Interpret prediction
    if prediction[0] > 0.5:
        return "Happy", prediction[0][0], image_base64
    else:
        return "Sad", 1 - prediction[0][0], image_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the uploaded file temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    # Get prediction
    result, confidence, image_base64 = predict_image(temp_path)
    
    # Remove temporary file
    os.remove(temp_path)
    
    if result is None:
        return jsonify({'error': 'Error processing image'})
    
    return jsonify({
        'prediction': result,
        'confidence': f'{confidence:.2%}',
        'image': image_base64
    })

def main():
    # Check if model exists
    model_path = 'models/imageclassifier.h5'
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        
        # Setup GPU
        setup_gpu()
        
        # Clean invalid images
        print("Cleaning invalid images...")
        clean_invalid_images()
        
        # Load data
        print("Loading data...")
        data = load_data()
        
        # Create and train model
        print("Creating and training model...")
        model = create_model()
        hist, test_data = train_model(model, data)
        
        # Evaluate model
        print("Evaluating model...")
        evaluate_model(model, test_data)
        
        # Save model
        print("Saving model...")
        save_model(model)
    
    # Start Flask app
    app.run(debug=True)

if __name__ == "__main__":
    main() 