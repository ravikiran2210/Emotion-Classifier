# Image Emotion Classifier

A web application that classifies images as either "Happy" or "Sad" using a Convolutional Neural Network (CNN) built with TensorFlow.

## Features

- Web interface for easy image upload and prediction
- Real-time emotion classification
- Confidence score display
- Modern and responsive design
- Support for various image formats (jpg, jpeg, png, bmp)

## Requirements

- Python 3.7+
- TensorFlow 2.8.0+
- OpenCV
- Flask
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ravikiran-10/ImageEmotionClassifier.git
cd ImageEmotionClassifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image and click "Predict Emotion" to get the classification result.

## Project Structure

```
ImageEmotionClassifier/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── LICENSE            # MIT License
├── .gitignore         # Git ignore rules
├── models/            # Directory for saved models
├── data/              # Directory for training data
│   ├── happy/        # Happy emotion images
│   └── sad/          # Sad emotion images
└── templates/
    └── index.html    # Web interface template
```

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- Max Pooling layers for dimensionality reduction
- Dense layers for classification
- Sigmoid activation for binary classification

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow for the deep learning framework
- Flask for the web framework
- OpenCV for image processing
