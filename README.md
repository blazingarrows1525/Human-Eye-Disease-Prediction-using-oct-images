# Retinal OCT Disease Detection

A machine learning web application for automated detection of retinal diseases using Optical Coherence Tomography (OCT) images. Built with Streamlit and TensorFlow.

## Features

- **AI-Powered Classification**: Detects four retinal conditions:
  - **Normal**: Healthy retina with no abnormalities
  - **CNV**: Choroidal Neovascularization (abnormal blood vessel growth)
  - **DME**: Diabetic Macular Edema (fluid accumulation in the macula)
  - **Drusen**: Age-related Macular Degeneration (Early AMD)

- **User-Friendly Interface**: Easy-to-use web interface for uploading OCT images
- **Real-time Predictions**: Instant disease classification with confidence scores
- **Medical Recommendations**: Disease-specific information and recommendations
- **Pre-trained Model**: Uses a deep learning model trained on thousands of OCT images

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.11
- **ML Framework**: TensorFlow 2.20.0 with Keras
- **Image Processing**: NumPy, Pillow

## Installation

### Requirements

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Human-Eye-Disease-Prediction.git
cd Human-Eye-Disease-Prediction/Human_Eye_Disease_Prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### How to Use

1. Navigate to the **"Disease Identification"** tab
2. Upload an OCT retinal image (JPG, PNG format)
3. Click **"Analyze Image"** button
4. View the classification result and recommendations

## Project Structure

```
├── app.py                    # Main Streamlit application
├── recommendation.py         # Disease recommendations and information
├── Trained_Model.keras      # Pre-trained deep learning model (22.8 MB)
├── Training_history.pkl     # Training metrics and history
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Model Details

- **Architecture**: MobileNetV3-based CNN
- **Input Size**: 224x224 pixels
- **Output**: 4-class classification
- **Accuracy**: Trained on thousands of OCT images

## Dataset

The model was trained on OCT images from the Retinal Image Database (RIDB). The training includes:
- Normal retinal scans
- CNV cases
- DME cases
- Drusen cases

## Limitations

⚠️ **Important**: This tool is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified ophthalmologists for medical advice.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This application provides predictions based on machine learning models and should not be considered medical advice. Always consult healthcare professionals for proper diagnosis and treatment.

## Author

Based on the Human-Eye-Disease-Prediction research project.

## Acknowledgments

- Deep learning frameworks: TensorFlow/Keras
- Web framework: Streamlit
- Image processing: NumPy and Pillow
