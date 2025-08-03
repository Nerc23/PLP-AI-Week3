import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_drawable_canvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔢 MNIST Digit Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
### Interactive Machine Learning Demo
This web application uses a **Random Forest Classifier** trained on the MNIST dataset to classify handwritten digits (0-9).
Draw a digit or upload an image to get real-time predictions!

*Note: Using scikit-learn for Python 3.13 compatibility*
""")

@st.cache_resource
def load_model():
    """Load or train the MNIST model"""
    model_path = 'mnist_sklearn_model.joblib'
    
    try:
        # Try to load existing model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("✅ Pre-trained model loaded successfully!")
            return model
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")
    
    # Train new model
    st.info("🔄 Training new model... This may take a moment.")
    
    with st.spinner('Loading MNIST dataset...'):
        # Load MNIST dataset from sklearn
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Use subset for faster training in demo
        X_subset = X[:10000]
        y_subset = y[:10000]
        
        # Normalize data
        X_subset = X_subset / 255.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )
    
    with st.spinner('Training Random Forest model...'):
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=20
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        st.success(f"✅ Model trained! Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        # Save model
        joblib.dump(model, model_path)
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Invert colors (MNIST has white digits on black background)
    image = 1.0 - image
    
    # Flatten for sklearn model
    image = image.reshape(1, -1)
    
    return image

def predict_digit(model, image):
    """Make prediction on preprocessed image"""
    # Get prediction and probabilities
    prediction = model.predict(image)[0]
    probabilities = model.predict_proba(image)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence, probabilities

def create_confidence_chart(probabilities):
    """Create a bar chart of prediction confidence"""
    fig, ax = plt.subplots(figsize=(10, 6))
    digits = range(10)
    bars = ax.bar(digits, probabilities, color='lightblue', alpha=0.7)
    
    # Highlight the highest probability
    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(1.0)
    
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence for Each Digit')
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (digit, prob) in enumerate(zip(digits, probabilities)):
        ax.text(digit, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Sidebar for model information
st.sidebar.header("📊 Model Information")
st.sidebar.markdown("""
**Algorithm**: Random Forest Classifier
- **Trees**: 100 estimators
- **Max Depth**: 20
- **Features**: 784 (28×28 pixels)
- **Classes**: 10 digits (0-9)
- **Framework**: Scikit-learn
""")

st.sidebar.header("🔧 Technical Details")
st.sidebar.markdown("""
**Why Random Forest?**
- ✅ Works with Python 3.13
- ✅ Fast training and inference
- ✅ No GPU requirements
- ✅ Robust to overfitting
- ✅ Good baseline performance

**TensorFlow Alternative**
TensorFlow doesn't support Python 3.13 yet, so we use scikit-learn for compatibility.
""")

# Load model
model = load_model()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎨 Draw a Digit")
    
    # Drawing canvas
    canvas_result = st_drawable_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="rgba(255, 255, 255, 1)",
        background_color="rgba(0, 0, 0, 1)",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("🔍 Predict Drawn Digit", type="primary"):
        if canvas_result.image_data is not None:
            # Convert canvas to image
            input_image = canvas_result.image_data.astype(np.uint8)
            
            # Preprocess and predict
            processed_image = preprocess_image(input_image)
            digit, confidence, probabilities = predict_digit(model, processed_image)
            
            # Display results
            st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction: **{digit}**")
            st.markdown(f"### 📈 Confidence: **{confidence:.2%}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show preprocessed image
            st.subheader("Preprocessed Image (28×28)")
            processed_display = processed_image.reshape(28, 28)
            st.image(processed_display, width=150, caption="Model Input")

with col2:
    st.header("📁 Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a handwritten digit"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=280)
        
        if st.button("🔍 Predict Uploaded Image", type="primary"):
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Preprocess and predict
            processed_image = preprocess_image(image_array)
            digit, confidence, probabilities = predict_digit(model, processed_image)
            
            # Display results
            st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction: **{digit}**")
            st.markdown(f"### 📈 Confidence: **{confidence:.2%}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show preprocessed image
            st.subheader("Preprocessed Image (28×28)")
            processed_display = processed_image.reshape(28, 28)
            st.image(processed_display, width=150, caption="Model Input")

# Display confidence chart if we have predictions
if 'probabilities' in locals():
    st.header("📊 Detailed Prediction Analysis")
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        # Create and display confidence chart
        fig = create_confidence_chart(probabilities)
        st.pyplot(fig)
    
    with col4:
        st.subheader("Probability Breakdown")
        for i, prob in enumerate(probabilities):
            st.write(f"**Digit {i}**: {prob:.4f}")

# Model performance section
st.header("🎯 Model Performance")
col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Algorithm", "Random Forest", "🌳")

with col6:
    st.metric("Expected Accuracy", "~92-95%", "📈")

with col7:
    st.metric("Training Speed", "Fast", "⚡")

# Comparison with deep learning
st.header("🤖 Model Comparison")
comparison_data = {
    "Metric": ["Accuracy", "Training Time", "Inference Speed", "Memory Usage", "Python 3.13 Support"],
    "Random Forest": ["92-95%", "< 2 minutes", "< 10ms", "Low", "✅ Yes"],
    "CNN (TensorFlow)": ["98-99%", "5-10 minutes", "< 5ms", "High", "❌ No"]
}

st.table(comparison_data)

# Instructions and tips
st.header("💡 Tips for Best Results")
st.markdown("""
1. **For Drawing**: 
   - Draw digits clearly in the center of the canvas
   - Use thick strokes for better recognition
   - Try to fill most of the canvas area

2. **For Upload**: 
   - Use images with dark digits on light background
   - Ensure the digit is centered and clearly visible
   - Avoid complex backgrounds

3. **Model Notes**:
   - This Random Forest model achieves ~92-95% accuracy
   - Slightly lower than deep learning but still very good
   - Much faster training and compatible with Python 3.13
   - Works well for most handwritten digits
""")

# Sample predictions section
st.header("🧪 Try These Test Cases")
st.markdown("Here are some digit examples you can try drawing:")

# Create sample digits display
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    # Create simple digit patterns for demonstration
    sample_digit = np.zeros((28, 28))
    # Add some sample patterns (simplified representations)
    if i == 0:  # Circle for 0
        cv2.circle(sample_digit, (14, 14), 10, 1, 2)
    elif i == 1:  # Vertical line for 1
        sample_digit[5:23, 12:16] = 1
    elif i == 2:  # Curved line for 2
        sample_digit[5:10, 8:20] = 1
        sample_digit[10:15, 15:20] = 1
        sample_digit[15:20, 8:13] = 1
        sample_digit[20:23, 8:20] = 1
    # ... (simplified patterns for other digits)
    
    axes[i].imshow(sample_digit, cmap='gray')
    axes[i].set_title(f'Digit {i}')
    axes[i].axis('off')

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Built with Streamlit and Scikit-learn | 🎓 PLP AI Week 3 Assignment</p>
    <p>Model: Random Forest Classifier trained on MNIST dataset</p>
    <p>Compatible with Python 3.13 ✅</p>
</div>
""", unsafe_allow_html=True)
