# 🎨 VISUALi

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg)](https://www.tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated visual analytics platform that leverages deep learning and unsupervised ML techniques to extract actionable insights from advertising creative assets at scale. Built for data scientists and marketing analysts who need to understand visual patterns, optimize creative strategies, and quantify the impact of design decisions on campaign performance.

![Creative Intelligence Demo](https://img.shields.io/badge/Demo-Live-brightgreen)

## 🚀 Key Features

### Advanced Computer Vision Pipeline
- **Deep Feature Extraction**: Utilizes pre-trained CNNs (VGG16, ResNet50, EfficientNet) for robust visual feature extraction
- **Dimensionality Reduction**: UMAP/t-SNE/PCA for high-dimensional feature visualization
- **Unsupervised Clustering**: K-means with automatic elbow detection for creative segmentation
- **Visual Similarity Mapping**: Interactive 2D embeddings with image previews on hover

### Comprehensive Visual Analytics
- **Color Intelligence**: Dominant color extraction using optimized K-means clustering
- **Complexity Metrics**: Edge detection-based complexity scoring using Canny filters
- **Brightness/Contrast Analysis**: HSV color space analysis for perceptual metrics
- **Saturation & Colorfulness**: Statistical measures of color vibrancy

### Production-Ready Architecture
- **Intelligent Caching**: Multi-level caching strategy for expensive computations
- **Batch Processing**: Optimized batch inference for TensorFlow models
- **Parallel Downloads**: Concurrent image fetching with configurable workers
- **Memory Management**: Streaming processing for large datasets

## 🔬 Use Cases

### 1. **Creative Performance Optimization**
Identify which visual characteristics correlate with high CTR/conversion rates by clustering similar creatives and analyzing their performance metrics.

```python
# Example: Find top-performing visual clusters
clusters = model.cluster_images(features, n_clusters=5)
performance_by_cluster = df.groupby(clusters)['CTR'].mean()
```

### 2. **Competitive Visual Analysis**
Map your brand's creative strategy against competitors to identify whitespace opportunities and visual differentiation points.

### 3. **Creative Fatigue Detection**
Track visual diversity over time to prevent audience fatigue and maintain engagement through varied creative approaches.

### 4. **Automated Creative Tagging**
Generate visual tags and categories for large creative libraries using unsupervised learning, enabling better asset management.

### 5. **Design System Validation**
Ensure brand consistency by measuring visual similarity across campaigns and flagging outliers that deviate from brand guidelines.

## 🛠️ Tech Stack

### Core Framework
- **Streamlit**: Chosen for rapid prototyping and interactive data apps with minimal frontend overhead
- **Python 3.9+**: Modern Python features for type hints and improved performance

### Deep Learning & ML
- **TensorFlow 2.15**: Industry-standard for production deep learning
- **Scikit-learn**: Robust implementations of clustering and dimensionality reduction
- **UMAP**: State-of-the-art manifold learning for visual similarity mapping

### Computer Vision
- **OpenCV**: Optimized C++ implementations for image processing operations
- **Pillow**: Pure Python imaging for compatibility and ease of deployment

### Data Visualization
- **Plotly**: Interactive, publication-quality visualizations with WebGL rendering
- **Matplotlib/Seaborn**: Statistical visualizations for reports

### Why This Stack?
1. **Production-Ready**: All libraries are mature with extensive community support
2. **Scalable**: Designed to handle millions of images through batching and caching
3. **Interpretable**: Focus on explainable features rather than black-box models
4. **Deployable**: Streamlit Cloud compatible with minimal configuration

## 📊 Data Requirements

### Input Format
CSV file with the following structure:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Link to Creative` | URL | ✅ | Direct URL to creative image |
| `Advertiser` | String | ❌ | Company/advertiser name |
| `Brand Root` | String | ❌ | Main brand identifier |
| `Publisher` | String | ❌ | Publishing platform |
| `CTR` | Float | ❌ | Click-through rate (0-1) |
| `Impressions` | Integer | ❌ | Number of impressions |
| `Date` | DateTime | ❌ | Campaign date |

### Supported Image Formats
- JPEG/JPG, PNG, WebP, GIF (first frame)
- Recommended: 800x800px minimum for feature extraction
- Automatic resizing to model input dimensions (224x224)

## 🏗️ Architecture & Modules

### Core Modules

#### `feature_extraction.py`
- **Purpose**: Deep learning feature extraction pipeline
- **Key Functions**:
  - `load_feature_extractor()`: Model initialization with caching
  - `extract_features_batch()`: Optimized batch processing
  - `reduce_dimensions()`: UMAP/PCA/t-SNE implementations

#### `analysis.py`
- **Purpose**: Statistical analysis and ML algorithms
- **Key Functions**:
  - `analyze_creatives()`: Main analysis orchestrator
  - `cluster_images()`: K-means with silhouette scoring
  - `calculate_diversity_score()`: Visual diversity metrics

#### `visualization.py`
- **Purpose**: Interactive data visualization components
- **Key Functions**:
  - `create_visual_similarity_map()`: 2D embedding visualization
  - `create_metrics_dashboard()`: Multi-panel statistical views
  - `create_performance_heatmap()`: Performance correlation matrices

#### `utils.py`
- **Purpose**: Image processing and helper functions
- **Key Functions**:
  - `download_images_batch()`: Parallel image fetching
  - `extract_color_palette()`: Dominant color extraction
  - `calculate_visual_metrics()`: Low-level feature computation

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.9 or higher
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/creative-intelligence-dashboard.git
cd creative-intelligence-dashboard

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Standard execution
streamlit run app.py

# With custom configuration
streamlit run app.py --server.maxUploadSize 200 --server.enableCORS false

# Lightweight version (faster, reduced features)
streamlit run app_light.py
```

### Configuration
Edit `config.py` to customize:
```python
# Model selection
MODEL_CONFIGS = {
    'vgg16': {...},      # Fastest, good for general features
    'resnet50': {...},   # Balanced performance
    'efficientnet': {...} # Best accuracy, slower
}

# Processing parameters
DEFAULT_BATCH_SIZE = 32  # Adjust based on GPU memory
MAX_IMAGES_TO_PROCESS = 1000
CACHE_TTL = 3600  # 1 hour
```

## 📈 Performance Considerations

### Optimization Strategies
1. **Feature Caching**: Extracted features are cached to disk/memory
2. **Batch Processing**: Configurable batch sizes for GPU utilization
3. **Lazy Loading**: Images downloaded only when needed
4. **Progressive Analysis**: Start with subset, scale up as needed

### Benchmarks
- 1,000 images: ~2-3 minutes (CPU)
- 10,000 images: ~15-20 minutes (GPU recommended)
- Memory usage: ~4GB for 1,000 images

## 🔧 Advanced Usage

### Custom Feature Extractors
```python
@st.cache_resource
def load_custom_model():
    # Add your custom model here
    model = tf.keras.applications.YourModel(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    return model
```

### Extending Metrics
```python
def calculate_custom_metric(image: np.ndarray) -> float:
    # Add your metric computation
    return metric_value
```

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
```


---
