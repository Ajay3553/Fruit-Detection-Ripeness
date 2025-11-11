# Two-Layer Fruit Ripeness Detection System

A complete machine learning system for detecting fruit type and ripeness level using RGB color values.

## System Overview

This project implements a **two-layer detection pipeline**:

1. **Layer 1: Fruit Type Classification** - Identifies which fruit (Apple, Banana, Mango, Orange, Strawberry) is being detected
2. **Layer 2: Ripeness Classification** - Determines ripeness stage (Early Ripe, Partially Ripe, Ripe, Decay) specific to the detected fruit

## Architecture

```
RGB Color Input (R, G, B)
         ↓
    ┌────────────────────────────┐
    │  LAYER 1: Fruit Detector   │
    │  (fruit_classifier.py)     │
    │  Neural Network (3-32-16-5)│
    └────────────────────────────┘
         ↓ (Fruit Type)
         ↓ (e.g., Banana)
    ┌────────────────────────────┐
    │ LAYER 2: Ripeness Detector │
    │(train_ripeness_per_fruit.py)│
    │ 4 Fruit-Specific Models    │
    │ Each: (3-16-8-4 neurons)   │
    └────────────────────────────┘
         ↓ (Ripeness Code)
         ↓ (0-3)
    ┌────────────────────────────┐
    │ OUTPUT: Ripeness Level     │
    │ 0 = Early Ripe             │
    │ 1 = Partially Ripe         │
    │ 2 = Ripe                   │
    │ 3 = Decay                  │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │   Pin 2: Green  (0)        │
    │   Pin 3: Yellow (1)        │
    │   Pin 4: White  (2)        │
    │   Pin 5: Red    (3)        │
    └────────────────────────────┘
```

## Project Files

### Core Training Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `fruit_classifier.py` | Train fruit type detector | CSV (R,G,B,fruit_type) | `fruit_classifier_model.joblib` |
| `train_ripeness_per_fruit.py` | Train ripeness models per fruit | CSV (R,G,B,fruit_type,ripeness) | `ripeness_models/` (5 models) |
| `main_ripeness_detector.py` | End-to-end inference | RGB values | Fruit + Ripeness prediction |

### Data Files

| File | Description |
|------|-------------|
| `fruit_ripeness_dataset.csv` | Synthetic dataset (1000 samples, 4 fruits, 4 ripeness stages) |
| `demo_rgb_data.csv` | Small demo dataset for testing |

## Getting Started

### 1. Installation

```bash
# Create virtual environment
python -m venv proj_sklearn-env

# Activate environment
proj_sklearn-env\Scripts\activate.bat  # Windows
# or
source proj_sklearn-env/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy pandas scikit-learn joblib matplotlib pillow opencv-python tensorflow
```

### 2. Training

```bash
# Train Layer 1: Fruit Classifier
python fruit_classifier.py --csv fruit_ripeness_dataset.csv --epochs 200

# Train Layer 2: Ripeness Models
python train_ripeness_per_fruit.py --csv fruit_ripeness_dataset.csv --epochs 200
```

### 3. Testing

```bash
# Single prediction
python main_ripeness_detector.py --rgb-input 150 120 85

# Batch testing
python main_ripeness_detector.py --csv-input fruit_ripeness_dataset.csv

# Interactive mode
python main_ripeness_detector.py
```

## Dataset Format

### Training CSV Structure

```csv
R,G,B,fruit_type,ripeness_label
120,80,60,0,0
200,140,90,2,2
150,120,85,1,2
...
```

**Columns:**
- `R, G, B`: RGB color values (0-255)
- `fruit_type`: 0=Apple, 1=Banana, 2=Mango, 3=Orange, 4=Strawberry
- `ripeness_label`: 0=Early Ripe, 1=Partially Ripe, 2=Ripe, 3=Decay

## Label Mappings

### Fruit Types
```
0 = Apple
1 = Banana
2 = Mango
3 = Orange
4 = Strawberry
```

### Ripeness Stages
```
0 = Early Ripe      → Green
1 = Partially Ripe  → Yellow 
2 = Ripe           → White 
3 = Decay          → Red 
```

## Model Architecture

### Layer 1: Fruit Classifier
- **Input**: 3 neurons (R, G, B)
- **Hidden Layer 1**: 32 neurons (ReLU activation)
- **Hidden Layer 2**: 16 neurons (ReLU activation)
- **Output**: 4 neurons (softmax - one per fruit type)
- **Activation**: ReLU for hidden layers
- **Optimizer**: Adam
- **Loss**: Categorical Cross-entropy

### Layer 2: Ripeness Models (×5, one per fruit)
- **Input**: 3 neurons (R, G, B)
- **Hidden Layer 1**: 16 neurons (ReLU activation)
- **Hidden Layer 2**: 8 neurons (ReLU activation)
- **Output**: 4 neurons (softmax - one per ripeness stage)
- **Activation**: ReLU for hidden layers
- **Optimizer**: Adam
- **Loss**: Categorical Cross-entropy

## Expected Performance

On Synthetic Dataset:
- **Fruit Type Detection**: 95-99% accuracy
- **Ripeness Prediction**: 90-95% accuracy

## Key Features

✅ **Two-Layer Architecture**: Fruit detection followed by ripeness classification
✅ **Multi-Fruit Support**: Handles 5 different fruit types with specific color models
✅ **Lightweight**: Suitable for Raspberry Pi deployment
✅ **Flexible Inference**: Single input, batch processing, or interactive modes
✅ **Comprehensive Documentation**: Setup guides and quick references
✅ **Reproducible**: Fixed random seeds for consistent results
✅ **Validated**: Per-class precision/recall/F1 metrics and confusion matrices

## Usage Examples

### Example 1: Detect Apple at Peak Ripeness
```bash
python main_ripeness_detector.py --rgb-input 200 80 50
```
Output: Apple, Ripe (Code: 2)

### Example 2: Detect Banana in Early Stage
```bash
python main_ripeness_detector.py --rgb-input 180 150 70
```
Output: Banana, Partially Ripe (Code: 1)

### Example 3: Batch Accuracy Testing
```bash
python main_ripeness_detector.py --csv-input fruit_ripeness_dataset.csv
```
Output: Overall accuracy metrics for both layers

## Customization

### Adding New Fruits

1. Create training data with new fruit type ID (4+)
2. Update `FRUIT_MAP` in all three scripts
3. Retrain both layers with new data

### Adjusting Model Complexity

Edit hidden layer sizes in `build_model()` functions:
- Smaller networks: Faster inference, less accurate
- Larger networks: Slower inference, more accurate

### Changing Ripeness Stages

Update `RIPENESS_MAP` and redefine ripeness labels (currently 0-3, can be expanded to 0-5)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `pip install numpy pandas scikit-learn joblib` |
| "FileNotFoundError: CSV" | Ensure CSV file is in same directory as scripts |
| "Model not found" | Run training scripts first |
| "Poor accuracy" | Collect more real-world training data |

## Next Steps

1. ✅ Install and run on your system
2. ✅ Train models with provided synthetic dataset
3. ✅ Collect real sensor data
4. ✅ Retrain with real data for production accuracy
6. ✅ Monitor and log predictions for continuous improvement

**Version**: 2.0 (Two-layer detection system)
