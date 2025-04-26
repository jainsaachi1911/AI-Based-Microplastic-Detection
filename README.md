# CLEANSE - AI  
**Classification and Environmental Analysis of Microplastics for a Sustainable Ecosystem using AI**

## Project Overview

CLEANSE-AI presents an end-to-end computer vision framework designed to detect and classify microplastics in microscopic water images. The solution leverages advanced data augmentation techniques, YOLOv8 for object detection, and a Random Forest classifier to assess pollution severity. This system aims to provide scalable, interpretable, and automated environmental monitoring.

---

## Methodology

1. **Data Augmentation**  
   Simulates real-world imaging conditions using:
   - Horizontal Flip (18.5%)
   - Gaussian Noise (18.5%)
   - Motion Blur (18.5%)
   - Lighting variations

2. **Microplastic Detection using YOLOv8**  
   - Detects individual particles with bounding boxes.
   - Outputs features like size and shape.

3. **Pollution Classification using Random Forest**  
   - Extracted features: count, size metrics, aspect ratio.
   - Predicts pollution level: Low, Moderate, High, or Critical.

---

## Solution Workflow

**Phase 1**: Dataset Enhancement  
- Augmentation using OpenCV and imgaug  
- Improves model generalization to diverse real-world conditions

**Phase 2**: Detection with YOLOv8  
- High-speed detection of microplastic particles  
- Outputs used for downstream classification

**Phase 3**: Pollution Assessment  
- Feature extraction from YOLO detections  
- Random Forest model classifies pollution severity  
- Streamlit app used for visualization and explainability

---

## Results

- **Total Images**: 1728 augmented + 320 original  
- **Train / Val / Test Split**: 1728 / 219 / 217  
- **Accuracy**: 96%  
- **Precision (Class-wise)**: >94% for fiber, film, fragment  
- **Critical pollution predicted** with >90% confidence  
- **Final Metrics**:  
  - Precision: 0.96  
  - Recall: 0.90  
  - mAP@0.5: 0.97  
  - mAP@0.5:0.95: 0.65 (after 50 epochs)

---

## Technologies Used

- YOLOv10 
- OpenCV, imgaug    
- Streamlit  
- Python 3.10
