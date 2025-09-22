# pallet-detection
Real-time pallet detection and counting using YOLOv8
# 📦 Pallet Detection and Counting using Computer Vision

## 📌 Business Problem
Manual tracking of pallets in warehouses is **time-consuming** and often **prone to errors**.  
This leads to:
- Inefficient inventory management due to lack of real-time data
- Higher operational costs
- Revenue loss from misplaced or unaccounted pallets  

---

## 🎯 Objectives
- **Minimize** manual counting errors  
- **Maximize** inventory tracking efficiency  
- **Minimize** pallet loss incidents  
- **Maximize** real-time visibility of warehouse inventory  

---

## ✅ Success Criteria
### Business
- ≥ 95% accuracy in detecting pallets from static images  
- Simple web interface usable by non-technical warehouse staff  

### ML
- mAP50 ≥ 0.96  
- Precision ≥ 0.98  
- Recall ≥ 0.92  
- Segmentation Mask mAP50 ≥ 0.90  

### Economic
- Reduce manual pallet counting effort by **80%**  

---

## ⚙️ Tech Stack
- **Programming Language**: Python  
- **Libraries**: Ultralytics YOLOv8, OpenCV, NumPy, Pandas, Matplotlib  
- **Annotation Tool**: Roboflow (Polygon instance segmentation)  
- **Deployment**: Streamlit (lightweight web app)  
- **Training Environment**: Google Colab / Local GPU  

---

## 📂 Project Structure
