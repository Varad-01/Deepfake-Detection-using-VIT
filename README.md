# **Deepfake Detection System** 🎥  

## **Project Description**  
This is a **Streamlit-based web application** that utilizes a **Vision Transformer (ViT) model** to classify videos as real or deepfake. The application processes videos by converting them into frames using **OpenCV**, then applies the trained ViT model to classify each frame. The final classification is based on the aggregated results of all frames.  

---

## **Installation & Setup**  

### **1. Clone the Repository (if applicable)**  
```bash
git clone <repo-link>
cd <repo-folder>
```
### **2. Create a Virtual Environment**  
```bash
python -m venv env
```

### **3. Activate the Virtual Environment**  
```bash
env\Scripts\activate
```

### **4. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **5. Run the Application**  
```bash
streamlit run app.py
```

---

## **Usage**  

1. **Upload a Video** 🎥  
   - Open the Streamlit app and upload a video file.  

2. **Frame Extraction** 🖼️  
   - The uploaded video is processed and converted into individual frames using OpenCV.  

3. **Deepfake Classification** 🤖  
   - Each extracted frame is analyzed using the trained **Vision Transformer (ViT) model** to determine if it is real or fake.  

4. **Final Classification** ✅  
   - The results from all frames are aggregated to classify the video as **Real** or **Deepfake** based on majority voting.  
