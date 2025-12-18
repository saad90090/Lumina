# ✨ Lumina: Web-Based Digital Image Processing Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-green?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Lumina** is a comprehensive, web-based Image Processing Studio built to demonstrate the practical application of Digital Image Processing (DIP) algorithms. It features a modern, glassmorphism UI that allows users to perform real-time image analysis, enhancement, and segmentation using a client-server architecture.

---

## 📸 Screenshots

| **Main Dashboard** | **Live Histogram Analysis** |
|:---:|:---:|
| ![Dashboard UI](https://via.placeholder.com/600x300?text=Upload+Your+Dashboard+Screenshot+Here) | ![Histogram](https://via.placeholder.com/600x300?text=Upload+Histogram+Screenshot+Here) |
| *Real-time adjustments and preview* | *RGB channel distribution visualization* |

> *Tip: Upload your own screenshots to a `screenshots/` folder in your repo and update the links above.*

---

## 🚀 Key Features

### 1. 💡 Intensity Transformations & Enhancement
- **Auto Levels (Linear Stretching):** Automatically expands the dynamic range of the image to fix low-contrast "foggy" images.
- **Gamma Correction:** Implements Power-Law transformations ($s = cr^\gamma$) for non-linear brightness adjustment (Shadow/Highlight recovery).
- **Shadows & Highlights:** Uses non-linear masking to specifically target and boost dark areas or suppress blown-out highlights.
- **Contrast & Brightness:** Standard linear bias and gain adjustments with pivot control.

### 2. 📐 Geometric Operations
- **Scaling & Interpolation:** Resizes images using Nearest Neighbor (Aliasing demo) or Bicubic Interpolation (Smooth).
- **Rotation & Flipping:** Supports 90° steps and horizontal/vertical flipping.

### 3. 🎨 Spatial Domain Filtering
- **Gaussian Blur:** Low-pass filtering for noise reduction.
- **Laplacian Sharpening:** High-pass filtering for edge enhancement.
- **Canny Edge Detection:** Advanced multi-stage algorithm for structural analysis.
- **CLAHE:** Contrast Limited Adaptive Histogram Equalization for local contrast enhancement.

### 4. 🔲 Segmentation & Morphology
- **Thresholding:** Includes Binary, **Otsu’s Binarization** (Automatic), and Adaptive Thresholding.
- **Morphological Ops:** Erosion, Dilation, Opening, Closing, and Gradient for shape analysis and noise removal.

### 5. 📊 Analysis & Performance
- **Live RGB Histogram:** Real-time visualization of pixel intensity distribution using Chart.js.
- **Optimized Preview:** Uses a downscaled proxy image for lag-free slider adjustments, while preserving the full-resolution image for the final download.

---

## 🛠️ Tech Stack

* **Backend:** Python 3, Flask
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript (AJAX Fetch API)
* **Visualization:** Chart.js (Histograms), Phosphor Icons (UI Elements)

---

## 💻 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Lumina-DIP-Suite.git](https://github.com/YOUR_USERNAME/Lumina-DIP-Suite.git)
    cd Lumina-DIP-Suite
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```

5.  **Access the App**
    Open your browser and navigate to:
    `http://127.0.0.1:5000`

---

## 📂 Project Structure

```text
/Lumina-DIP-Suite
│
├── app.py                 # Main application logic (Flask Server)
├── requirements.txt       # Python dependencies
├── README.md              # Project Documentation
│
├── static/
│   ├── uploads/           # Temporary storage for uploaded images
│   └── results/           # Temporary storage for processed downloads
│
└── templates/
    └── index.html         # Frontend Interface (HTML/JS/CSS)
