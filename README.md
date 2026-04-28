# 🛡️ Intelligent Weapon Detection System

An advanced, high-performance system designed for autonomous detection of firearms and knives using **YOLOv8**. This project features a modern Web UI, real-time processing capabilities, and multi-language support.

---

## 🌟 Key Features

- **🚀 Real-time Detection**: Powered by YOLOv8 for industry-leading speed and accuracy.
- **🖼️ Multi-format Support**: Process both images and video streams seamlessly.
- **🖥️ Modern Web UI**: User-friendly interface built with [Streamlit](https://streamlit.io/).
- **🌐 Bilingual Support**: Native support for **English** and **Telugu** localization.
- **📊 Detailed Analytics**: Automatic counting of detected objects with status summaries.
- **🏗️ Modular Architecture**: Clean separation of model logic, utilities, and UI.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)

### Step-by-Step Guide

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd Project_0ax1
   ```

2. **Set up Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Launch the system using the following command:

```bash
streamlit run ui/app.py
```

### Navigating the Interface:
1. **Sidebar**: Toggle between English and Telugu languages.
2. **Media Selection**: Choose between Image or Video processing from the dropdown menu.
3. **Upload**: Drag and drop your files to see instantaneous detection results.

---

## 📁 Repository Structure

- `model/`: core YOLOv8 model loading and inference logic.
- `utils/`: utility functions for processing results and localization.
- `ui/`: frontend Streamlit application components.
- `runs/`: directory for pre-trained YOLOv8 weights and training artifacts.
- `data/`: dataset configuration and sample files.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

*Developed with ❤️ for safety and security.*
