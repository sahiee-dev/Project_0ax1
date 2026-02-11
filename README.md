# Intelligent Weapon Detection System

This project is an advanced system for the autonomous detection of firearms and knives using YOLOv8, now featuring a user-friendly Web UI and Telugu localization.

## Features

- **YOLOv8 Detection**: Efficient real-time object detection for Guns and Knives.
- **Web Interface**: Built with [Streamlit](https://streamlit.io/) for easy interaction.
- **Bilingual Support**: Toggle between **English** and **Telugu** interface.
- **Detection Summary**: Automatic counting of weapons and detection status.
- **Image & Video Support**: Upload and process both images and videos.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8.git
   cd Weapons-and-Knives-Detector-with-YOLOv8
   ```

2. **Install Dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run ui/app.py
   ```

2. **Using the App:**
   - Open the URL provided in the terminal (usually `http://localhost:8501`).
   - Use the **Sidebar** to switch languages (English/Telugu).
   - Select **Image** or **Video** from the dropdown.
   - Upload your file to see detection results and summaries.

## Directory Structure

- `model/`: YOLOv8 model loading and inference logic.
- `utils/`: Helper functions for processing results and localization.
- `ui/`: Streamlit application code.
- `runs/`: Pre-trained YOLOv8 weights.

## License

This project is distributed under the [MIT] license.
