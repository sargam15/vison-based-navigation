# 🧭 Vision-Based Navigation for Underwater ROVs

A real-time visual odometry system using a monocular camera for underwater remotely operated vehicles (ROVs). The system performs feature tracking, pose estimation, and visual trajectory generation using OpenCV and deep learning object detection.





## 📽️ Demo Video

▶️ [Click here to watch the demo](docs/implementation1.mp4)






## 📄 Project Report

📥 [Download the full technical report (PDF)](docs/project_report.pdf) *(Upload this to /docs when ready)*






## 🚀 Features

- ✅ Real-time monocular visual odometry
- ✅ Feature detection and tracking using ORB/SIFT
- ✅ Pose estimation with essential matrix decomposition
- ✅ Deep learning-based object detection using YOLOv8
- ✅ Camera calibration for accurate motion recovery
- ✅ Trajectory visualization in 2D/3D






📥 [Download the full technical report (PDF)](docs/project_report.pdf) *(Upload this to /docs when ready)*





## 🛠️ Tech Stack

- Python 3.10
- OpenCV
- NumPy
- Ultralytics YOLOv8
- pytransform3d (for camera visualization)
- Matplotlib (optional for plotting)



## 📂 Folder Structure

vison-based-navigation/

├── CameraCalliberation.py # Generates intrinsic parameters

├── VisualOdometry.py # Visual odometry pipeline

├── ObjectTracker.py # Multi-object tracking logic

├── requirements.txt # Python dependencies

├── README.md # This file

├── docs/

│ ├── implementation1.mp4 # Demo video

│ └── project_report.pdf # Project report 



## ⚙️ Setup Instructions

### 1. Clone the repository

bash
git clone https://github.com/sargam15/vison-based-navigation.git
cd vison-based-navigation

### 2. Install dependencies

pip install -r requirements.txt

Make sure to also install ultralytics and pytransform3d if not in the file:

pip install ultralytics pytransform3d

### 3. Calibrate Your Camera (Optional)

Run this to generate intrinsicNew.npy:

python CameraCalliberation.py

This uses chessboard images in the current folder and saves the camera matrix for use in odometry.
### 4. Run the Visual Odometry System

python VisualOdometry.py



## Algorithms Used

 Feature detection: ORB / SIFT

 Pose estimation: Essential Matrix, RANSAC

 Odometry pipeline: Feature tracking → Matrix decomposition → Accumulated pose

 Object detection: YOLOv8 with Ultralytics

 Visualization: pytransform3d / matplotlib





## 🤝 Contributions

Pull requests are welcome. Please open an issue to discuss major changes first.



## 📜 License

This project is licensed under the MIT License.



## 🙋‍♀️ Author

Sargam Malik

📧 sargammalik004@gmail.com

🔗 Linkedin-https://www.linkedin.com/in/sargammalik/
