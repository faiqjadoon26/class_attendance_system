Face Recognition Attendance System
A Python-based face recognition system that marks attendance by detecting and recognizing faces in real time. The system uses OpenCV and the face_recognition library to identify individuals and log their attendance automatically.

Features
Real-time face detection and recognition

Automatic attendance marking with timestamps

Easy to add new faces to the dataset

Stores attendance in a CSV file

Technologies Used
Python 3

OpenCV

face_recognition

NumPy

Pandas

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add known faces to the images (or dataset) folder.

Usage
Run the script:

bash
Copy
Edit
python main.py
The system will start the webcam, detect faces, and record attendance in attendance.csv.

Folder Structure
bash
Copy
Edit
face-recognition-attendance/
│── images/           # Known faces
│── main.py           # Main script
│── attendance.csv    # Attendance log
│── requirements.txt  # Dependencies
│── README.md         # Project documentation
Future Improvements
Add a graphical user interface (GUI)

Support for cloud-based databases

Improve accuracy with deep learning models

Created by Faiq Jadoon
