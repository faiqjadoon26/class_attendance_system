import cv2
import os
import numpy as np
from datetime import datetime
from openpyxl import load_workbook, Workbook

# Constants
DATASET_PATH = 'dataset'
TRAINER_PATH = 'trainer/trainer.yml'
ATTENDANCE_PATH = 'attendance.xlsx'

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model if it exists
if os.path.exists(TRAINER_PATH):
    recognizer.read(TRAINER_PATH)

# Global variable to store name labels
name_labels = {}


def capture_faces():
    name = input("Enter student name: ")
    student_path = os.path.join(DATASET_PATH, name)
    os.makedirs(student_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_path = os.path.join(student_path, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    global name_labels
    faces = []
    labels = []
    label_id = 0

    for root, dirs, files in os.walk(DATASET_PATH):
        for name in dirs:
            name_labels[name] = label_id
            student_path = os.path.join(root, name)
            for file in os.listdir(student_path):
                if file.endswith("jpg"):
                    img_path = os.path.join(student_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(np.array(img, 'uint8'))
                    labels.append(label_id)
            label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)
    print("[INFO] Training model completed.")
    print(f"[INFO] Name labels: {name_labels}")


def mark_attendance(name, status):
    date_str = datetime.now().date().isoformat()
    time_str = datetime.now().time().strftime("%H:%M:%S")
    if not os.path.exists(ATTENDANCE_PATH):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Name", "Date", "Time", "Status"])
        workbook.save(ATTENDANCE_PATH)

    workbook = load_workbook(ATTENDANCE_PATH)
    sheet = workbook.active

    for row in sheet.iter_rows(values_only=True):
        if row[0] == name and row[1] == date_str:
            return True

    sheet.append([name, date_str, time_str, status])
    workbook.save(ATTENDANCE_PATH)
    return False


def recognize_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)

            name = None
            for n, l in name_labels.items():
                if l == label:
                    name = n
                    break

            if name:
                print(f"[INFO] Recognized face as: {name} with confidence: {confidence:.2f}")
                if mark_attendance(name, 'Present'):
                    print(f"[INFO] {name} is already marked present for {datetime.now().date()}.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Return to main menu
            else:
                print("[INFO] Unknown face detected.")

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def remove_student():
    name = input("Enter the name of the student to remove: ")
    student_path = os.path.join(DATASET_PATH, name)
    if os.path.exists(student_path):
        for file in os.listdir(student_path):
            file_path = os.path.join(student_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(student_path)
        print(f"[INFO] Removed student {name} and their data.")
    else:
        print("[ERROR] Student not found.")


def main():
    while True:
        print("\n1. Capture Faces")
        print("2. Train Model")
        print("3. Recognize Faces")
        print("4. Remove Student")
        print("5. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            remove_student()
        elif choice == '5':
            break
        else:
            print("[ERROR] Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
