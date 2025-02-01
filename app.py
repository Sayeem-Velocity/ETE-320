import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 100
imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return face_points


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    predictions = model.predict(facearray)
    probabilities = model.predict_proba(facearray).max(axis=1)
    return predictions, probabilities


def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    X_train, X_test, y_train, y_test = train_test_split(np.array(faces), np.array(labels), test_size=0.2,
                                                        random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)


def add_attendance(name):
    username, userid = name.split('_')
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if not ((df['Name'] == username) & (df['Roll'] == userid)).any():
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{datetime.now().strftime("%H:%M:%S")}')


def getallusers():
    users = os.listdir('static/faces')
    names, rolls = zip(*[user.split('_') for user in users]) if users else ([], [])
    return users, names, rolls, len(users)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', mess='No trained model found. Please add a face to continue.')

    cap = cv2.VideoCapture(0)
    added_faces = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)

        if len(faces) > 0:
            face_data = []
            face_locations = []
            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, -1)
                face_data.append(face.ravel())
                face_locations.append((x, y, w, h))

            face_data = np.array(face_data)
            predictions, confidences = identify_face(face_data)

            for (x, y, w, h), name, confidence in zip(face_locations, predictions, confidences):
                if confidence > 0.8 and name not in added_faces:
                    add_attendance(name)
                    added_faces.add(name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
                cv2.putText(frame, f"{name} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))


@app.route('/add', methods=['POST'])
def add():
    newusername, newuserid = request.form['newusername'], request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i = 0
    while i < nimgs:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            name = f'{newusername}_{i}.jpg'
            cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
            i += 1

        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)