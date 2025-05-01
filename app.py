from flask import Flask, render_template_string, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

# Initialize camera and models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("C:/Users/Sushil Kumar Singh/Desktop/Model/keras_model.h5",
                        "C:/Users/Sushil Kumar Singh/Desktop/Model/labels.txt")

offset = 20
imgSize = 300
labels = ["Bad", "Hello", "iloveyou", "Namskaram", "No", "Okay", "Please", "thankyou", "welcome", "Yes"]

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 10), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# HTML Page with Bootstrap
HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Real-Time Sign Language Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body {
        background-color: #f8f9fa;
      }
      .navbar {
        margin-bottom: 30px;
      }
      .video-frame {
        border-radius: 15px;
        border: 6px solid #007bff;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
      }
      footer {
        margin-top: 40px;
        padding: 20px 0;
        background-color: #343a40;
        color: white;
      }
    </style>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">SignLang AI</a>
      </div>
    </nav>

    <div class="container text-center">
      <h1 class="mb-4">Real-Time Sign Language Detection</h1>
      <p class="lead mb-4">
        This system detects Indian Sign Language gestures live using a deep learning model trained on our own dataset.
      </p>
      <div class="card p-3 shadow-sm">
        <img src="{{ url_for('video_feed') }}" class="video-frame" width="720" height="540">
        <div class="card-body">
          <h5 class="card-title">Live Feed</h5>
          <p class="card-text text-muted">Show your sign in front of the camera to get real-time predictions.</p>
        </div>
      </div>

      <section class="mt-5">
        <h2>About This Project</h2>
        <p class="mt-3 fs-5">
          This web-based sign language recognition system supports real-time detection of Indian Sign Language. Our model is trained on a manually curated dataset of common gestures like “Hello”, “Thank you”, “Yes”, “No”, and more. The goal is to bridge the communication gap using AI and computer vision.
        </p>
        <p class="fs-5">
          Built using OpenCV, TensorFlow, and Flask — all running directly in your browser.
        </p>
        <a href="/shutdown" class="btn btn-danger mt-4">Quit Application</a>
      </section>
    </div>

    <footer class="text-center">
      <div class="container">
        <p>© 2025 SignLang AI — Built with ❤️ using our own Indian Sign Language dataset</p>
      </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    cap.release()
    cv2.destroyAllWindows()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Application shutting down..."

if __name__ == '__main__':
    app.run(debug=True)
