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
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      body {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        background-image: url('https://www.transparenttextures.com/patterns/diamond-upholstery.png');
        background-attachment: fixed;
      }

      body::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, #7f00ff30 0%, #e100ff30 100%);
        pointer-events: none;
      }

      .navbar {
        background: linear-gradient(to right, #000428, #004e92) !important;
        backdrop-filter: blur(15px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.2rem 0;
      }

      .navbar-brand {
        font-weight: 700;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 2px;
      }

      .container {
        background: linear-gradient(145deg, rgba(25, 118, 210, 0.1), rgba(3, 169, 244, 0.2));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 2px solid rgba(33, 150, 243, 0.2);
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 
            0 8px 32px rgba(3, 169, 244, 0.2),
            inset 0 0 80px rgba(33, 150, 243, 0.1);
        transition: all 0.3s ease;
      }

      .container:hover {
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.3),
            inset 0 0 100px rgba(52, 152, 219, 0.2);
      }

      .video-frame {
        border-radius: 20px;
        border: 4px solid #2196F3;
        box-shadow: 0 0 20px rgba(33, 150, 243, 0.3);
        transition: all 0.4s ease;
      }

      .video-frame:hover {
        transform: scale(1.02);
        border-color: #03A9F4;
        box-shadow: 0 0 30px rgba(3, 169, 244, 0.4);
      }

      .card {
        background: white;
        border: 1px solid rgba(33, 150, 243, 0.2);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(25, 118, 210, 0.1);
      }

      .card:hover {
        transform: translateY(-5px);
        background: white;  /* Keep background white on hover */
        box-shadow: 0 12px 48px rgba(33, 150, 243, 0.2);
        transition: all 0.3s ease;
      }

      h1 {
        background: linear-gradient(45deg, #1565C0, #0D47A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: none;
      }

      h2, h3 {
        background: linear-gradient(45deg, #1565C0, #0D47A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
      }

      .lead {
        color: #333333;
        font-size: 1.2rem;
        font-weight: 400;
        text-shadow: none;
      }

      .btn-danger {
        background: linear-gradient(45deg, #2196F3, #03A9F4);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
      }

      .btn-danger:hover {
        transform: translateY(-3px);
        background: linear-gradient(45deg, #03A9F4, #2196F3);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
      }

      .bg-dark {
        background: linear-gradient(145deg, #1976D2, #1565C0) !important;
        border-radius: 20px;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.3);
      }

      .card-title {
        color: #1565C0;
        font-weight: 700;
        letter-spacing: 1px;
        background: none;
        -webkit-background-clip: initial;
        -webkit-text-fill-color: initial;
      }

      .card-text {
        color: #333333;
        font-weight: 400;
      }

      footer {
        background: linear-gradient(to right, #000428, #004e92);
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
      }

      .animate-in {
        animation: fadeIn 1s ease-out;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .feature-icon {
        color: #1565C0;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: none;
      }

      .feature-icon i {
        background: linear-gradient(45deg, #1565C0, #0D47A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }

      /* Add these new hover effects */
      .card:hover {
        transform: translateY(-5px);
        background: white;  /* Keep background white on hover */
        box-shadow: 0 12px 48px rgba(33, 150, 243, 0.2);
        transition: all 0.3s ease;
      }

      .feature-icon:hover i {
        transform: scale(1.1);
        background: linear-gradient(45deg, #00b0ff, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.3s ease;
      }

      .bg-dark p {
        color: #333333;
      }

      footer p {
        color: white;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      }

      /* Feature section text color fix */
      .row.mt-5.text-center p {
        color: #333333;
      }

      /* About section text color fix */
      .bg-dark .fs-5 {
        color: white;  /* Make the paragraph text white too */
      }

      .bg-dark h2 {
        color: white;
        background: none;
        -webkit-background-clip: initial;
        -webkit-text-fill-color: initial;
      }
    </style>
  </head>
  <body>
    <!-- Navbar -->
    <nav class="navbar navbar-dark animate-in">
      <div class="container-fluid justify-content-center">
        <a class="navbar-brand mx-auto fw-bold" href="#">
          <i class="fas fa-hands"></i>
          Indian Sign Language Recognition System
        </a>
      </div>
    </nav>

    <!-- Main Content -->
    <div class="container animate-in">
      <h1 class="text-center animate__animated animate__fadeInDown">Real-Time Sign Language Detection</h1>
      <p class="lead text-center mb-5 animate__animated animate__fadeIn">
        Experience seamless communication through AI-powered sign language recognition.
      </p>

      <!-- Centered Video Card -->
      <div class="d-flex justify-content-center animate__animated animate__fadeInUp">
        <div class="card p-4">
          <img src="{{ url_for('video_feed') }}" class="video-frame" width="720" height="540">
          <div class="card-body text-center">
            <h5 class="card-title">Live Feed</h5>
            <p class="card-text">Show your sign in front of the camera for instant recognition</p>
          </div>
        </div>
      </div>

      <!-- Features Section -->
      <div class="row mt-5 text-center text-white g-4">
        <div class="col-md-4 animate__animated animate__fadeInLeft">
          <div class="feature-icon"><i class="fas fa-bolt"></i></div>
          <h3>Real-Time Detection</h3>
          <p>Instant recognition of signs with high accuracy</p>
        </div>
        <div class="col-md-4 animate__animated animate__fadeInUp">
          <div class="feature-icon"><i class="fas fa-brain"></i></div>
          <h3>AI-Powered</h3>
          <p>Advanced deep learning model for precise results</p>
        </div>
        <div class="col-md-4 animate__animated animate__fadeInRight">
          <div class="feature-icon"><i class="fas fa-language"></i></div>
          <h3>Indian Signs</h3>
          <p>Specialized in Indian Sign Language gestures</p>
        </div>
      </div>

      <!-- About Section -->
      <section class="mt-5 animate__animated animate__fadeIn">
        <div class="bg-dark text-white p-5 rounded-lg">
          <h2 class="text-center mb-4">About This Project</h2>
          <p class="fs-5">
            This web-based sign language recognition system supports real-time detection of Indian Sign Language. Our model is trained on a manually curated dataset of common gestures like "Hello", "Thank you", "Yes", "No", and more. The goal is to bridge the communication gap using AI and computer vision.
          </p>
          <div class="text-center mt-4">
            <a href="/shutdown" class="btn btn-danger">Quit Application</a>
          </div>
        </div>
      </section>
    </div>

    <!-- Footer -->
    <footer class="text-center animate__animated animate__fadeIn">
      <div class="container">
        <p>© 2025 SignLang AI — Built with ❤️ using our own Indian Sign Language dataset</p>
      </div>
    </footer>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://kit.fontawesome.com/your-font-awesome-kit.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
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
