from flask import Flask, jsonify, request
from PIL import Image
from facerecognition import FaceRecognizer
from emotionrecognition import EmotionRecognizer
from waitress import server

app = Flask(__name__)

# Initialize models
faceRecognizer = FaceRecognizer()
emotionRecognizer = EmotionRecognizer()

@app.route('/logAttendance', methods = ['POST'])
def identify():
  file = request.files.get('file')
  uid = request.headers.get('userId')
  if file is None or uid is None:
    return jsonify({"verified":False, "emotion":"Unknown", "message":"Either image or UserId is missing"}), 400 # Bad request error code
  img = Image.open(file)
  if faceRecognizer.recognize(img, uid): # send image from multipart data and userId from header
    # Person identifued successfully
    emotion = emotionRecognizer.classify(img) # send image
    return jsonify({"verified":True, "emotion":emotion, "message":"Logging Successful"}), 200 # Successful Status Code
  else:
    # Recognition failed
    return jsonify({"verified":False, "emotion":"Unknown", "message":"Provided UserId does not match for the Face"}), 401 # Unauthorized Status Code

@app.route("/register", methods = ['POST'])
def registerFace():
  file = request.files.get('file')
  if file is None:
    return jsonify({"successful":False, "userId": "", "message":"No image provided"}), 400 # Bad Request
  img = Image.open(file)
  if not faceRecognizer.recognize(img): # send image from multipart data (DO NOT SEND USER ID)
    # if face does not exist in database
    generatedId = faceRecognizer.register(img) # send image from multipart data
    return jsonify({"successful":True, "userId": generatedId, "message":"Face registered successfully"}), 200 # Successful Status Code
  else:
    return jsonify({"successful":False, "userId": "", "message":"Face already registered"}), 403 # Forbidden Ststus Code
    
if __name__ == '__main__':
  #app.run(debug=False, host='0.0.0.0')
  server(app, port=5000, host='0.0.0.0')
