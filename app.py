from flask import Flask, jsonify, request
from PIL import Image
from facerecognition import FaceRecognizer
from emotionrecognition import EmotionRecognizer

app = Flask(__name__)

# Initialize models
faceRecognizer = FaceRecognizer()
emotionRecognizer = EmotionRecognizer()

@app.route('/logAttendance', methods = ['POST'])
def identify():
  file = request.files['file']
  img = Image.open(file)
  uid="724ce139b7514ba987941736bcc21178" #uid should be retrived from the request it is only for demo
  if faceRecognizer.recognize(img, uid): # send image from multipart data and userId from header
    # Person identifued successfully
    emotion = emotionRecognizer.classify(img) # send image
    return jsonify({"verified":True, "emotion":emotion, "message":"Logging Successful"}), 200 # Successful Status Code
  else:
    # Recognition failed
    return jsonify({"verified":False, "emotion":"Unknown", "message":"This person does not exist in the database"}), 401 # Unauthorized Status Code

@app.route("/register", methods = ['POST'])
def registerFace():
  file = request.files['file']
  img = Image.open(file)
  if not faceRecognizer.recognize(img): # send image from multipart data (DO NOT SEND USER ID)
    # if face does not exist in database
    generatedId = faceRecognizer.register(img) # send image from multipart data
    return jsonify({"successful":True, "userId": generatedId, "message":"Face registered successfully"}), 200 # Successful Status Code
  else:
    return jsonify({"successful":False, "userId": "", "message":"Face registration failed"}), 403 # Forbidden Ststus Code
    
if __name__ == '__main__':
  app.run(host='0.0.0.0')
