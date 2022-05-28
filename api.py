from flask import Flask, jsonify, request
from PIL import Image
import base64

app = Flask(__name__)

# Initialize models
faceRecognizer = FaceRecognizer()
emotionRecognizer = EmotionRecognizer()

@app.route('/logAttendance', methods = ['POST'])
def identify():
  file = request.files['file']
  img = Image.open(file)
  if faceRecognizer.recognize(img, uid): # send image from multipart data and userId from header
    # Person identifued successfully
    emotion = emotionRecognizer.classify(img) # send image
    return jsonify({"status":200, "verified":True, "emotion":emotion})
  else:
    # Recognition failed
    return jsonify({"status":201, "verified":False, "message":"This person does not exist in the database"})

@app.route("/register", methods = ['POST'])
def registerFace():
  if not faceRecognizer.recognize(): # send image from multipart data (DO NOT SEND USER ID)
    # if face does not exist in database
    generatedId = faceRecognizer.register() # send image from multipart data
    return jsonify({"status":200, "successful":True, "userId": generatedId, "message":"Face registered successfully"})
  else:
    return jsonify({"status":201, "successful":False, "message":"Face registration failes"})
    
if __name__ == '__main__':
  app.run(debug=True)
