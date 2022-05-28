import cv2
import os
import face_recognition
import base64
import uuid
import numpy as np
from io import BytesIO
import pymongo
from PIL import Image

class FaceRecognizer:
  def __init__(self, *args, **kwargs):
    '''
    TODO:

    Initialize model for validation
    '''
    self.client = pymongo.MongoClient("mongodb+srv://kaushik:kd147953@project.hgun0.mongodb.net/myFirstDatabase?retryWrites=true&w=majority",tls=True,tlsAllowInvalidCertificates=True)
    self.collection = self.client['EmployeeManagementSystem']
    self.db = self.collection["EmployeeData"]

  def recognize(self, image, userId=None):
    '''
    TODO:

    Checks if image is in database
    Returns True if exists, else False

    If userId is None, check if image exists in the database
    '''
    if userId is None:
      return self.checkRepetition(image) 
    
    query = {"userId": userId}
    results = list(self.db.find(query))
    if len(results) == 0:
      return False # Access Denied
    storedImage = Image.open(BytesIO(base64.b64decode(results[0]["faceId"])))
    storedImage = self.processPILImage(storedImage)
    encodedStoredImage = face_recognition.face_encodings(storedImage)[0]

    img = self.processPILImage(image, convert=False)
    encodedTestImage = face_recognition.face_encodings(img)[0]

    return self.faceRecognition(encodedStoredImage, encodedTestImage)
  def register(self, image):
    '''
    TODO:

    Adds face to database
    Generates a userId and returns it
    '''
    return self.addToDatabase(image)

  def processPILImage(self, image, convert=True):
    if convert:
      return cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    return np.asarray(image)
  
  def addToDatabase(self, image):
    if not self.checkRepetition(image):
      uid = uuid.uuid4().hex
      buffered = bytesIO()
      image.save(buffered, format='PNG')
      data = {"faceId": base64.b64encode(buffered.getvalue()), "userId": uid}
      self.db.insert_one(data)
      return uid
    return None

  def checkRepetition(self, image):
    img = self.processPILImage(image, convert=False)
    encodedTestImage = face_recognition.face_encodings(img)[0]
    for document in self.db.find({}):
      storedImage = Image.open(BytesIO(base64.b64decode(document["faceId"])))
      storedImage = self.processPILImage(storedImage)
      encodedStoredImage = face_recognition.face_encodings(storedImage)[0]
      if self.faceRecognition(encodedStoredImage, encodedTestImage):
        return True
    return False
  
  def faceRecognition(self, encodedDbImage, encodedTestImage):
    return face_recognition.compare_faces([encodedDbImage],encodedTestImage)[0]
