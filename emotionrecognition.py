import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D



class EmotionRecognizer:
  def __init__(self, *args, **kwargs):
    '''
    TODO:

    Initialize model for validation
    '''
    self.model = Sequential()
    self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    self.model.add(Flatten())
    self.model.add(Dense(1024, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(7, activation='softmax'))

    self.model.load_weights('model.h5')

    #cv2.ocl.setUseOpenCL(False)

    
    #{0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    #self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    self.emotion_dict = {0: "Angry", 1: "Sad", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Neutral"}

  
  def classify(self, image):
    '''
    TODO:

    Classifies the emotion of the face
    Returns a string denoting the classified emotion
    '''
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    #faces = self.facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    #print(faces)
    #for (x, y, w, h) in faces:
      #roi_gray = gray[y:y + h, x:x + w]
      #cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
      # cv2.imshow("Window",cropped_img)
      # cv2.imshow("Window 2",gray)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      #prediction = self.model.predict(cropped_img)
      #maxEmotion = self.emotion_dict.get(int(np.argmax(prediction)))
      #return maxEmotion
    prediction=self.model.predict(np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0))
    maxEmotion = self.emotion_dict.get(int(np.argmax(prediction)))
    return maxEmotion
    