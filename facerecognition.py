class FaceRecognizer:
  def __init__(self, *args, **kwargs):
    '''
    TODO:

    Initialize model for validation
    '''
    self.model = None ## face recognition model

  def recognize(self, image, userId=None):
    '''
    TODO:

    Checks if image is in database
    Returns True if exists, else False

    If userId is None, check if image exists in the database
    '''
    pass
  def register(self, image):
    '''
    TODO:

    Adds face to database
    Generates a userId and returns it
    '''
    pass
