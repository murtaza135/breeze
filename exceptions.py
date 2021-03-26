class Error(Exception):
    """Base class for other exceptions
    
    Attributes:
        message -- explanation of the error
    """
    pass

class NoModelTrainedError(Error):
    """Raised when someone tries to manipulate a model in some way
    without first having trained it"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)