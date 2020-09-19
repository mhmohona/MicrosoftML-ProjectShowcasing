from openvino.inference_engine import IECore, IENetwork
import cv2
import logging


class Model:
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        self.model_structure = model_path
        self.model_weights = model_path.replace('.xml', '.bin')
        self.device_name = device
        self.threshold = threshold
        self.logger = logging.getLogger('fd')
        self.model_name = 'Basic Model'
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.network = None
        try:
            self.core = IECore()
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            self.logger.error("Error While Initilizing" + str(self.model_name) + str(e))
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def load_model(self):
        """
        This method with load model using IECore object
        return loaded model
        """
        try:
            self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error While Loading"+str(self.model_name)+str(e))

    def predict(self):
        pass

    def preprocess_output(self):
        pass

    def preprocess_img(self, image):
        """
        Input: image
        Description: We have done basic preprocessing steps
            1. Resizing image according to the model input shape
            2. Transpose of image to change the channels of image
            3. Reshape image
        Return: Preprocessed image
        """
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            self.logger.error("Error While preprocessing Image in " + str(self.model_name) + str(e))
        return image

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.network.requests[0].wait(-1)
        return status