import numpy as np
from model import Model


class FaceDetectionModel(Model):
    """
    This is a class for the operation of Face Detection Model
    """

    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Face Detection Model class object
        """
        Model.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Face Detection Model'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def predict(self, image, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        try:
            p_image = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: p_image})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs[self.output_name]
                detections, cropped_image = self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Error While Prediction in Face Detection Model" + str(e))
        return detections, cropped_image

    def preprocess_output(self, coords, image):
        """
        We will have multiple detection for single image
        This function will take image and preprocessed cordinates
        and return image with bounding boxes and scaled cordinates
        """
        width, height = int(image.shape[1]), int(image.shape[0])
        detections = []
        cropped_image = image
        coords = np.squeeze(coords)
        try:
            for coord in coords:
                image_id, label, threshold, xmin, ymin, xmax, ymax = coord
                if image_id == -1:
                    break
                if label == 1 and threshold >= self.threshold:
                    xmin = int(xmin * width)
                    ymin = int(ymin * height)
                    xmax = int(xmax * width)
                    ymax = int(ymax * height)
                    detections.append([xmin, ymin, xmax, ymax])
                    cropped_image = image[ymin:ymax, xmin:xmax]
        except Exception as e:
            self.logger.error("Error While drawing bounding boxes on image in Face Detection Model" + str(e))
        return detections, cropped_image
