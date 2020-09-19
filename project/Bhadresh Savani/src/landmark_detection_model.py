from model import Model


class LandmarkDetectionModel(Model):
    """
    This is a class for the operation of Landmark Detection Model
    """

    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Landmark Detection Model class object
        """
        Model.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Landmark Detection Model'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))

    def predict(self, image, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        left_eye_image, right_eye_image, eye_cords = [], [], []
        try:
            p_image = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: p_image})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs[self.output_name]
                left_eye_image, right_eye_image, eye_cords = self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Error While making prediction in Landmark Detection Model" + str(e))
        return left_eye_image, right_eye_image, eye_cords

    def preprocess_output(self, outputs, image):
        """
        The net outputs a blob with the shape: [1, 10],
        containing a row-vector of 10 floating point values
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        All the coordinates are normalized to be in range [0,1].
        We only need
        """
        h = image.shape[0]
        w = image.shape[1]
        left_eye_image, right_eye_image, eye_cords = [], [], []
        try:
            outputs = outputs[0]

            left_eye_xmin = int(outputs[0][0][0] * w) - 10
            left_eye_ymin = int(outputs[1][0][0] * h) - 10
            right_eye_xmin = int(outputs[2][0][0] * w) - 10
            right_eye_ymin = int(outputs[3][0][0] * h) - 10

            left_eye_xmax = int(outputs[0][0][0] * w) + 10
            left_eye_ymax = int(outputs[1][0][0] * h) + 10
            right_eye_xmax = int(outputs[2][0][0] * w) + 10
            right_eye_ymax = int(outputs[3][0][0] * h) + 10

            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
            right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]

            eye_cords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax],
                         [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]

        except Exception as e:
            self.logger.error("Error While drawing bounding boxes on image in Landmark Detection Model" + str(e))
        return left_eye_image, right_eye_image, eye_cords