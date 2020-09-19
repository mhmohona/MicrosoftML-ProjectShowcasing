import math
from model import Model


class GazeEstimationModel(Model):
    """
    Class for the Gaze Estimation Model.
    """

    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Gaze Estimation Model class object
        """
        Model.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Face Detection Model'
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [o for o in self.model.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, hpe_cords, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        try:
            left_eye_image = self.preprocess_img(left_eye_image)
            right_eye_image = self.preprocess_img(right_eye_image)
            self.network.start_async(request_id, inputs={'left_eye_image': left_eye_image,
                                                         'right_eye_image': right_eye_image,
                                                         'head_pose_angles': hpe_cords})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                mouse_cord, gaze_vector = self.preprocess_output(outputs, hpe_cords)
        except Exception as e:
            self.logger.error("Error While Prediction in Gaze Estimation Model" + str(e))
        return mouse_cord, gaze_vector

    def preprocess_output(self, outputs, hpe_cords):
        """
        Model output is dictionary like this
        {'gaze_vector': array([[ 0.51141196,  0.12343533, -0.80407059]], dtype=float32)}
        containing Cartesian coordinates of gaze direction vector

        We need to get this value and convert it in required format
        hpe_cords which is output of head pose estimation is in radian
        It needed to be converted in catesian cordinate
        """
        gaze_vector = outputs[self.output_name[0]][0]
        mouse_cord = (0, 0)
        try:
            angle_r_fc = hpe_cords[2]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_cord = (x, y)
        except Exception as e:
            self.logger.error("Error While preprocessing output in Gaze Estimation Model" + str(e))
        return mouse_cord, gaze_vector
