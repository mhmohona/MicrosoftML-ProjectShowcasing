from model import Model


class HeadPoseEstimationModel(Model):
    """
    This is a class for the operation of Head Pose Estimation Model
    """

    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Head Pose Estimation Model class object
        """
        Model.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Head Pose Estimation Model'
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
                outputs = self.network.requests[0].outputs
                f_output = self.preprocess_output(outputs)
        except Exception as e:
            self.logger.error("Error While prediction in Head Pose Estimation Model" + str(e))
        return f_output

    def preprocess_output(self, outputs):
        """
        Model output is a dictionary having below three arguments:
             "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
             "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
             "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        """
        final_output = []
        try:
            final_output.append(outputs['angle_y_fc'][0][0])
            final_output.append(outputs['angle_p_fc'][0][0])
            final_output.append(outputs['angle_r_fc'][0][0])
        except Exception as e:
            self.logger.error("Error While preprocessing output in Head Pose Estimation Model" + str(e))
        return final_output
