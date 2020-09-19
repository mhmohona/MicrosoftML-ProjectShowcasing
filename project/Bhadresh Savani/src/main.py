import cv2
import os
import logging
import time
import numpy as np
import pandas as pd
from input_feeder import InputFeeder
from face_detection_model import FaceDetectionModel
from landmark_detection_model import LandmarkDetectionModel
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel
from argparse import ArgumentParser
from scipy.spatial.distance import cosine

def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")

    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of xml file of landmark regression model")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Head Pose Estimation model")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Gaze Estimation model")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Specify path of input Video file or cam for webcam")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flag from ff, fl, fh, fg like -flags ff fl(Space separated if multiple values)"
                             "ff for faceDetectionModel, fl for landmarkRegressionModel"
                             "fh for headPoseEstimationModel, fg for gazeEstimationModel")

    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify Device for inference"
                             "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    return parser


def draw_preview(
        frame, preview_flags, cropped_image, left_eye_image, right_eye_image,
        face_cords, eye_cords, pose_output, gaze_vector):
    preview_frame = frame.copy()

    if 'ff' in preview_flags:
        if len(preview_flags) != 1:
            preview_frame = cropped_image
        cv2.rectangle(frame, (face_cords[0][0], face_cords[0][1]), (face_cords[0][2], face_cords[0][3]),
                      (0, 0, 0), 3)

    if 'fl' in preview_flags:
        cv2.rectangle(cropped_image, (eye_cords[0][0]-10, eye_cords[0][1]-10), (eye_cords[0][2]+10, eye_cords[0][3]+10),
                      (255, 0, 0), 2)
        cv2.rectangle(cropped_image, (eye_cords[1][0]-10, eye_cords[1][1]-10), (eye_cords[1][2]+10, eye_cords[1][3]+10),
                      (255, 0, 0), 2)

    if 'fh' in preview_flags:
        cv2.putText(
            frame,
            "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                pose_output[0], pose_output[1], pose_output[2]),
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

    if 'fg' in preview_flags:

        cv2.putText(
            frame,
            "Gaze Cords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                gaze_vector[0], gaze_vector[1], gaze_vector[2]),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

        x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
        le = cv2.line(left_eye_image.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        re = cv2.line(right_eye_image.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        preview_frame[eye_cords[0][1]:eye_cords[0][3], eye_cords[0][0]:eye_cords[0][2]] = le
        preview_frame[eye_cords[1][1]:eye_cords[1][3], eye_cords[1][0]:eye_cords[1][2]] = re

    return preview_frame


def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')

    is_benchmarking = False
    total_score = 0

    # initialize variables with the input arguments for easy access
    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }
    preview_flags = args.previewFlags
    input_filename = args.input
    device_name = args.device
    prob_threshold = args.prob_threshold
    output_path = args.output_path

    # add path for exercise video data
    exercise_video_path = '../bin/demo.mp4'
    exercise_gaze_path = '../bin/demo.csv'

    exercise_gaze_df = pd.read_csv(exercise_gaze_path)

    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    exercise_feeder = InputFeeder(input_type='video', input_file=exercise_video_path)

    for model_path in list(model_path_dict.values()):
        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file" + str(model_path))
            exit(1)

    # instantiate model
    face_detection_model = FaceDetectionModel(model_path_dict['FaceDetectionModel'], device_name, threshold=prob_threshold)
    landmark_detection_model = LandmarkDetectionModel(model_path_dict['LandmarkRegressionModel'], device_name, threshold=prob_threshold)
    head_pose_estimation_model = HeadPoseEstimationModel(model_path_dict['HeadPoseEstimationModel'], device_name, threshold=prob_threshold)
    gaze_estimation_model = GazeEstimationModel(model_path_dict['GazeEstimationModel'], device_name, threshold=prob_threshold)

    # load Models
    start_model_load_time = time.time()
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_model_load_time

    feeder.load_data()
    exercise_feeder.load_data()

    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.get_fps()/10),
                                (1000, 500), True)

    frame_count = 0
    gaze_vectors = []
    start_inference_time = time.time()
    for ret, frame in feeder.next_batch():

        # flip the image to make it similar to video image
        frame = np.flip(frame, 1)
        ex_ret, ex_frame = next(exercise_feeder.next_batch())

        if not ret:
            break

        # This will stop the cam when exercise video is over
        if len(exercise_gaze_df) <= len(gaze_vectors):
            break

        frame_count += 1

        key = cv2.waitKey(60)

        try:
            face_cords, cropped_image = face_detection_model.predict(frame)

            if type(cropped_image) == int:
                logger.warning("Unable to detect the face")
                if key == 27:
                    break
                continue

            left_eye_image, right_eye_image, eye_cords = landmark_detection_model.predict(cropped_image)
            pose_output = head_pose_estimation_model.predict(cropped_image)
            mouse_cord, gaze_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)
            gaze_vectors.append(gaze_vector)

        except Exception as e:
            logger.warning("Could predict using model" + str(e) + " for frame " + str(frame_count))
            continue

        if not len(preview_flags) == 0:
            preview_frame = draw_preview(
                frame, 'ff', cropped_image, left_eye_image, right_eye_image,
                face_cords, eye_cords, pose_output, gaze_vector)
            cropped_image = np.hstack((cv2.resize(ex_frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))

        instructor_gaze_vector = exercise_gaze_df.iloc[frame_count - 1].values
        score = cosine(instructor_gaze_vector, gaze_vector)
        if score > 0.1:
            total_score += 1

        # show score on output video
        cv2.putText(ex_frame, "Instructor Gaze Vector: {} ".format(instructor_gaze_vector), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 0), 2)
        cv2.putText(ex_frame, "User Gaze Vector: {}".format(gaze_vector), (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 0), 2)
        cv2.putText(ex_frame, "Gaze Match Score : {}".format(total_score), (40, 145), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
        ex_frame = cv2.rectangle(ex_frame, (20, 20), (1200, 160), (0, 0, 0), 2)


        image = np.hstack((cv2.resize(ex_frame, (500, 500)), cv2.resize(cropped_image, (500, 500))))

        cv2.imshow('preview', image)
        out_video.write(image)

        if key == 0:
            break

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    if input_filename=="cam":
        filename = "cam.csv"
    else:
        filename = input_filename.split("/")[-1].split(".")[0]+".csv"

    gaze_df = pd.DataFrame(gaze_vectors, columns=['vector_x', 'vector_y', 'vector_z'])
    gaze_df.to_csv(filename, index=False)
    logger.info('Model load time: ' + str(total_model_load_time))
    logger.info('Inference time: ' + str(total_inference_time))
    logger.info('FPS: ' + str(fps))
    logger.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()

    """
    try:
        os.mkdir(output_path)
    except OSError as error:
        logger.error(error)
        
    with open(output_path+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(total_model_load_time) + '\n')
    """

if __name__ == '__main__':
    main()
