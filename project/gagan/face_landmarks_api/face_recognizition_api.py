import logging

import dlib
import cv2


face_detector = dlib.get_frontal_face_detector()

predictor_model = 'models/shape_predictor_68_face_landmarks.dat'
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = 'models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def resize_img(img):
    """
    Resizes the image and applies aspect ratio correction
    """
    if img.shape[0] > 800:
        baseheight = 500
        w = (baseheight / img.shape[0])
        p = int(img.shape[1] * w)
        img = cv2.resize(img, (baseheight, p))

    elif img.shape[1] > 800:
        baseheight = 500
        w = (baseheight / img.shape[1])
        p = int(img.shape[0] * w)
        img = cv2.resize(img, (p, baseheight))

    return img


def tuple_to_rect(rect: tuple):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param rect:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])


def raw_face_locations(img, number_of_times_to_upsample: int=1):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(img, number_of_times_to_upsample)


def raw_face_landmarks(face_image, face_locations=None) -> list:
    """
    Extracts the landmark points
    """
    if face_locations is None:
        face_locations = raw_face_locations(face_image)
    else:
        face_locations = [tuple_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=1) -> list:
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = raw_face_landmarks(face_image, known_face_locations)

    return [list(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
            for raw_landmark_set in raw_landmarks]


def get_encoding(image: ) -> dict:
    """
    Applies various transformations like resizing and aspect ratio correction
    """
    img = resize_img(image)

    try:
        key_points = face_encodings(img)
        points = {}
        points['points'] = key_points[0]
        return points
    except Exception as e:
        logging.error('Error'+str(e))
        raise AttributeError("Something went wrong")
