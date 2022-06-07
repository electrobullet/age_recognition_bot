from typing import Any, Dict

import cv2 as cv
import numpy as np
import requests
from openvino.preprocess import PrePostProcessor
from openvino.runtime import AsyncInferQueue, Core, InferRequest, Layout, Model, Type

from main import TOKEN


def preprocess_model(model: Model):
    ppp = PrePostProcessor(model)

    ppp.input().tensor().set_layout(Layout('NHWC')).set_element_type(Type.u8)
    ppp.input().model().set_layout(Layout('NCHW'))

    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(Type.f32)

    return ppp.build()


def preprocess_image(image: np.ndarray, model: Model):
    _, h, w, _ = model.input().shape
    image = cv.resize(image, (w, h))
    return np.expand_dims(image, 0)


def face_detection_callback(infer_request: InferRequest, data: Dict[str, Any]):
    confidence = 0.6

    predictions = infer_request.results[face_detection_model.output()].reshape(-1, 7)
    predictions = predictions[predictions[:, 2] > confidence]

    if len(predictions) == 0:
        requests.post(
            f'https://api.telegram.org/bot{TOKEN}/sendMessage',
            {'chat_id': data['chat_id'], 'text': 'It seems there are no faces on the image.'},
        )
        return

    h, w, _ = data['image'].shape

    for _, _, confidence, x_min, y_min, x_max, y_max in predictions:
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        face_width = x_max - x_min
        face_height = y_max - y_min
        face_pad = abs(face_width - face_height) // 2

        if face_width < face_height:
            x_min = np.clip(x_min - face_pad, 0, w)
            x_max = np.clip(x_max + face_pad, 0, w)
        else:
            y_min = np.clip(y_min - face_pad, 0, h)
            y_max = np.clip(y_max + face_pad, 0, h)

        face_crop = data['image'][y_min:y_max, x_min:x_max]

        input_tensor = preprocess_image(face_crop, age_gender_model)
        age_gender_queue.start_async({0: input_tensor}, {'image': face_crop, 'chat_id': data['chat_id']})


def age_gender_callback(infer_request: InferRequest, data: Dict[str, Any]):
    age_prediction = infer_request.results[age_gender_model.output('fc3_a')].reshape(1)
    gender_prediction = infer_request.results[age_gender_model.output('prob')].reshape(2)

    age = age_prediction[0] * 100
    gender = ['female', 'male'][np.argmax(gender_prediction)]

    caption = f'Age: {age:.2f}\nGender: {gender} ({np.max(gender_prediction) * 100 :.2f}%)'
    _, image_bytes = cv.imencode('.png', data['image'])

    requests.post(
        f'https://api.telegram.org/bot{TOKEN}/sendPhoto',
        {'chat_id': data['chat_id'], 'caption': caption},
        files={'photo': image_bytes},
    )


def predict_and_answer(image: np.ndarray, chat_id: int):
    input_tensor = preprocess_image(image, face_detection_model)
    face_detection_queue.start_async({0: input_tensor}, {'image': image, 'chat_id': chat_id})


core = Core()

face_detection_model = core.read_model('models/face-detection-retail-0005.xml')
face_detection_model = core.compile_model(preprocess_model(face_detection_model))

face_detection_queue = AsyncInferQueue(face_detection_model, 2)
face_detection_queue.set_callback(face_detection_callback)

age_gender_model = core.read_model('models/age-gender-recognition-retail-0013.xml')
age_gender_model = core.compile_model(preprocess_model(age_gender_model))

age_gender_queue = AsyncInferQueue(age_gender_model, 2)
age_gender_queue.set_callback(age_gender_callback)
