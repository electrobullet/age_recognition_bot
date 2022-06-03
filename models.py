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
    ppp.output().tensor().set_element_type(Type.f32)

    return ppp.build()


def preprocess_image(image: np.ndarray, model: Model):
    _, h, w, _ = model.input().shape
    image = cv.resize(image, (w, h))
    return np.expand_dims(image, 0)


def face_detection_callback(infer_request: InferRequest, data: Dict[str, Any]):
    confidence = 0.8

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
        x_min = int(np.clip(x_min * w, 0, w))
        y_min = int(np.clip(y_min * h, 0, h))
        x_max = int(np.clip(x_max * w, 0, w))
        y_max = int(np.clip(y_max * h, 0, h))

        face_crop = data['image'][y_min:y_max, x_min:x_max]
        _, face_crop_bytes = cv.imencode('.png', face_crop)

        requests.post(
            f'https://api.telegram.org/bot{TOKEN}/sendPhoto',
            {'chat_id': data['chat_id']},
            files={'photo': face_crop_bytes},
        )


core = Core()

face_detection_model = core.read_model('models/face-detection-retail-0005.xml')
face_detection_model = core.compile_model(preprocess_model(face_detection_model))

face_detection_queue = AsyncInferQueue(face_detection_model, 2)
face_detection_queue.set_callback(face_detection_callback)


def predict_and_answer(image: np.ndarray, chat_id: int):
    data = {
        'image': image,
        'chat_id': chat_id,
    }

    input_tensor = preprocess_image(image, face_detection_model)
    face_detection_queue.start_async({0: input_tensor}, data)
