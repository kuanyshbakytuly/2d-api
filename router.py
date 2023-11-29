import asyncio
import base64

import cv2
import numpy as np
from fastapi import APIRouter
from loguru import logger

import schemas as schemas
from model import get_model_2d
from face_detection import face_detection

router = APIRouter(
    prefix='/face',
    tags=['face'],
)

face_liveness_model = get_model_2d()
INDEX = False

def predict_image_by_2D(frame):
    global INDEX
    CLASSES = ['cce', 'hp', 'print1', 'print2', 'real']

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    correct, face = face_detection(frame)
    cv2.imwrite('face.jpg', face)
    
    if correct and len(face):
        image = cv2.resize(face, (300, 300))
        output = face_liveness_model.predict(np.expand_dims(image/.255, 0))
        INDEX = np.argmax(output)

    return CLASSES[INDEX]

@router.post(
    "/verify",
    response_model=schemas.FaceLivenessOutput
)
async def verify_face(
        face_liveness_input = schemas.FaceLivenessInput,
):
    camera_image_b64: str = face_liveness_input.camera_image_b64

    camera_image: np.ndarray = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(base64.b64decode(camera_image_b64), np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    OUTPUT = predict_image_by_2D(camera_image)

    if OUTPUT == 'real':
        status = schemas.FaceLivenessStatus.true
    else:
        status = schemas.FaceLivenessStatus.false

    logger.info(f'status is {status}')
    return schemas.FaceLivenessOutput(status=status)


async def main():
    camera_image_path = '/Users/kuanyshbakytuly/Desktop/Relive/2d_api/images/Photo on 18.04.2023 at 23.58.jpg'

    camera_image: np.ndarray = cv2.imread(camera_image_path)
    image_bytes = cv2.imencode('.jpg', camera_image)[1].tobytes()
    camera_image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    input_data = schemas.FaceLivenessInput(
        camera_image_b64=camera_image_b64,
    )

    res = await verify_face(input_data)


if __name__ == '__main__':
    asyncio.run(main())
