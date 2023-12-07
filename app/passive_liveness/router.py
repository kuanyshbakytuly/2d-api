import asyncio
import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

import app.passive_liveness.schemas as schemas
from app.passive_liveness.model import get_model_2d
from app.face_detection.face_detection import face_detection

router = APIRouter(
    prefix='/passive_liveness',
    tags=['passive_liveness'],
)

face_liveness_model = get_model_2d()
CLASSES = ['cce', 'hp', 'print1', 'print2', 'real']


def predict_image_by_2d(
        frame: np.ndarray,
):
    
    face = face_detection(frame)
    if face is None:
        return None

    image = cv2.resize(face, (300, 300)) 
    liveness_prediction: np.ndarray = face_liveness_model.predict(np.expand_dims(image / .255, 0))
    return CLASSES[np.argmax(liveness_prediction)]


@router.post(
    "/verify",
    response_model=schemas.FaceLivenessOutput
)
async def passive_liveness(
        face_liveness_input: schemas.FaceLivenessInput,
):
    camera_image_b64: str = face_liveness_input.camera_image_b64

    camera_image: np.ndarray = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(base64.b64decode(camera_image_b64), np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    result = predict_image_by_2d(camera_image)
    if result is None:
        raise HTTPException(status_code=400, detail="Face not found in the input image")

    if result == 'real':
        status = schemas.FaceLivenessStatus.true
    else:
        status = schemas.FaceLivenessStatus.false

    logger.info(f'status is {status}')
    return schemas.FaceLivenessOutput(status=status)


async def main():
    camera_image_path = 'images/Photo on 18.04.2023 at 23.58.jpg'

    camera_image: np.ndarray = cv2.imread(camera_image_path)
    image_bytes = cv2.imencode('.jpg', camera_image)[1].tobytes()
    camera_image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    input_data = schemas.FaceLivenessInput(
        camera_image_b64=camera_image_b64,
    )

    res = await passive_liveness(input_data)


if __name__ == '__main__':
    asyncio.run(main())
