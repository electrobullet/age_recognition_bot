import io
import os

import cv2 as cv
import numpy as np
from aiogram import Bot, Dispatcher, executor, types

import models

TOKEN = os.environ.get('AGE_RECOGNITION_BOT_TOKEN')

bot = Bot(TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def info(message: types.Message):
    await message.reply("Hi!\nI'm AgeRecognitionBot!\nPowered by aiogram and OpenVINO.")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer('Send me a photo with a face.')


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    image = await get_image(message.photo[-1])
    models.predict_and_answer(image, message.chat.id)


async def get_image(photo: types.PhotoSize):
    image_bytes = io.BytesIO()
    await photo.download(destination_file=image_bytes)
    image = np.array(bytearray(image_bytes.getvalue()), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    image_bytes.close()
    return image


if __name__ == '__main__':
    executor.start_polling(dp)
