import os

from aiogram import Bot, Dispatcher, executor, types

bot = Bot(os.environ.get('AGE_RECOGNITION_BOT_TOKEN'))
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!\nI'm AgeRecognitionBot!\nPowered by aiogram and OpenVINO.")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer('Send me a photo with a face.')


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message: types.Message):
    await message.photo[-1].download(destination_file='test.png')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
