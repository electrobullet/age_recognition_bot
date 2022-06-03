# age_recognition_bot

Telegram bot for Age and gender recognition powered by aiogram and OpenVINO™.  
Message processing and image inference is completely asynchronous.  

> At the moment, the bot does not have a permanent hosting, but I run it from time to time.  
> You can send a message [here](https://t.me/age_recognition_bot). It will be processed as soon as the bot appears on the network.  

## Example
![example](example.bmp)

## How to run
1. Add AGE_RECOGNITION_BOT_TOKEN environment variable containing the bot telegram token.

2. Download the necessary models from the repository [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) and place them in the `models` folder.

    * [age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013)
    
    * [face-detection-retail-0005](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0005)

3. Run the telegram bot.
```
python main.py
```
