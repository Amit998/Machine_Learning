from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import authenticate


api_key=authenticate.api_key_for_speech_to_text
url=authenticate.URL_for_speech_to_text





authenticator=IAMAuthenticator(api_key)
stt=SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)


import subprocess 
import os
# command = 'ffmpeg -i audio.wav -vn -ar 44100 -ac 2 -b:a 192k audio.mp3'
# subprocess.call(command, shell=True)
# command = 'ffmpeg -i audio.mp3 -f segment -segment_time 360 -c copy %03d.mp3'
# subprocess.call(command, shell=True)

files = []
for filename in os.listdir('.'):
    if filename.endswith(".mp3") and filename !='audio.mp3':
        files.append(filename)
files.sort()

print(files)