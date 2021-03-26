from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import config



authenticator=IAMAuthenticator(apikey=config.API_key)
tts=TextToSpeechV1(authenticator=authenticator)

tts.set_service_url(config.URL)


# with open('./speech.mp3','wb') as audio_file:
#     res=tts.synthesize("Bratati You Are Matha Mota you know that? right?",accept='audio/mp3',voice='en-US_AllisonV3Voice').get_result()
#     audio_file.write(res.content)

text=""

with open('test.txt','r',encoding="utf8") as f:
    texts=f.readlines()
    # print(f.readlines.text)

print(text)

texts=[ line.replace('\n',"") for line in texts]
text=''.join(str(line) for line in texts)


# print(text)


with open('./test.mp3','wb') as audio_file:
    res=tts.synthesize(text,accept='audio/mp3',voice='en-US_KevinV3Voice').get_result()
    audio_file.write(res.content)

