from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import config



authenticator=IAMAuthenticator(apikey=config.API_key)
tts=TextToSpeechV1(authenticator=authenticator)

tts.set_service_url(config.URL)



import json
voices = tts.list_voices().get_result()
print(json.dumps(voices, indent=2))

