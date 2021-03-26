from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import config



authenticator=IAMAuthenticator(apikey=config.API_key)
tts=TextToSpeechV1(authenticator=authenticator)

tts.set_service_url(config.URL)

frere = """Frère Jacques
    Frère Jacques
    Dormez-vous?
    Dormez-vous?
    Sonnez les matines
    Sonnez les matines
    Ding, ding, dong
    Ding, ding, dong
    Frère Jacques
    Frère Jacques
    Dormez-vous?
    Dormez-vous?
    Sonnez les matines
    Sonnez les matines
    Ding, ding, dong
    Ding, ding, dong
    Ding, ding, dong
    Ding, ding, dong"""



with open('./frere.mp3', 'wb') as audio_file:
    res = tts.synthesize(frere, accept='audio/mp3', voice='fr-FR_ReneeV3Voice').get_result()
    audio_file.write(res.content)   