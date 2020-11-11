from nltk import text
from textblob import TextBlob as Blob
import speech_recognition as sr

r=sr.Recognizer()

tb=Blob('Hi this is amit')


iter_time=5
index=0
while(index<iter_time):
    with sr.Microphone() as source:
        print('say something')
        audio=r.listen(source,timeout=42)
        
        try:
            text=r.recognize_google(audio)
            tb=Blob(text)
            print(text)
            print(tb.sentiment)
        except :
            print('sorry... try again')
        index+=1

# print(tb.tags)
# print(tb.sentiment)


print(tb.sentiment)


