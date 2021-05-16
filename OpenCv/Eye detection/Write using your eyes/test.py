import pyglet
import time

# sound = pyglet.media.load("pop.wav",streaming=False)
# left_sound=pyglet.media.load('left.wav',streaming=False)
# right_sound=pyglet.media.load('right.wav',streaming=False)



# sound.play()
# time.sleep(1)
# left_sound.play()
# time.sleep(1)
# right_sound.play()
# time.sleep(1)
from playsound import playsound
  
# for playing note.wav file
playsound('left.wav')
time.sleep(10)