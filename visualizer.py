# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 02:19:24 2020

@author: strai
"""

import pyaudio
import numpy as np
import time
import librosa
import librosa.display
import sys
import pygame
from pygame.locals import *
import random


colors = [(10, 255, 50), (255, 10, 100), (10, 150, 150), (100, 10, 255), (50, 70, 200), (170, 29, 85)]


#For color fading
global_color = (0, 50, 255)
if_change = False
color = (0, 50, 255)
down = True

pygame.init()
FPS = 40
fpsClock = pygame.time.Clock()
#(width, height)
screen = pygame.display.set_mode((800,900), 0, 32)
screen.fill((0,0,0))
pygame.display.set_caption('fft_visualizing')
pygame.display.flip()


class audioHandler():
        
        def __init__(self):
            
            self.pa = pyaudio.PyAudio()
            self.stream = 0
            
            self.audio_data = 0
            self.callback_output = []
            self.rate = 44100
            self.channels = 1
            self.format = pyaudio.paFloat32
            
            self.D = None
            self.D_harmonic = None
            self.D_percussiv = None
            
            #For pygame
            self.max_height = 750
            self.bar_start_h = 890
            self.bar_start_w = 44
            self.bar_width = 5
            
        
        def start(self):
            
            def callback(in_data, frame_count, time_info, flag):
                
                self.audio_data = np.frombuffer(in_data, dtype=np.float32)
                if len(self.callback_output) > 40:
                    self.callback_output.pop(0)
                self.callback_output.append(self.audio_data)
                self.conc = np.concatenate(self.callback_output)
                    
                #For tempo
                onset_env = librosa.onset.onset_strength(self.conc, sr=self.rate)
                self.tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.rate, aggregate=None)
                
                # the magnitude of frequency bin f at frame; amplitude values according to frequency and time indexes
                self.D = np.abs(librosa.stft(self.audio_data,  hop_length=512, n_fft=2048*4))
                # decibel matrix
                self.spectrogram = librosa.amplitude_to_db(self.D, ref=np.max)

                self.frequencies = librosa.core.fft_frequencies(n_fft=2048*4)

                self.times = librosa.core.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=self.rate, hop_length=512, n_fft=2048*4)

                self.time_index_ratio = len(self.times)/self.times[len(self.times) - 1]

                self.frequencies_index_ratio = len(self.frequencies)/self.frequencies[len(self.frequencies)-1]
                #self.D_harmonic, self.D_percussive = librosa.decompose.hpss(self.D)
                #print(self.D_harmonic, self.D_percussive)
                
                return (None, pyaudio.paContinue)
            
            
            #past input_device_index only if use stereomix
            self.stream = self.pa.open(format=self.format,
                 channels=self.channels,
                 rate=self.rate,
                 output=False,
                 input=True,
                 input_device_index = 0,
                 stream_callback=callback)
            
            self.stream.start_stream()
            self.active = True
            #time sleep for small delay, otherwise stream can't make callback - nothing to register
            time.sleep(0.1)
            
        def get_decibel(self, target_time, freq):
            return self.spectrogram[int(freq*self.frequencies_index_ratio)][int(target_time*self.time_index_ratio)]
        
        def get_decibel_array(self, target_time, freq_arr):
            arr = []
            for f in freq_arr:
                arr.append(self.get_decibel(target_time,f))
            return arr
                    
        def loop(self):
            while self.stream.is_active():
                try:
                    global  down, color, if_change, global_color
                    fft_data = np.fft.rfft(self.audio_data * np.blackman(len(self.audio_data)))
                    #attempt to use harmonic
                    #harmonic = librosa.amplitude_to_db(self.D_harmonic)
                    #fft_data = np.fft.rfft(harmonic)
                    
                    fft_freq = np.fft.fftfreq(len(fft_data),d=1/44100)
                    global color, colors
                                                

                    ps = self.scale_tempo_to_pause(self.tempo[0])
                    
                    screen.fill((255, 255, 255))

                    ps = self.scale_tempo_to_pause(self.tempo[0])
                    #Because we have only 256 positive frequency values
                    #rectangle for 2 frequencies
                    for i in range(1, 126):
                        rect = self.make_rectangle(fft_data[i:i+2], i)
                        pygame.draw.rect(screen, color, rect)
                        
                    pygame.display.flip()
                    
                    #For color fading
                    if if_change:
                        color = random.choice(colors)
                        global_color = color
                        if_change = False
                    if down:
                        if (color[0] <= global_color[0] and color[1] <= global_color[1] and color[2] <= global_color[2]):
                            down = False
                        else:
                            color = list(color)
                            for i in range(len(color)):
                                if color[i] >= global_color[i]+ps:
                                    color[i] -= ps 
                                elif color[i] >= global_color[i]:
                                    color[i] -= 1
                            color = tuple(color)
                        
                    else:
                        if (color[0] >= 240 and color[1] >= 240 and color[2] >= 240):
                            down  = True
                        else:
                            color = list(color)
                            for i in range(len(color)):
                                if color[i] != 240:
                                    if (color[i] + ps) <= 240:
                                        color[i] += ps
                                    elif color[i] <= 240: 
                                        color[i] += 1
                                    
                            color = tuple(color)    
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.end()
                            sys.exit()
                    
                except KeyboardInterrupt:
                    self.end()
        
        #Large tempo - frequent fading
        def scale_tempo_to_pause(self, tempo, tp_min=90.0, tp_max=190, ps_min=1, ps_max = 10):
            return int((tempo - tp_min) * (ps_max-ps_min) / (tp_max-tp_min) + ps_min)
        
        def make_rectangle(self, freq_val, num):
            global if_change
            #When using stereomix, should be decreased to 
            if np.average(np.abs((freq_val))) > 0.5:
                if_change = True
            left = self.bar_start_w + self.bar_width*num
            width = self.bar_width
            #When using stereomix, should be increased
            height = int(np.average(np.abs((freq_val))) * 1000)
            if height > self.max_height:
                height = self.max_height
            if np.average(np.abs((freq_val))) == 0:
                height = 1
            top = self.bar_start_h - height
            return (left, top, width, height)
        
        
        def end(self):
            self.stream.close()
            self.pa.terminate()
            pygame.display.quit()
            pygame.quit()
            sys.exit()
            
        def on_press(self, key):
            if key == 'q':
                self.end()
            self.end() 
            
                
adh = audioHandler()
adh.start()
adh.loop()
