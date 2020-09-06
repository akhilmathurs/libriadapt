
## DEPENDENCIES: https://github.com/jiaaro/pydub ##

'''

Script to augment clean speech files with noise. Use as follows:

python3 augment_noise.py \
        --dir <path to the directory with clean speech files>\
        --microphone <microphone from which the noise files should be sampled. Choose from {matrix | respeaker | pseye | usb | shure | nexus6}> \
        --noise <Noise type. Choose from {rain | wind | laughter}>

Example:

python3 augment_noise.py --dir /data/libriadapt/en-gb/clean/matrix/test/ --microphone matrix --noise rain

The script will read all *.wav files inside /data/libriadapt/en-gb/clean/matrix/test/, augment them with 'rain' noise sampled from
NOISE_DIR/matrix/. The noise-augmented audios will be saved in a new directory: /data/libriadapt/en-gb/rain/matrix/test/

'''

import random
from pydub import AudioSegment #pip install pydub 
import os
import argparse
import glob, pdb

random.seed(2020)

NOISE_DIR = "/data/libriadapt/noise"
GAIN_CONSTANT = -34.
NOISE_VOLUME=0.3


def add_ambient_noise(speech_file, noise_file, noise_type, normalise=False, bg_volume=0.3):

    # create an output file path by replacing 'clean' with the given noise_type  
    output_file = os.path.abspath(speech_file).replace('clean', noise_type)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        audio = AudioSegment.from_file(speech_file, 'wav')
        background = AudioSegment.from_file(noise_file, 'wav')
        if normalise:
            audio = audio.normalize()
            background = background.normalize()
        background = background.apply_gain(GAIN_CONSTANT * (1. - bg_volume))
        output = audio.overlay(background, position=0, loop=True)
        output = output.set_frame_rate(16000)
        output.export(output_file, format='wav')
    except Exception as e:
        print(e)
        raise

def augment_dataset():
    rain_noises = ['1-56311-A-10.wav', '1-63871-A-10.wav', '2-101676-A-10.wav', '2-117625-A-10.wav', '5-181766-A-10.wav', '5-188655-A-10.wav', '3-157487-A-10.wav', '3-157615-A-10.wav', '4-160999-A-10.wav', '4-161127-A-10.wav']
    wind_noises = ['1-51037-A-16.wav', '1-69760-A-16.wav', '2-104952-A-16.wav', '2-104952-B-16.wav', '3-246513-A-16.wav', '3-246513-B-16.wav', '4-144083-A-16.wav', '4-144083-B-16.wav', '5-157204-A-16.wav', '5-157204-B-16.wav']
    laughter_noises = ['1-72695-A-26.wav', '1-73123-A-26.wav', '2-109759-A-26.wav','2-109759-B-26.wav', '3-152912-A-26.wav', '3-152997-A-26.wav', '4-132803-A-26.wav', '4-132810-A-26.wav', '5-242932-B-26.wav','5-244526-A-26.wav']
    noise_dict = {'rain': rain_noises, 'wind': wind_noises, 'laughter': laughter_noises}

    noise_type = args.noise
    files = glob.glob(args.dir + '/*.wav')

    for file in files:
        speech_file = os.path.join(args.dir, file.strip())
        noise_file = os.path.join(NOISE_DIR, args.microphone, random.choice(noise_dict[noise_type]))
        add_ambient_noise(speech_file, noise_file, noise_type, normalise=True, bg_volume=NOISE_VOLUME)

parser = argparse.ArgumentParser(description='Noise augmentation')
parser.add_argument('--dir', dest='dir', action='store', type=str, help='path to the directory with clean speech files')
parser.add_argument('--microphone', dest='microphone', action='store', type=str, help='Microphone from which the noise files should be sampled. Choose from {matrix | respeaker | pseye | usb | shure | nexus6}')
parser.add_argument('--noise', dest='noise', action='store', type=str, help='Noise type. Choose from {rain | wind | laughter}')
args = parser.parse_args()

if __name__ == "__main__": 
    augment_dataset()