import os
from pydub import AudioSegment
from webapp import loader, utils

basedir = os.path.join(os.getcwd(), 'data')

m4a_audio = AudioSegment.from_file(os.path.join(basedir, "heartsound_phone_rec.m4a"), format="m4a")
m4a_audio.export(os.path.join(basedir, "heartsound_phone_rec.wav"), format="wav")

data = loader.DataLoader(basedir)
mel_result = data.mel_spectrogram_loader("heartsound_phone_rec.wav")