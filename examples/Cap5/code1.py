import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as signal

#乱数の種を初期化 np.random.seed(0)
#畳み込みに用いる音声波形 
clean_wave_file="cmu_arctic_us_aew_a0001.wav"
wav=wave.open(clean_wave_file)
data=wav.readframes(wav.getnframes())
data=np.frombuffer(data, dtype=np.int16)
data=data/np.iinfo(np.int16).max
wav.close()

# サ ン プ リン グ 周 波 数
sample_rate=16000
#畳み込むインパルス応答長
n_impulse_length=512
# イ ン パ ル ス 応 答 を 乱 数 で 生 成( ダ ミ ー )
impulse_response=np.random.normal(size=n_impulse_length)
conv_data=signal.convolve(data,impulse_response,mode='full')
