#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave
#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np
#scipyのsignalモジュールをインポート（stft等信号処理計算用)
import scipy.signal as sp
#sounddeviceモジュールをインポート
import sounddevice as sd
#matplotlib
import matplotlib.pyplot as plt

# STFTを自前で実装してみる
# [INPUT]
#     sig: input signal (len x 1) 入力信号
#      fs: サンプリング周波数(wav.getframerate()で取得可能)
#   wndow: analysis window function (frlen x 1, default: hamming window) 窓関数(wid)
#  nperseg: frame length (even number, must be dividable by frsft) フレーム長(frlen)
# noverlap: frame shift  (default: frlen/2) フレームシフト(frsft)
# ex) STFT(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
# [OUTPUT]
# f : 出力されるデータの周波数軸の各インデックスの具体的な周波数〔Hz〕
# t : フレーム軸の各インデックスの具体的な時間〔sec〕
#  stft_data: STFT of input signal (frlen/2+1 x nf) STFTの信号 短時間 フーリエ変換後の複素数の信号 y(l, k)

def STFT(sig,frlen,frsft,wnd):

    return sig



#読み込むサンプルファイル
sample_wave_file="/Users/kenta/Programing/github/yama-pyroomacoustics-learning/python_source_separation-master/section2/CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0001.wav"
#ファイルを読み込む
wav=wave.open(sample_wave_file)
#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())
#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#特定の周波数成分を消す(100番目の周波数よりも高い周波数成分を全て消す)
stft_data[100:,:]=0

#時間領域の波形に戻す
t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
data_post=data_post.astype(np.int16)

#dataを再生する
sd.play(data_post,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()

#waveファイルを閉じる
wav.close()

# 変換前と復元後が一緒であることを確認する（二つ並べてみる）
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

# 変換前のスペクトグラム
spectrum, freqs, t, im=ax1.specgram(data,NFFT=512,noverlap=512/16*15, Fs=wav.getframerate())
# サブタイトル
ax1.set_title('Original Sound')
# x軸のラベル
ax1.set_xlabel("Time [sec]")
# y軸のラベル
ax1.set_ylabel("Frequency [Hz]")

# 復元後のスペクトグラム
spectrum, freqs, t, im=ax2.specgram(data_post,NFFT=512,noverlap=512/16*15, Fs=wav.getframerate())
# サブタイトル
ax2.set_title('Restored Sound')
# x軸のラベル
ax2.set_xlabel("Time [sec]")
# y軸のラベル
ax2.set_ylabel("Frequency [Hz]")

#レイアウトの設定
fig.tight_layout()
#plt.savefig("./spectrogram_stft_istft.png")
plt.show()

