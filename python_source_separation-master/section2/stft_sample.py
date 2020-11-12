# stft_sample.py
#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave
#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np
#scipyのsignalモジュールをインポート（stft等信号処理計算用)
import scipy.signal as sp
import scipy
#sounddeviceモジュールをインポート
import sounddevice as sd
#matplotlib
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import cv2


def hamming(nperseg):
    t = sp.hamming(nperseg) # 関数を使う方法
    #print(t)
    t = np.array(t)
    #print(type(t))
    """
    t = np.ones((1,nperseg))
    for i in range(len(t)):
        t[i]=0.54+0.46*math.cos(2.0*math.pi/i*t)
    print(t)
    """

    return t

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
# stft_data: STFT of input signal (frlen/2+1 x nf) STFTの信号 短時間 フーリエ変換後の複素数の信号 y(l, k)

def STFT(sig,fs,frlen,frsft,wnd):
    # フレームの数を測る
    nf = int(np.ceil((len(sig)-frlen)/frsft)+1) # データサイズをフレームサイズで割っている
    print(nf)
    # ハミング窓を計算するために転地する
    hamming_trans = (hamming(frlen)).T

    stft_data = []
    for i in range(1,nf): # iの関係で1から215まで
        st =(i-1) * frsft # ひとつ前の場所から窓をかける
        # 窓関数をかける & FFTを行う
        data_stft_split = np.fft.fft((sig[st:st+frlen].T*hamming(frlen))) # [x+iy,x+iy,...]の形(frlen個)
        #print(i,st,sig.shape)
        stft_data.append(data_stft_split)
    stft_data = np.array(stft_data)

    # 転置して元に戻す
    stft_data = stft_data.T
    num = int(stft_data.shape[0]/2)+1
    stft_data = stft_data[0:num] # 大きさを半分にする(元の音声が実数だから)
    return fs,nf,stft_data

# opt_synwndを作成
def opt_synwnd(anawnd,frsft):
    # ハニング窓を元に戻すやつ
    print(anawnd.shape)
    col,frlen = anawnd.shape
    print("AAAAAAAAAAAAAAAA",frlen,col)
    synwnd = np.zeros((frlen),dtype="float64")
    print(synwnd.shape)
    for i in range(frsft):
        amp = 0
        for j in range(int(frlen/frsft)):
            print(i+1 + (j)* frsft)
            amp = amp + anawnd[i+1 + (j)* frsft] * anawnd[i+1 + (j)* frsft]

        for j in range(int(frlen/frsft)):
            synwnd[i+(j-1) * frsft] = anawnd[i+(j-1)*frsft,1]/amp
        print(synwnd.shape)
    return synwnd


# invSTFT: inverse Short-time Fourier Transform
# [inputs]
#     tf: STFT of input signal (frlen/2+1 x nf)
# length: length of output signal
#  frsft: frame shift  (default: frlen/2)
#    wnd: synthesis window function (frlen x 1, default: optimum window for hamming analysis window)
# [outputs]
#    sig: input signal (length x 1)

def invSTFT(stft_data,fs,frlen,frsft,wnd):
    #t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
    nf = stft_data.shape[1] # 分割したフレームの数
    subspc = np.empty((frlen,1),dtype = "float64")
    tmpsig = np.zeros(((nf-1)*frsft+frlen,1))
    hamming_data = [hamming(frlen)]
    hamming_data = np.array(hamming_data)
    print(hamming_data.shape)
    print("----------------------------")
    #hamming_data = opt_synwnd(hamming_data,frsft)
    print("hamming_data.shape",hamming_data.shape)
    for i in range(nf):
        st = (i*frsft) # 最初の場所を指定している
        subspc[0:int(frlen/2)+1,0]=stft_data[:,i]
        subspc[0,0]=subspc[0,0]/2
        subspc[int(frlen/2)+1,0]=subspc[int(frlen/2)+1,0]/2

        # iFFTして出力配列に代入
        #a = np.fft.ifft(subspc)
        #b = np.fft.ifft(subspc)*(hamming(frlen))
        #print(hamming(frlen))
        #print("b",b.shape)
        #print(((np.fft.ifft(subspc)*(hamming(frlen)).T).real*2).shape)
        #print(tmpsig[st+0:st+frlen+1,0].shape,((np.fft.ifft(subspc))*hamming_data.T.real*2).shape)
        tmpsig[st:st+frlen] = tmpsig[st:st+frlen]+((np.fft.ifft(subspc)*hamming_data.T).real*2)
    print(tmpsig.shape)
    hamming1 = [hamming(frlen)]
    hamming1 = np.array(hamming1)

    print(hamming1.shape)

    # 最後に切り出し？？
    sig=tmpsig[frlen+1:(nf-1)*frsft+frlen]
    len = sig.shape[0]
    print(sig.shape)
    sig = sig.reshape(len)
    print("sig.shape",sig.shape)
    return sig

# -----スペクトログラムの表示-----------
# 入力
# tf:STFT変換後の２次元配列
# frsft:フレーム長
# sfeq:サンプリング周期
def tfplot(tf, frsft, sfrq):
    row = tf.shape[0]
    col = tf.shape[1]
    frlen = (row - 1)*2 # 波長の長さを確認
    #x = np.array(range(col)*frsft/sfrq)
    #y = np.array(range(col)*sfrq/frlen/1000.0)
    image_tf = np.empty_like(tf)
    print(tf.shape)
    
    for i in range(row):
        for j in range(col):
            image_tf[i][j] = 20* math.log10(abs(tf[i][j]))
    image_tf.clip(0, 255)
    image_tf = image_tf.astype('uint8')
    cv2.imshow("image",image_tf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_tf


#読み込むサンプルファイル
sample_wave_file="/Users/kenta/Programing/github/yama-pyroomacoustics-learning/python_source_separation-master/section2/CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0001.wav"
#ファイルを読み込む
wav=wave.open(sample_wave_file)
#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())
#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)
print("元のデータの大きさ",data.shape)

#短時間フーリエ変換を行う(ライブラリ)
f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
print("ライブラリでの大きさ",stft_data.shape)
#自作ライブラリでのSTFT
f,t,my_stft_data= STFT(data,wav.getframerate(),512,256,"hann")
print("自作STFTでの大きさ",my_stft_data.shape)

#print(stft_data,my_stft_data)
print(tfplot(stft_data,512,wav.getframerate()))

"""
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(stft_data)
ax2.plot(my_stft_data)
ax1.set_title('library STFT')
ax2.set_title('My STFT')
fig.tight_layout()
plt.show()
"""

"""
# 時間領域の波形に戻す
#t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
#print("data_post.shape:",data_post.shape)
data_post = invSTFT(stft_data,wav.getframerate(),512,256,"hann")
print("data_post.shape2:",data_post.shape)

# 2バイトのデータに変換
data_post=data_post.astype(np.int16)


# 変換前と復元後が一緒であることを確認する（二つ並べてみる）
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

# 変換前のスペクトグラム
print("data.shape",data.shape)
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
"""
"""
#dataを再生する
sd.play(data_post,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()

#waveファイルを閉じる
wav.close()
"""