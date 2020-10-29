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
    nf = int(scipy.ceil((len(sig)-frlen)/frsft)+1) # データサイズをフレームサイズで割っている
    #print("スライスの大きさ",(sig[0:frlen]).shape)
    # ハミング窓を計算するために転地する
    hamming_trans = (hamming(frlen)).T

    #print("転地した後",hamming_trans.shape)
    print("nf",nf)
    stft_data = []
    for i in range(1,nf): # iの関係で1から215まで
        st=(i-1) * frsft # ひとつ前の場所から窓をかける
        # スライス処理と窓関数のかけ算
        #print(sig[st:st+frlen].shape,(hamming(frlen).T).shape)
        # 窓関数をかけるFFTを行う
        data_stft_split = np.fft.fft((sig[st:st+frlen].T*hamming(frlen)))
        #print(a)
        stft_data.append(data_stft_split)

    stft_data = np.array(stft_data)
    #print("stft_data.shape",stft_data.shape)
    return fs,nf,stft_data

# invSTFT: inverse Short-time Fourier Transform
# [inputs]
#     tf: STFT of input signal (frlen/2+1 x nf)
# length: length of output signal
#  frsft: frame shift  (default: frlen/2)
#    wnd: synthesis window function (frlen x 1, default: optimum window for hamming analysis window)
# [outputs]
#    sig: input signal (length x 1)

def invSTFT(stft_data,fs,frlen,frsft,wnd):
    # t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
    nf = stft_data.shape[1] # 分割したフレームの数
    #print(nf)
    subspc = np.empty((frlen,1),dtype = "float64")
    tmpsig = np.zeros(((nf-1)*frsft+frlen,1))
    for i in range(nf):
        st = (i*frsft)
        #print(subspc.shape,stft_data.shape)
        subspc[0:int(frlen/2)+1,0]=stft_data[:,i]
        #print(subspc.shape)
        subspc[0,0]=subspc[0,0]/2
        subspc[int(frlen/2)+1,0]=subspc[int(frlen/2)+1,0]/2

        # iFFTして出力配列に代入
        print(subspc.shape)
        a = np.fft.ifft(subspc)
        #print("a",a.shape)
        #print(a)
        #print("hamming",(hamming(frlen).shape))
        b = np.fft.ifft(subspc)*(hamming(frlen))
        #print(hamming(frlen))
        #print("b",b.shape)
        #print(((np.fft.ifft(subspc)*(hamming(frlen)).T).real*2).shape)
        #print(tmpsig[st+0:st+frlen+1,0].shape,((np.fft.ifft(subspc)).real*2).shape)
        tmpsig[st:st+frlen] = tmpsig[st:st+frlen]+(np.fft.ifft(subspc)).real*2
    print(tmpsig.shape)

    return stft_data


#読み込むサンプルファイル
sample_wave_file="/Users/kenta/Programing/github/yama-pyroomacoustics-learning/python_source_separation-master/section2/CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0001.wav"
#ファイルを読み込む
wav=wave.open(sample_wave_file)
#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())
#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)
print("元のデータの大きさ",data.shape)
#短時間フーリエ変換を行う
#f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
#print(f,t)
f,t,stft_data= STFT(data,wav.getframerate(),512,256,"hann")
#print(stft_data)
print("STFT後の大きさ",stft_data.shape)
#特定の周波数成分を消す(100番目の周波数よりも高い周波数成分を全て消す)
stft_data[100:,:]=0

# 時間領域の波形に戻す
#t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)
data_post = invSTFT(stft_data,wav.getframerate(),512,256,"hann")
# 2バイトのデータに変換
data_post=data_post.astype(np.int16)

"""
#dataを再生する
sd.play(data_post,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()

#waveファイルを閉じる
wav.close()
"""
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

