from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
def hamming(nperseg):
    # sp.hamming(N)
    t = np.ones((nperseg),dtype='float64')
    #print(t)
    for i in range(len(t)):
        #print(i+1)
        #print((math.pi/(i+1)))
        t[i] = 0.54-0.46*math.cos(2.0*math.pi/((nperseg*(i+1))))
    return t

print(hamming(512))

N = 512            # サンプル数
dt = 0.01          # サンプリング間隔
t = np.arange(0, N * dt, dt)  # 時間軸

# 窓関数の一例
fw1 = signal.hann(N)             # ハニング窓
fw2 = signal.hamming(N)          # ハミング窓
#fw2 = hamming(N)
fw3 = signal.blackman(N)         # ブラックマン窓
print(fw2)
# グラフ表示
fig = plt.figure(figsize=(7.0, 5.0))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 時間信号（元）
plt.plot(t, fw1, label='hann')
plt.plot(t, fw2, label='hamming')
plt.plot(t, fw3, label='blackman')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid()
leg = plt.legend(loc=1, fontsize=15)
plt.show()