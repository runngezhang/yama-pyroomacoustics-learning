#fft.py
import numpy as np
import matplotlib.pyplot as plt

N = 512           # サンプル数
dt = 0.001          # サンプリング周期 [s]
f1, f2 = 50, 120    # 周波数 [Hz]

t = np.arange(0, N*dt, dt) # 時間 [s]
x = 1.5*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 3 # 信号
print("x",x)
"""
fig, ax = plt.subplots()
ax.plot(t, x)
# ax.set_xlim(0, 0.1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.grid()
plt.show()
"""
F = np.fft.fft(x) # 変換結果
freq = np.fft.fftfreq(N, d=dt) # 周波数
print(F.shape) #Imaginary part"
print(freq.shape) #Frequency
"""
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()
"""