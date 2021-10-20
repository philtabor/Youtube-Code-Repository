import numpy as np
import matplotlib.pyplot as plt
cer_1k = np.load('CER_const_eps_1000.npy')
cer_10k = np.load('CER_const_eps_10000.npy')
cer_100k = np.load('CER_const_eps_100000.npy')

ver_1k = np.load('VER_const_eps_1000.npy')
ver_10k = np.load('VER_const_eps_10000.npy')
ver_100k = np.load('VER_const_eps_100000.npy')

running_cer1k_avg = np.zeros(len(cer_1k))
running_cer10k_avg = np.zeros(len(cer_10k))
running_cer100k_avg = np.zeros(len(cer_100k))
running_ver1k_avg = np.zeros(len(ver_1k))
running_ver10k_avg = np.zeros(len(ver_10k))
running_ver100k_avg = np.zeros(len(ver_100k))

for i in range(len(cer_1k)):
    running_cer1k_avg[i] = np.mean(cer_1k[max(0, i-100):(i+1)])
    running_cer10k_avg[i] = np.mean(cer_10k[max(0, i-100):(i+1)])
    running_cer100k_avg[i] = np.mean(cer_100k[max(0, i-100):(i+1)])
    running_ver1k_avg[i] = np.mean(ver_1k[max(0, i-100):(i+1)])
    running_ver10k_avg[i] = np.mean(ver_10k[max(0, i-100):(i+1)])
    running_ver100k_avg[i] = np.mean(ver_100k[max(0, i-100):(i+1)])


x_axis = np.arange(len(cer_1k))
plt.plot(x_axis, running_cer1k_avg, 'r--', label='CER (1,000)')
plt.plot(x_axis, running_ver1k_avg, 'b--', label='VER (1,000)')
plt.xlabel('Episode')
plt.ylabel('Avg Score')
plt.legend(loc='lower right')
plt.savefig('CER_vs_VER_1000_const_eps.png')
plt.close()

x_axis = np.arange(len(cer_10k))
plt.plot(x_axis, running_cer10k_avg, 'r--', label='CER (10,000)')
plt.plot(x_axis, running_ver10k_avg, 'b--', label='VER (10,000)')
plt.xlabel('Episode')
plt.ylabel('Avg Score')
plt.legend(loc='lower right')
plt.savefig('CER_vs_VER_10000_const_eps.png')
plt.close()

x_axis = np.arange(len(cer_100k))
plt.plot(x_axis, running_cer100k_avg, 'r--', label='CER (100,000)')
plt.plot(x_axis, running_ver100k_avg, 'b--', label='VER (100,000)')
plt.xlabel('Episode')
plt.ylabel('Avg Score')
plt.legend(loc='lower right')
plt.savefig('CER_vs_VER_100000_const_eps.png')
plt.close()
