import csiread
import numpy as np
from scipy import signal
import spectral_analysis as sa
import datetime
import math
# import scipy.linalg as la
import matplotlib.pyplot as plt

# Load CSI dataset
# csifile = './example/motion threshold/kitchen/kitchen_empty.dat' #"input_ellipse.dat"
csifile = "./Rectangle_1.dat"
csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=1)
csidata.read()
csi = csidata.get_scaled_csi()
csi = csi[:, :, :, 0]
# csi_trans = np.reshape(csi.transpose(0, 2, 1), (52365, 90))
csi_trans = np.reshape(csi.transpose(0, 2, 1), (np.shape(csi)[0], 90))

# Parameter definition
c = 299792458 * 100  # Light Speed (cm)
freq_center = 5.32e9  # Center Frequency
frequency_spacing = 312.5e3  # Frequency Spacing
subcarrier_index_L = [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1]
subcarrier_index_H = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]
subcarrier_index = np.concatenate((subcarrier_index_L, subcarrier_index_H), axis=0)  # Subcarrier Index
freq_sub = freq_center + subcarrier_index * frequency_spacing  # Subcarrier Frequency
Fs = 1000.0  # Sampling Frequency
win = 100  # In a window, 100 samples are used for estimation each time
inc = 100  # The number of CSI samples between two adjacent windows
refine_win = 10  # The estimated parameter in 30 windows are used for refinement
motion_threshold = 0  # A threshold that is used to determine if a moving person appears
# Pre-estimated Hardward-dependent Phase Difference and Antenna Spacing
phase_h_12 = 5.972  # angle(Hh_1 .* conj(Hh_2))
phase_h_23 = 1.496  # angle(Hh_2 .* conj(Hh_3))
phase_h_13 = 1.178  # angle(Hh_1 .* conj(Hh_3))
D_12 = 2.682  # Antenna Spacing between antenna 1 and 2 (cm)
D_23 = 2.251  # Antenna Spacing between antenna 2 and 3 (cm)
# Antenna Position (T, 3,2,1)
TR_Dist_1 = 235     # 127.0  # Distance of Transmitter to Receiver 1 (cm)
receiver_x_1 = 0.0  # x coordinate of receiver 1 (cm)
receiver_y_1 = 0.0  # y coordinate of receiver 1 (cm)
receiver_x_2 = D_12  # x coordinate of receiver 2 (cm)
receiver_y_2 = 0.0  # y coordinate of receiver 2 (cm)
receiver_x_3 = (D_12 + D_23)  # x coordinate of receiver 2 (cm)
receiver_y_3 = 0.0  # y coordinate of receiver 2 (cm)

# Initialize global variables
global_Motion_Threshold = []
global_DFS = []
global_U_12 = np.empty((0, 30), complex)
global_U_23 = global_U_12
global_U_31 = global_U_12
global_Hx_1 = np.empty((0, 30), complex)
global_Hx_2 = global_Hx_1
global_Hx_3 = global_Hx_1

flag_ini = 0

cachDoppler = []
cachMotion_Threshold = []
cachAoA = []
cachAoA_TX = []
cachDistance = []
cachKalDistance = []
cachPos_X = []
cachPos_Y = []

count_ii = 0
count_jj = 0
for ii in range(0, csi_trans.shape[0] - inc, inc):
    count_ii = count_ii + 1

    # starting time
    starttime = datetime.datetime.now()

    # Load CSI Data
    csi_1 = csi_trans[ii:ii + win,  0: 30]
    csi_2 = csi_trans[ii:ii + win, 30: 60]
    csi_3 = csi_trans[ii:ii + win, 60: 90]

    # Step 1: Conjugate multiplication between CSIs of any two antennas
    # CSI_12 = Hh_1.*conj(Hh_2).*Hs_1.*conj(Hs_2) +
    #          Hh_1.*conj(Hh_2).*( Hs_1.*conj(Hx_2)+Hx_1.*conj(Hs_2)+Hx_1.*conj(Hx_2) )
    CSI_12 = csi_1 * np.conj(csi_2)
    CSI_23 = csi_2 * np.conj(csi_3)
    CSI_31 = csi_3 * np.conj(csi_1)
    Con_filt_CSI = np.concatenate((CSI_12, CSI_23, CSI_31), 1)

    ## Step 2: Filtering
    # 2.1 Savitzky–Golay Filtering
    # Con_filt_CSI_real = signal.savgol_filter(np.real(Con_filt_CSI), window_length=5, polyorder=2, axis=0)  # Real Part
    # Con_filt_CSI_imag = signal.savgol_filter(np.imag(Con_filt_CSI), window_length=5, polyorder=2,
    #                                          axis=0)  # Imaginary  Part
    # Con_filt_CSI = Con_filt_CSI_real + 1j * Con_filt_CSI_imag
    # 2.2 LowPass Filtering
    b, a = signal.butter(5, 60, btype='lowpass', fs=Fs)
    Con_filt_CSI = signal.filtfilt(b, a, Con_filt_CSI, 0)

    filt_CSI_12 = Con_filt_CSI[:,  0:30]   # Filtered CSI
    filt_CSI_23 = Con_filt_CSI[:, 30:60]
    filt_CSI_31 = Con_filt_CSI[:, 60:90]

    # Step 3: Doppler Frequency Estimation
    # 3.1 Static Component
    # - U_12 = Hh_1. * conj(Hh_2). * Hs_1. * conj(Hs_2)
    # - U_23 = Hh_2. * conj(Hh_3). * Hs_2. * conj(Hs_3)
    # - U_31 = Hh_3. * conj(Hh_1). * Hs_3. * conj(Hs_1)
    U_12 = np.mean(filt_CSI_12, 0)
    U_23 = np.mean(filt_CSI_23, 0)
    U_31 = np.mean(filt_CSI_31, 0)
    global_U_12 = np.append(global_U_12, [U_12], axis=0)
    global_U_23 = np.append(global_U_23, [U_23], axis=0)
    global_U_31 = np.append(global_U_31, [U_31], axis=0)

    # 3.2 Dynamic Component
    # - V_12 = Hx_1. / Hs_1 + conj(Hx_2. / Hs_2),
    # - V_23 = Hx_2. / Hs_2 + conj(Hx_3. / Hs_3),
    # - V_31 = Hx_3. / Hs_3 + conj(Hx_1. / Hs_1)
    V_12 = (filt_CSI_12 - U_12) / U_12
    V_23 = (filt_CSI_23 - U_23) / U_23
    V_31 = (filt_CSI_31 - U_31) / U_31
    # 3.3 Conduct a transformation using estimated static and dynamic components to obatin
    # - diff_Hxs_12 = Hx_1. / Hs_1 - Hx_2. / Hs_2
    # - diff_Hxs_23 = Hx_2. / Hs_2 - Hx_3. / Hs_3
    # - diff_Hxs_31 = Hx_3. / Hs_3 - Hx_1. / Hs_1
    diff_Hxs_12 = np.conj(V_31) - V_23
    diff_Hxs_23 = np.conj(V_12) - V_31
    diff_Hxs_31 = np.conj(V_23) - V_12
    # 3.4 Doppler Estimation Using RootMUSIC
    # diff_Hxs = diff_Hxs_12 - diff_Hxs_23
    # diff_Hxs = np.sum(diff_Hxs / np.abs(diff_Hxs), 1)
    # Doppler = sa.root_MUSIC(diff_Hxs, 1, None, Fs)
    # global_DFS = np.append(global_DFS, Doppler)
    diff_Hxs = np.concatenate((diff_Hxs_12, diff_Hxs_23, diff_Hxs_31), axis=1)
    Doppler = sa.root_MUSIC(diff_Hxs.T, 1, None, Fs)
    global_DFS = np.append(global_DFS, Doppler)
    print('Doppler', Doppler)

    # Step 4: Human motion detection function based on Doppler Frequency
    # Motion_Threshold = np.abs(np.sum(np.exp(1j * np.angle(diff_Hxs)) *
    #                                  np.exp(-1j * 2 * np.pi * Doppler * np.arange(0, win) * 1 / Fs))) / win
    Motion_Threshold = np.sum(np.abs(np.sum(diff_Hxs.T / np.abs(diff_Hxs.T) *
                                            np.exp(-1j * 2 * np.pi * Doppler * np.arange(0, win) * 1 / Fs),
                                            axis=1))) / (90 * win)
    global_Motion_Threshold = np.append(global_Motion_Threshold, Motion_Threshold)

    # with open('Kitchen_move.txt', 'a') as f:
    #     f.write('%d  %f' % (count_ii, Motion_Threshold))
    #     f.write('\n')

    # Step 5: Separate Dynamic Component from Instantaneous Power of Each Antenna
    # 5.1 Calculate Instantaneous Power of Each Antenna
    # CSI_11 = abs(Hs_1) ^ 2 + 2 * abs(Hs_1). * abs(Hx_1). * cos(angle(Hs_1) - angle(Hx_1)) + abs(Hx_1) ^ 2)
    CSI_11 = np.real(csi_1 * np.conj(csi_1))
    CSI_22 = np.real(csi_2 * np.conj(csi_2))
    CSI_33 = np.real(csi_3 * np.conj(csi_3))
    Pow_filt_CSI = np.concatenate((CSI_11, CSI_22, CSI_33), 1)
    # 5.2 Filtering
    # Savitzky–Golay Filtering
    # Pow_filt_CSI = signal.savgol_filter(Pow_filt_CSI, window_length=5, polyorder=2, axis=0)
    # LowPass Filtering
    threshold_freq = 15  # 10 Hz threshold
    refine_L_b, refine_L_a = signal.butter(5, int(np.abs(Doppler)) + threshold_freq, btype='lowpass', fs=Fs)
    Pow_filt_CSI = signal.filtfilt(refine_L_b, refine_L_a, Pow_filt_CSI, 0)
    filt_CSI_11 = Pow_filt_CSI[:, 0:30]
    filt_CSI_22 = Pow_filt_CSI[:, 30:60]
    filt_CSI_33 = Pow_filt_CSI[:, 60:90]
    # 5.3 Separate Dynamic Component
    V_11 = (filt_CSI_11 - np.mean(filt_CSI_11, 0)) / (2 * np.sqrt(np.mean(filt_CSI_11, 0)))
    V_22 = (filt_CSI_22 - np.mean(filt_CSI_22, 0)) / (2 * np.sqrt(np.mean(filt_CSI_22, 0)))
    V_33 = (filt_CSI_33 - np.mean(filt_CSI_33, 0)) / (2 * np.sqrt(np.mean(filt_CSI_33, 0)))
    # HighPass Filtering
    # if np.abs(Doppler) > 16:
    #     refine_H_b, refine_H_a = signal.butter(5, int(np.abs(Doppler)) - threshold_freq, btype='highpass', fs=Fs)
    #     V_11 = signal.filtfilt(refine_H_b, refine_H_a, V_11, 0)
    #     V_22 = signal.filtfilt(refine_H_b, refine_H_a, V_22, 0)
    #     V_33 = signal.filtfilt(refine_H_b, refine_H_a, V_33, 0)

    # Step 6: Estimate Phase and Confidence at all sub-carriers
    # rhoxs_1. * cos(delta_thetaxs_1 + 2 * pi * doppler_1 * delta_tal) = V_1
    # rhoxs_1. * (cos(delta_thetaxs_1). * sin(2 * pi * doppler_1 * delta_tal) - cos(2 * pi * doppler_1 * delta_tal). * sin(delta_thetaxs_1)) = W_1
    # - phase_xs_1 = angle(Hx_1. / Hs_1)
    # - phase_xs_2 = angle(Hx_2. / Hs_2)
    # - phase_xs_3 = angle(Hx_3. / Hs_3)
    phase_x_1 = np.zeros(30)
    phase_x_2 = np.zeros(30)
    phase_x_3 = np.zeros(30)
    confidence_1 = np.zeros(30)
    confidence_2 = np.zeros(30)
    confidence_3 = np.zeros(30)
    Doppler_phase_spacing = 2 * np.pi * Doppler * np.arange(0, win) * 1 / Fs
    A = np.append([np.cos(Doppler_phase_spacing)], [-np.sin(Doppler_phase_spacing)], axis=0).T
    for kk in range(0, 30):
        # 1
        x_1 = np.linalg.lstsq(A, V_11[:, kk], rcond=None)
        confidence_1[kk] = np.power((np.pi / 2 - np.arctan(x_1[1] / win)) / (np.pi / 2), 3)
        phase_xs_1 = np.arctan2(x_1[0][1], x_1[0][0])
        phase_x_1[kk] = np.mod(phase_xs_1 - 2 * np.pi * freq_sub[kk] / c * TR_Dist_1, 2 * np.pi)

        # 2
        x_2 = np.linalg.lstsq(A, V_22[:, kk], rcond=None)
        confidence_2[kk] = np.power((np.pi / 2 - np.arctan(x_2[1] / win)) / (np.pi / 2), 3)
        phase_xs_2 = np.angle(np.exp(1j * np.arctan2(x_2[0][1], x_2[0][0])) * np.conj(U_12[kk]) * np.exp(-1j * phase_h_12))
        phase_x_2[kk] = np.mod(phase_xs_2 - 2 * np.pi * freq_sub[kk] / c * TR_Dist_1, 2 * np.pi)

        # 3
        x_3 = np.linalg.lstsq(A, V_33[:, kk], rcond=None)
        confidence_3[kk] = np.power((np.pi / 2 - np.arctan(x_3[1] / win)) / (np.pi / 2), 3)
        phase_xs_3 = np.angle(np.exp(1j * np.arctan2(x_3[0][1], x_3[0][0])) * U_31[kk] * np.exp(1j * phase_h_13))
        phase_x_3[kk] = np.mod(phase_xs_3 - 2 * np.pi * freq_sub[kk] / c * TR_Dist_1, 2 * np.pi)

    global_Hx_1 = np.append(global_Hx_1, [confidence_1 * np.exp(1j * phase_x_1)], axis=0)
    global_Hx_2 = np.append(global_Hx_2, [confidence_2 * np.exp(1j * phase_x_2)], axis=0)
    global_Hx_3 = np.append(global_Hx_3, [confidence_3 * np.exp(1j * phase_x_3)], axis=0)

    if len(global_DFS) < refine_win:
        print("Waiting...")
    else:
        ## Step 7: Determine the presence of human motion
        if np.median(global_Motion_Threshold) >= motion_threshold:
            flag_ini += 1
            print("Presence of Human Motion...")

            ## Step 8: AoA and Distance Estimation
            # Doppler Frequency Refinement
            filt_global_DFS = signal.savgol_filter(global_DFS, 5, 3)
            # filt_global_DFS = global_DFS
            # AoA
            est_AoA = np.arange(-90, 90, 1) * np.pi / 180
            # est_AoA = np.arange(-45, 45, 1) * np.pi / 180
            est_P_AoA = np.zeros((refine_win, len(est_AoA)), float)
            # Distance
            est_Dist = np.arange(np.round(TR_Dist_1) + 5, 1500, 5)
            est_P_Dist = np.zeros((refine_win, len(est_Dist)), float)
            for num_ii in range(0, refine_win):
                # AoA
                est_P_AoA[num_ii, :] = np.sum(np.abs(
                    np.matrix(global_Hx_1[num_ii, :]).T +
                    np.matrix(global_Hx_2[num_ii, :]).T * np.matrix(np.exp(1j * 2 * np.pi * freq_center / c * D_12 * np.sin(est_AoA))) +
                    np.matrix(global_Hx_3[num_ii, :]).T * np.matrix(np.exp(1j * 2 * np.pi * freq_center / c * (D_12 + D_23) * np.sin(est_AoA)))), axis=0)
                # Distance
                exp_Dist = np.exp(1j * 2 * np.pi * np.matrix(freq_sub).T / c * np.matrix(est_Dist)).T
                est_P_Dist[num_ii, :] = np.abs(np.sum(exp_Dist.A * np.matrix(global_Hx_1[num_ii, :]).A, axis=1)) + \
                                        np.abs(np.sum(exp_Dist.A * np.matrix(global_Hx_2[num_ii, :]).A, axis=1)) + \
                                        np.abs(np.sum(exp_Dist.A * np.matrix(global_Hx_3[num_ii, :]).A, axis=1))
                est_Dist = est_Dist + (1 * filt_global_DFS[num_ii] * c / freq_center * (win - 1) * 1 / Fs)
            # AoA
            P_AoA = np.sum(est_P_AoA, axis=0)
            # plt.clf()
            # plt.plot(P_AoA)
            # plt.pause(0.1)

            AoA = est_AoA[np.where(P_AoA == np.max(P_AoA))]
            AoA = AoA[0]

            # Distance
            P_Dist = np.sum(est_P_Dist, axis=0)
            est_Dist = np.arange(np.round(TR_Dist_1) + 5, 1500, 5)
            Distance = est_Dist[np.where(P_Dist == np.max(P_Dist))]
            Distance = Distance[0]

            ## Step 9: Distance Optimization using Kalman Filter
            if flag_ini == 1:
                # Kalman
                Kal_A = np.matrix([[1, win * refine_win / Fs],
                                   [0, 1]])  # 系统矩阵
                Kal_Q = np.matrix([[0.01, 0.0],
                                   [0.0, 0.01]])  # 过程噪声协方差
                Kal_R = np.matrix([[20, 0.0],
                                   [0.0, np.std(filt_global_DFS * c / freq_center, ddof=1)]])  # 测量噪声协方差
                Kal_H = np.matrix([[1.0, 0.0],
                                   [0.0, 1.0]])

                Kal_x_hat = np.matrix([[Distance],
                                       [np.median(filt_global_DFS) * c / freq_center]])  # 初始状态
                Kal_p = np.matrix([[1.0, 0.0],
                                   [0.0, 1.0]])  # 状态协方差矩阵
            else:
                Kal_z = np.matrix([[Distance],
                                   [np.median(filt_global_DFS) * c / freq_center]])  # 初始状态
                # 预测
                Kal_x_phat = Kal_A * Kal_x_hat
                Kal_p_p = Kal_A * Kal_p * Kal_A.T + Kal_Q

                # 校正
                Kal_k = Kal_p_p * Kal_H.T * (Kal_H * Kal_p_p * Kal_H.T + Kal_R) ** -1
                Kal_x_hat = Kal_x_phat + Kal_k * (Kal_z - Kal_H * Kal_x_phat)
                Kal_p = (np.mat(np.identity(2)) - Kal_k * Kal_H) * Kal_p_p
            Kal_Distance = Kal_x_hat.A[0][0]
            if Kal_Distance < TR_Dist_1:
                Kal_Distance = TR_Dist_1 + 1

            ## Step 10: Localization
            # est_AoA_TX = np.arange(-90, 90, 1) * np.pi / 180
            # sum_AoA_TX_12 = np.angle(np.sum(np.exp(1j * np.angle(global_U_12))) * np.exp(-1j * phase_h_12))
            # sum_AoA_TX_23 = np.angle(np.sum(np.exp(1j * np.angle(global_U_23))) * np.exp(-1j * phase_h_23))
            # sum_AoA_TX_13 = np.angle(np.sum(np.exp(1j * np.angle(np.conj(global_U_31)))) * np.exp(-1j * phase_h_13))
            #
            # est_P_AoA_TX = np.abs(np.exp(1j * sum_AoA_TX_12) * np.exp(1j * 2 * np.pi * freq_center / c * D_12 * np.sin(est_AoA_TX))-1)+\
            #                np.abs(np.exp(1j * sum_AoA_TX_23) * np.exp(1j * 2 * np.pi * freq_center / c * D_23 * np.sin(est_AoA_TX))-1)+\
            #                np.abs(np.exp(1j * sum_AoA_TX_13) * np.exp(1j * 2 * np.pi * freq_center / c * (D_12+D_23) * np.sin(est_AoA_TX))-1)
            # AoA_TX = est_AoA_TX[np.where(est_P_AoA_TX == np.min(est_P_AoA_TX))]
            est_AoA_TX = np.arange(-90, 90, 1) * np.pi / 180
            sum_AoA_TX = np.angle(np.sum(global_U_12 * np.exp(-1j * phase_h_12) + \
                                         global_U_23 * np.exp(-1j * phase_h_23)))
            est_P_AoA_TX = np.abs(np.exp(1j * sum_AoA_TX) *
                                  np.exp(-1j * 2 * np.pi * freq_center / c * (D_12 + D_23) / 2 * np.sin(est_AoA_TX)) - 1)
            Real_AoA_TX = est_AoA_TX[np.where(est_P_AoA_TX == np.min(est_P_AoA_TX))]
            AoA_TX = Real_AoA_TX

            print('AoA_TX', AoA_TX)

            alpha = (AoA - AoA_TX)
            TPR_Dist = Distance
            PR_Dist = (math.pow(TPR_Dist, 2) - math.pow(TR_Dist_1, 2)) / (2 * (TPR_Dist - TR_Dist_1 * np.cos(alpha)))

            Pos_X = PR_Dist * np.sin(AoA)
            Pos_Y = PR_Dist * np.cos(AoA)

            # with open('LOS_input_31.txt', 'a') as f:
            #     f.write('%d  %f  %f' % (count_ii, Pos_X, Pos_Y))
            #     f.write('\n')

            # Display
            cachDoppler.append(np.median(filt_global_DFS))
            cachMotion_Threshold.append(np.median(global_Motion_Threshold))
            cachDistance.append(Distance)
            cachKalDistance.append(Kal_Distance)
            cachAoA.append(AoA)
            cachAoA_TX.append(AoA_TX)
            cachPos_X.append(Pos_X)
            cachPos_Y.append(Pos_Y)
        else:
            print("Absence of Human Motion...")

        # Output
        # print('Position_X: %f, Position_Y:%f' % (Pos_X, Pos_Y))
        print('AoA:%f' % AoA)
        print('Distance:%f' % Distance)

        global_DFS = np.delete(np.array(global_DFS), 0)
        global_Motion_Threshold = np.delete(global_Motion_Threshold, 0, 0)
        global_Hx_1 = np.delete(global_Hx_1, 0, 0)
        global_Hx_2 = np.delete(global_Hx_2, 0, 0)
        global_Hx_3 = np.delete(global_Hx_3, 0, 0)
        global_U_12 = np.delete(global_U_12, 0, 0)
        global_U_23 = np.delete(global_U_23, 0, 0)
        global_U_31 = np.delete(global_U_31, 0, 0)
    # ending time
    endtime = datetime.datetime.now()
    print('Running Time (s): %f' % ((endtime - starttime).microseconds / math.pow(10, 6)))



cachAoA = np.array(cachAoA)
plt.plot(cachAoA)
plt.figure()
plt.plot(cachKalDistance)
