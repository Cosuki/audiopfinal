import numpy as np

import os
import IPython
os.environ['NUMBA_CACHE_DIR'] = IPython.paths.get_ipython_cache_dir()
import librosa

import synctoolbox

from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic

import libfmp.c3
import libtsm
import scipy.io.wavfile

import analysis_functions as analysis

def synchronize_audios(X, Y, Fs, apply_pitch_shift=True, apply_timbre_adaptation=True):
    #estimamos desviación en frecuencia de cada una de las canciones
    tuning_offset_1 = librosa.estimate_tuning(y=X, sr=Fs)
    tuning_offset_2 = librosa.estimate_tuning(y=Y, sr=Fs)

    #estimamos la diferencia de altura de los cromagramas antes de hacer la correspondencia
    N = 2048
    H = 4096
    X_chroma = librosa.feature.chroma_stft(y=X, sr=Fs, norm=2, hop_length=H, n_fft=N, tuning=tuning_offset_1)
    X_chroma = X_chroma / X_chroma.max()
    Y_chroma = librosa.feature.chroma_stft(y=Y, sr=Fs, norm=2, hop_length=H, n_fft=N, tuning=tuning_offset_2)
    Y_chroma = Y_chroma / Y_chroma.max()

    opt_chroma_shift = compute_optimal_chroma_shift(X_chroma, Y_chroma)

    #buscamos la correspondencia temporal entre las canciones con DTW aplicado a los cromagramas
    N = 2048
    H = int(0.02*Fs)
    X_chroma = librosa.feature.chroma_stft(y=X, sr=Fs, norm=2, hop_length=H, n_fft=N, tuning=tuning_offset_1)
    X_chroma = X_chroma / X_chroma.max()
    Y_chroma = librosa.feature.chroma_stft(y=Y, sr=Fs, norm=2, hop_length=H, n_fft=N, tuning=tuning_offset_2)
    Y_chroma = Y_chroma / Y_chroma.max()

    Y_chroma = shift_chroma_vectors(Y_chroma, opt_chroma_shift)

    C = libfmp.c3.compute_cost_matrix(X_chroma, Y_chroma)
    D = libfmp.c3.compute_accumulated_cost_matrix(C)
    P = libfmp.c3.compute_optimal_warping_path(D)

    wp = make_path_strictly_monotonic(P.T)

    # Adaptamos vector de correspondencias
    time_map = wp.T * H
    time_map = np.concatenate((time_map, np.array([[len(X)-1,len(Y)-1]])))
    time_map = libtsm.ensure_validity(time_map)

    # Aplicamos las correspondencias halladas a una de las señales, ajustando pitch y timbre si corresponde
    if apply_pitch_shift:
        pitch_shift_for_audio_1 = -opt_chroma_shift % 12
        if pitch_shift_for_audio_1 > 6:
            pitch_shift_for_audio_1 -= 12
        audio_1_shifted = libtsm.pitch_shift(X, pitch_shift_for_audio_1 * 100, order="tsm-res")

        if apply_timbre_adaptation:
            L = 2048
            R = 256
            X_stft, X_env, X_exc, _, _ = analysis.analysis_STFT_LPC(X, Fs, L, R)
            Xs_stft, Xs_env, Xs_exc, _, _ = analysis.analysis_STFT_LPC(audio_1_shifted[:,0], Fs, L, R)
            Xm_stft = Xs_exc*X_env
            xm = analysis.synthesis_STFT(Xm_stft, L, R)
            y_hpstsm = libtsm.hps_tsm(xm, time_map) 
        
        else:
            y_hpstsm = libtsm.hps_tsm(audio_1_shifted, time_map)
    
    else:
        y_hpstsm = libtsm.hps_tsm(X, time_map)

    # Unimos la señal modificada con la correspondiente original
    stereo_sonification = np.hstack((Y.reshape(-1, 1), y_hpstsm))

    return stereo_sonification.T

if __name__ == '__main__':
    audio1, Fs = librosa.load('data/Emily_Linge-vocals.wav')
    audio2, Fs = librosa.load('data/Police-vocals-guitar.wav')
    sync_audios = synchronize_audios(audio1, audio2, Fs, apply_pitch_shift=True, apply_timbre_adaptation=True)
    scipy.io.wavfile.write("results/Emily_Police_synchronized.wav", Fs, sync_audios)