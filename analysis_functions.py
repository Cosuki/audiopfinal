import numpy as np
from scipy import signal
import scipy.fft
from scipy.linalg import toeplitz, solve

def lpc_analysis(s, p=20):
    """ compute the LPC analysis using the autocorrelation method
    
    Parameters
    ----------
    x : numpy array
        windowed signal frame as a numpy 1D array.
    p : int
        model order.
        
    Returns
    -------
    ak : numpy array
         model coefficients.
    e : float
        minimum mean squared error.
    e_norm : float
             normalized minimum mean squared error.
    """
    # frame length
    N = s.shape[0]
    
    # compute autocorrelation values
    r = np.zeros((p+1, 1))
    for k in range(p+1):
        r[k] = np.dot(s[:N-k].T, s[k:])

    # solve to compute model coefficients
    ak = solve(toeplitz(r[:p]), r[1:]).squeeze()

    # compute mean squared error
    e = r[0] - np.dot(ak.T, r[1:])

    # compute normalized mean squared error
    e_norm = e / r[0]

    return ak, e, e_norm

def lpc_decomposition(s_win, ak, e, fs, Ndft, Nw):
    """ obtener envolvente y excitación de una señal a partir de los coeficientes de LPC
    
    Parameters
    ----------
    s_win : numpy array
        windowed signal frame as a numpy 1D array.
    ak : numpy array
        coeficientes del LPC
    e : float
        error mínimo obtenido del LPC
    fs : int
        frecuencia de muestreo de la señal
    Ndft : int
        número de puntos de la DFT
    Nw : int
        largo de la señal enventanada
        
    Returns
    -------
    H : numpy array
        envolvente de la señal
    P : numpy array
        excitación de la señal
    """
    # filter obtained from the lpc analysis
    S = 1
    U = np.concatenate([[1], -ak])

    # compute gain 
    G = np.sqrt(e)

    # compute the frequency response of the digital filter
    w, H = signal.freqz(G*S, U, worN=Ndft, whole=True)
    fw = w / (2 * np.pi) * fs

    # impulse response of the LPC filter
    delta = np.zeros(Nw)
    delta[0] = 1
    h = signal.lfilter(G*S, U, delta)

    # magnitude spectrum
    magH = np.abs(H)
    ind_fmx = int(Ndft/2)

    # inverse filter
    A = S*G
    B = U

    # compute the excitation from the inverse filter
    p = signal.lfilter(B, A, s_win)

    # compute the spectrum of the excitation
    P = np.fft.fft(p, Ndft)

    return H, P

def analysis_STFT_LPC(x, fs, L=2048, R=256, win='hann'):
    """ compute the analysis phase of the phase vocoder, i.e. the STFT of the input audio signal
    
    Parameters
    ----------
    x : numpy array
        input audio signal (mono) as a numpy 1D array.
    L : int
        window length in samples.
    R : int
        hop size in samples.
    win : string
          window type as defined in scipy.signal.windows.    
        
    Returns
    -------
    X_stft : numpy array
             STFT of x as a numpy 2D array.
    X_env : numpy array
        envolvente de la señal
    X_exc : numpy array
        excitación de la señal
    omega_stft : numpy array
                 frequency values in radians.
    samps_stft : numpy array
                 time sample at the begining of each frame.

    """
    
    # length of the input signal
    M = x.size;      
    
    # number of points to compute the DFT (FFT)
    N = L
    
    # analysis window
    window = signal.windows.get_window(win, L)
   
    # total number of analysis frames
    num_frames = int(np.floor((M - L) / R))

    # initialize stft
    X_stft = np.zeros((N, num_frames), dtype = complex)
    X_env = np.zeros((N, num_frames), dtype = complex)
    X_exc = np.zeros((N, num_frames), dtype = complex)
    
    # process each frame
    for ind in range(num_frames):

        # initial and ending points of the frame
        n_ini = int(ind * R)
        n_end = n_ini + L

        # signal frame
        xr = window*x[n_ini:n_end]

        # save DFT of the signal frame
        X_stft[:, ind] = scipy.fft.fft(xr, N)

        # LPC
        if np.max(abs(xr))>1e-8:
            ak, e, _ = lpc_analysis(xr, p=20)
            X_env[:, ind], X_exc[:, ind] = lpc_decomposition(xr, ak, e, fs, N, N)
        
    # frequency values in radians    
    omega_stft = 2*np.pi*np.arange(N)/N

    # time sample at the center of each frame
    samps_stft = np.arange(L/2, M-L/2, R)[:-1]
 
    return X_stft, X_env, X_exc, omega_stft, samps_stft

def synthesis_STFT(X_stft, L=2048, R=256, win='hann'):
    """ compute the synthesis using the IFFT of each frame combined with overlap-add
    
    Parameters
    ----------
    X_stft : numpy array
             STFT of x as a numpy 2D array.
    L : int
        window length in samples.
    R : int
        hop size in samples.
    win : string
          window type as defined in scipy.signal.windows.    
        
    Returns
    -------
    y : numpy array
        output audio signal (mono) as a numpy 1D array.
        
    """
    
    # number of frequency bins
    N = X_stft.shape[0];      
 
    # analysis window
    window = signal.windows.get_window(win, L)
   
    # total number of analysis frames
    num_frames = X_stft.shape[1]

    # initialize otuput signal in the time domain
    y = np.zeros(num_frames * R + L)
    
    # process each frame
    for ind in range(num_frames):

        # reconstructed signal frame
        yr = scipy.fft.ifft(X_stft[:,ind], L).real

        # initial and ending points of the frame
        n_ini = ind*R
        n_end = ind*R + L

        # overlap-add the signal frame
        y[n_ini:n_end] += window*yr
        
    # compute the amplitude scaling factor
    C = (L/2)/R
    
    # compensate the amplitude scaling factor
    y /= C
    
    return y