from scipy.io import wavfile
from scipy.fftpack import fft, ifft, rfft
from scipy import signal
import numpy as np
import argparse


def analysis(wav_data, N, hop_a=None):
    assert (len(wav_data.shape) == 1)
    if hop_a is None:
        hop_a = N // 4
    # padding data to ensure seamless fft computation
    pad_len = hop_a - (wav_data.shape[0] - N) % hop_a
    wav_data = np.pad(wav_data, (0, pad_len), mode='constant')
    hann_window = signal.hann(N)

    new_shape = (N, 1 + (wav_data.shape[0] - N) // hop_a)
    spectra = np.zeros(new_shape, dtype=np.complex128)
    for i in range(new_shape[1]):
        windowed_frame = wav_data[hop_a * i:hop_a * i + N] * hann_window
        spectra[:, i] = fft(windowed_frame)  # Eq 3.1
    return spectra


def processing(spectra, sampling_freq, semitones=0, hop_a=None):
    N = spectra.shape[0]
    if hop_a is None:
        hop_a = N // 4
    # not sure about this one
    bin_freq = np.fft.fftfreq(N, 1 / (2 * np.pi))
    dt_a = hop_a / sampling_freq  # pre Eq. 3.2
    scaling_factor = 2 ** (semitones / 12)
    hop_s = int(scaling_factor * hop_a)
    dt_s = hop_s / sampling_freq

    phases = np.zeros_like(spectra, dtype=np.float64)
    phases[:, 0] = np.angle(spectra[:, 0])
    for i in range(1, spectra.shape[1]):
        dphase_a = np.angle(spectra[:, i]) - np.angle(spectra[:, i - 1])
        freq_deviation = dphase_a / dt_a - bin_freq  # Eq. 3.3
        wrapped_freq_deviation = np.mod(freq_deviation + np.pi, 2 * np.pi) - np.pi  # Eq. 3.4
        true_freq = bin_freq + wrapped_freq_deviation  # Eq. 3.5

        phases[:, i] = phases[:, i - 1] + dt_s * true_freq  # Eq. 3.6

    new_spectra = np.abs(spectra) * np.exp(1j * phases)
    return new_spectra, hop_s


def synthesis(new_spectra, hop_s):
    N = new_spectra.shape[0]
    L = new_spectra.shape[1]
    hann_window = signal.hann(N)

    waves = np.zeros_like(new_spectra, dtype=np.float64)
    for i in range(L):
        waves[:, i] = ifft(new_spectra[:, i]).real * hann_window  # Eq. 3.8

    stretched_wave = np.zeros(N + (L - 1) * hop_s)
    for i in range(L):
        stretched_wave[i * hop_s:i * hop_s + N] += waves[:, i]
    return stretched_wave

def parse_args():
    parser = argparse.ArgumentParser(description='phase_vocoder')
    parser.add_argument('input_path', help='input wav file path')
    parser.add_argument('output_path', help='output wav file path')
    parser.add_argument('semitones', default=3, type=float, help='pitch shift in semitones')
    parser.add_argument('--window', default=1024, type=int, help='fft window size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # reading data
    args = parse_args()
    fs, data = wavfile.read(args.input_path)
    N = args.window
    hop_a = N // 4
    semitones = args.semitones

    # phase vocoder algorithm
    spectra = analysis(data, N, hop_a)
    new_spectra, hop_s = processing(spectra, fs, semitones)
    stretched_wave = synthesis(new_spectra, hop_s)

    # resampling
    xp = np.arange(len(stretched_wave)) + 1
    x = np.linspace(0, len(stretched_wave), len(data) + 1)[1:]
    result_wave = np.interp(x, xp, stretched_wave)

    # writing data
    wavfile.write(args.output_path, fs, result_wave.astype(np.int16))

