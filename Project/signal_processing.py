import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import librosa.effects
import scipy.signal
from pylab import rcParams
import threading

from Project.image_preprocessing import traverse_images

SOUND_PATH = "../../Datasets/GTZAN/genres_original/"
MEL_PATH = "../../Datasets/GTZAN/mel_spectrograms/"


def aug_mel_spect(y_vals, win_len, hop_len, sample_rate, path):
    dir_name = path.split("/")[-2]
    y_vals_up = librosa.effects.pitch_shift(y_vals, sample_rate, n_steps=2.0)
    y_vals_down = librosa.effects.pitch_shift(y_vals, sample_rate, n_steps=-2.0)
    mel_spectrogram(y_vals_up, win_len, hop_len, sample_rate,
                    path.replace("/" + dir_name + "/", "/" + dir_name + "/up" + "/").replace(".jpg", "_up.jpg"))
    mel_spectrogram(y_vals_down, win_len, hop_len, sample_rate,
                    path.replace("/" + dir_name + "/", "/" + dir_name + "/down" + "/").replace(".jpg", "_down.jpg"))


# lengths are in ms
def mel_spectrogram(y_vals, win_len, hop_len, sample_rate, path):
    rcParams['figure.figsize'] = 6, 3

    samples_per_ms = sample_rate // 1000
    win_len *= samples_per_ms
    hop_len *= samples_per_ms

    powers = librosa.feature.melspectrogram(
        y_vals - np.mean(y_vals),
        sr=sample_rate,
        n_mels=128,
        n_fft=8192,
        window='hann',
        win_length=win_len,
        hop_length=hop_len)

    decibels = librosa.power_to_db(powers, ref=np.max)

    librosa.display.specshow(decibels, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_len)
    # plt.colorbar(format='%+2.0f dB')
    plt.xlabel("")
    plt.ylabel("")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.ylim([0, sample_rate / 2])
    plt.gca().set_axis_off()
    # plt.show(bbox_inches='tight', pad_inches=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.cla()


def generate_mel_spect(path_to_sound):
    data, sample_rate = librosa.load(path_to_sound)
    mel_spectrogram(data, win_len=10, hop_len=5, sample_rate=sample_rate,
                    path=path_to_sound.replace("genres_original", "mel_spectrograms").replace(".wav", ".jpg"))


def generate_mel_spect_aug(path_to_sound):
    data, sample_rate = librosa.load(path_to_sound)
    aug_mel_spect(data, win_len=10, hop_len=5, sample_rate=sample_rate,
                  path=path_to_sound.replace("genres_original", "mel_spectrograms_aug").replace(".wav", ".jpg"))


def generate_mel_spects():
    '''
    t_hiphop = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "hiphop"))
    t_jazz = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "jazz"))
    t_metal = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "metal"))
    t_pop = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "pop"))
    t_reggae = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "reggae"))
    t_rock = threading.Thread(target=traverse_images, args=(SOUND_PATH, generate_mel_spect, "rock"))

    t_hiphop.start()
    t_jazz.start()
    t_metal.start()
    t_pop.start()
    t_reggae.start()
    t_rock.start()

    t_hiphop.join()
    t_jazz.join()
    t_metal.join()
    t_pop.join()
    t_reggae.join()
    t_rock.join()

    print("Joined all threads!")
    '''

    traverse_images(SOUND_PATH, generate_mel_spect_aug)


def do_stuff():
    # data, sample_rate = load_sound()
    # win_len = 10
    # hop_len = 5
    # freqs, times, db = stft_spectrogram(data, win_len, hop_len, sample_rate)
    # mel_spectrogram(data, win_len, hop_len, sample_rate)
    # plotSpectrogram(times, freqs, db, "speccc")

    # TODO: SPECIFY SIZE TO GET RIGHT IN PIXELS? THEN GENERATE ALL AND PROFIT <3
    # Then try to see if classification is better, then try augmentation with tone shifting
    pass


if __name__ == '__main__':
    generate_mel_spects()
