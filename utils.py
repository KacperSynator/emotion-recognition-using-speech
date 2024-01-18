import soundfile
import librosa
import numpy as np
import pickle
import os
from convert_wavs import convert_audio


AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", # pleasant surprised
    "boredom"
}


def get_label(audio_config):
    """Returns label corresponding to which features are to be extracted
        e.g:
    audio_config = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    print(f"Extracting features from {file_name}...")
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            print("Sound file opened")
            pass
    except RuntimeError:
        # not properly formated, convert to 16000 sample rate & mono channel using ffmpeg
        # get the basename
        print("Converting audio file to 16000 sample rate & mono channel using ffmpeg...")
        basename = os.path.basename(file_name)
        print(basename)
        dirname  = os.path.dirname(file_name)
        print(dirname)
        name, ext = os.path.splitext(basename)
        print(name)
        new_basename = f"{name}_c.wav"
        print(new_basename)
        new_filename = os.path.join(dirname, new_basename)
        print(new_filename)
        v = convert_audio(file_name, new_filename)
        print(v)
        if v:
            raise NotImplementedError("Converting the audio files failed, make sure `ffmpeg` is installed in your machine and added to PATH.")
    else:
        print("Sound file is already in 16000 sample rate & mono channel")
        new_filename = file_name
    with soundfile.SoundFile(new_filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        print(f"Sample rate: {sample_rate}")
        if chroma or contrast:
            print("Converting to mono channel...")
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            print("Extracting MFCC...")
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            print("Extracting Chroma...")
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            print("Extracting MEL...")
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            print("Extracting Contrast...")
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            print("Extracting Tonnetz...")
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            print(tonnetz.shape)
            result = np.hstack((result, tonnetz))
            print(result.shape)
    return result


def get_best_estimators(classification):
    """
    Loads the estimators that are pickled in `grid` folder
    Note that if you want to use different or more estimators,
    you can fine tune the parameters in `grid_search.py` script
    and run it again ( may take hours )
    """
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config
    