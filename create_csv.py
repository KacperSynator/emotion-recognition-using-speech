import glob
import pandas as pd
import os


def write_emodb_csv(emotions=["sad", "neutral", "happy"], train_name="train_emo.csv",
                    test_name="test_emo.csv", train_size=0.8, verbose=1):
    """
    Reads speech emodb dataset from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_emo.csv'
        test_name (str): the output csv filename for testing data, default is 'test_emo.csv'
        train_size (float): the ratio of splitting training data, default is 0.8 (80% Training data and 20% testing data)
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }
    # delete not specified emotions
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_dir, "data", "emodb", "wav")
    for file in glob.glob(f"{path}/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
    if verbose:
        print("[EMO-DB] Total files to write:", len(target['path']))
        
    # dividing training/testing sets
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[EMO-DB] Training samples:", train_size)
        print("[EMO-DB] Testing samples:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy"], train_name="train_tess_ravdess.csv",
                            test_name="test_tess_ravdess.csv", verbose=1):
    """
    Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_tess_ravdess.csv'
        test_name (str): the output csv filename for testing data, default is 'test_tess_ravdess.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # for training speech directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(script_dir, "data", "training")
        total_files = glob.glob(f"{path}/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} training audio files for category:{category}")
    
        # for validation speech directory
        path = os.path.join(script_dir, "data", "validation")
        total_files = glob.glob(f"{path}/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} testing audio files for category:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


def write_custom_csv(emotions=['sad', 'neutral', 'happy'], train_name="train_custom.csv", test_name="test_custom.csv",
                    verbose=1):
    """
    Reads Custom Audio data from data/*-custom and then writes description files (csv)
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_custom.csv'
        test_name (str): the output csv filename for testing data, default is 'test_custom.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    for category in emotions:
        # train data
        script_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(script_dir, "data", "train-custom")
        for i, file in enumerate(glob.glob(f"{path}/*_{category}.wav")):
            train_target["path"].append(file)
            train_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} training audio files for category:{category}")
            except NameError:
                # in case {i} doesn't exist
                pass
        
        # test data
        path = os.path.join(script_dir, "data", "test-custom")
        for i, file in enumerate(glob.glob(f"{path}/*_{category}.wav")):
            test_target["path"].append(file)
            test_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} testing audio files for category:{category}")
            except NameError:
                pass
    
    # write CSVs
    if train_target["path"]:
        pd.DataFrame(train_target).to_csv(train_name)

    if test_target["path"]:
        pd.DataFrame(test_target).to_csv(test_name)