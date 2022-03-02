import librosa
import datasets
import pandas as pd
import numpy as np
import datasets
from typing import Tuple

def prepare_data(
        csv: str,
        data_path: str,
        class_num: int,
        sample_rate: int,
        num_workers: int) -> Tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]:
    """
    Preloads the audio files.
    The audio files are loaded and saved in disk to accelerate training.
    ----
    Args:
        preloaded_data_path:
            Path to the directory where the preloaded dataset will be saved.
        data_path:
            Path to audio files.
        class_num:
            Number of classes.
        sample_rate:
            Desired sample rate.
        num_workers:
            Number of processes (Multiprocessing).
    Returns:
        Huggingface datasets
    """
    datasets.config.IN_MEMORY_MAX_SIZE = 16*(1024**3)
    dataset = datasets.load_dataset('csv', data_files=csv, split='train[:15%]')
    dataset = dataset.map(
        audio_file_to_array,
        fn_kwargs={ "data_path": data_path,
                    "class_num": class_num,
                    "sample_rate": sample_rate},
        num_proc=num_workers
    )
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset['train'], dataset['test']

def audio_file_to_array(batch: datasets.arrow_dataset.Dataset,
                            data_path: str,
                            class_num: int,
                            sample_rate: int) -> datasets.arrow_dataset.Dataset:
    """
    Loads the audios from memory.
    The audios are loaded from disk and saved in a format that will speed up training afterwards.
    ----
    Args:
        batch:
            A huggingface datasets element.
        data_path:
            Path to audio files.
        class_num:
            Number of classes.
        sample_rate:
            Desired sample rate.
    Returns:
        A huggingface datasets element.
    """

    y, sr = librosa.load(f"{data_path}/{batch['filename']}.wav",
                            sr=sample_rate,
                            mono=True,
                            res_type='kaiser_fast')

    batch['audio'] = y
    batch['sample_rate'] = sr

    # Creates a hot one encoding version of the label especie_id
    batch['hot_target'] = batch['target']

    return batch