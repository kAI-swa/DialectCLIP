import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import Optional, Literal
from torch.utils.data import Dataset
from melo.api import TTS


class Aishell_Dataset(Dataset):
    def directory_architecture(self):
        architecture = \
            "Please reformat the directory architecture to:\n \
            |-speech \n \
            |-S0002 \n \
            |- | -xxxxxx.wav \n \
            |-S0003 \n \
            |- | -wxxxxx.wav \n \
            |-transcript \n \
            |-transcript.txt"
        return architecture

    def __init__(self, file_path: Optional[str] = "./Data/Aishell", mode: Literal["train", "test"] = "train"):
        '''
        -------------------
        Inputs:
            file_path: root directory to the dialect and high resource speech waveform data and corresponding translation text
        '''
        super().__init__()
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File Path {file_path} not exists")
            
        # Speech waveform: .wav -> numpy array(dtype=float32)
        speech_path = os.path.join(file_path, f"speech/{mode}")
        speech_list = []
        if not os.path.exists(speech_path):
            raise FileNotFoundError(self.directory_architecture())
        file_path_lists = os.listdir(path=speech_path)
        file_path_lists.sort()
        for file_path_name in file_path_lists:
            files_name = os.path.join(speech_path, file_path_name)
            files = os.listdir(path=files_name)
            files.sort()
            for file in files:
                speech_data, _ = sf.read(file=os.path.join(files_name, file), dtype="float32")
                speech_list.append(speech_data)
        self.speech = speech_list
        num_speech = len(self.speech)
            
        # transcript
        transcript_path = os.path.join(file_path, "transcript/transcript.txt")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_list = f.readlines()
        self.transcript = [transcript.split(sep=" ", maxsplit=1)[1] for transcript in transcript_list]
        if mode == "train":
            self.transcript = self.transcript[:num_speech]
        elif mode == "test":
            self.transcript = self.transcript[num_speech:]
        else:
            raise ValueError(f"mode not support, got {mode}, expected 'train' or 'test'")
    
    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        return self.speech[index], self.transcript[index]


class Uyghur_Dataset(Dataset):
    def directory_architecture(self):
        architecture = \
            "Please reformat the directory architecture to:\n \
            |-dialect \n\
            |-wave1.wav \n \
            |-speech \n \
            |-wave1.wav \n \
            |-transcript \n \
            |-transcript.txt"
        return architecture

    def __init__(self, file_path: Optional[str] = "./Data/Uyghur_Chinese"):
        '''
        -------------------
        Inputs:
            file_path: root directory to the dialect and high resource speech waveform data and corresponding transcript text
        '''
        super().__init__()
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File Path {file_path} not exists")

        # Dialect waveform: .wav -> numpy array(dtype=float32)
        dialect_path = os.path.join(file_path, "dialect")
        if not os.path.exists(dialect_path):
            raise FileNotFoundError(self.directory_architecture())
        files = os.listdir(path=dialect_path)
        dialect_list = []
        for file in files:
            dialect_data, _ = sf.read(file=os.path.join(dialect_path, file), dtype="float32")
            dialect_list.append(dialect_data)
        self.dialect = dialect_list

        # Speech waveform: .wav -> numpy array(dtype=float32)
        speech_path = os.path.join(file_path, "speech")
        if not os.path.exists(speech_path):
            raise FileNotFoundError(self.directory_architecture())
        files = os.listdir(path=speech_path)
        speech_list = []
        for file in files:
            speech_data, _ = sf.read(file=os.path.join(speech_path, file), dtype="float32")
            speech_list.append(speech_data)
        self.speech = speech_list

        # transcript-Uyghur
        transcript_path = os.path.join(file_path, "transcript/transcript.txt")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_list = f.readlines()
        self.transcript = [transcript.split(sep="\t")[1] for transcript in transcript_list]

    def __len__(self):
        return len(self.transcript)

    def __getitem__(self, index):
        return self.speech[index], self.dialect[index], self.transcript[index]
    

class temp_dataset:
    def __init__(self, file_path):
        super().__init__()
        self.speech = np.array(np.random.randn(120, 3000), dtype=np.float32)
        self.dialect = np.array(np.random.randn(120, 3000), dtype=np.float32)

        transcript_path = os.path.join(file_path, "transcript/transcript.txt")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_list = f.readlines()

        self.transcript = [transcript.split(sep="\t")[1] for transcript in transcript_list]
        self.transcript = self.transcript[:64]

    def __len__(self): 
        return len(self.transcript)

    def __getitem__(self, index):
        return self.speech[index], self.dialect[index], self.transcript[index]


class tts_wrapper:
    def __init__(
        self,
        language: Optional[Literal["ZH", "EN", "ES"]] = None,
        device: Optional[Literal["cpu", "cuda", "auto"]] = None):
        '''
        Use the SOTA tts model from melo.ai which is a multilingual TTS model to generate high resourcee language speech
        -------------------------
        Args:
            language: Target language
            device: Target device to run on
        '''
        super().__init__()
        self.language = language
        self.tts_model = TTS(language=language, device=device)

    def run(
        self,
        text_file: str, 
        save_path: Optional[str] = None, 
        speed: Optional[float] = None):
        '''
        ------------------
        Inputs:
            text_file: Speech transcript file path
            save_path: Saving path for generated speech from tts model
        '''
        if not os.path.exists(text_file):
            raise FileNotFoundError("Text transcription file can not be found")
        
        if not os.path.exists(save_path):
            print(f"{save_path} not exists, create a temp dict")
            os.makedirs("./Data/temp")
        
        with open(text_file, "r", encoding="utf-8") as f:
            transcription_list = f.readlines()
        source_list = [transcript.split("\t") for transcript in transcription_list]

        speaker_ids = self.tts_model.hps.data.spk2id

        with tqdm(source_list, total=len(source_list)) as t:
            for id, text in t:
                t.set_description("TTS generation")
                output_path = os.path.join(save_path, id+".wav")
                self.tts_model.tts_to_file(text, speaker_ids[self.language], output_path, speed=speed)
            print("================")
            print("Done!")
