import os
import sys

from SPEECHTOTEXTTENSORFLOW.exceptions import STTException
from SPEECHTOTEXTTENSORFLOW.logger import logging


class VectorizeChar:
    def __init__(self, max_len=50):
        try:
            logging.info("Initialize and call the VectorizeChar class of data utils")
            self.vocab = (
                ["-", "#", "<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + [" ", ".", ",", "?"]
            )
            self.max_len = max_len
            self.char_to_idx = {}
            for i, ch in enumerate(self.vocab):
                self.char_to_idx[ch] = i
        except Exception as e:
            raise STTException(e, sys)

    def __call__(self, text):
        try:
            text = text.lower()
            text = text[: self.max_len - 2]
            text = "<" + text + ">"
            pad_len = self.max_len - len(text)
            return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len
        except Exception as e:
            raise STTException(e, sys)

    def get_vocabulary(self):
        try:
            return self.vocab
        except Exception as e:
            raise(e, sys)
    
def get_data(wavs, id_to_text, maxlen=50):
    """ returns mapping of audio paths and transcription texts """
    try:
        logging.info("Entered get_data function of data utils")
        data = []
        # print(f"Maximum length of characters: {maxlen}")
        for w in wavs:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_text[id]) < maxlen:
                data.append({"audio": w, "text": id_to_text[id]})
        logging.info("Exited get_data function of model utils")
        return data
    except Exception as e:
        raise STTException(e, sys)