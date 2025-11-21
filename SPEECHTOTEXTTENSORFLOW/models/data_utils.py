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
        for w in wavs:
            # Use os.path to handle both Windows and Linux paths
            filename = os.path.basename(w)  # Get just the filename
            id = filename.split(".")[0]     # Remove the .wav extension
            
            # Debug print to see what's happening
            print(f"Processing file: {w}")
            print(f"Extracted ID: {id}")
            print(f"Available keys: {list(id_to_text.keys())[:5]}")  # Show first 5 keys
            
            if id in id_to_text and len(id_to_text[id]) < maxlen:
                data.append({"audio": w, "text": id_to_text[id]})
            elif id not in id_to_text:
                print(f"Warning: ID '{id}' not found in metadata")
                
        logging.info(f"Successfully processed {len(data)} audio files")
        logging.info("Exited get_data function of model utils")
        return data
    except Exception as e:
        raise STTException(e, sys)