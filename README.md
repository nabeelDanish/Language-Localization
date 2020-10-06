# Language-Localization
An NLP based approach to detect and identify languages word-by-word for audio data on local Pakistani languages

## Language Localization on Audio Files and Live Audio
### Nabeel Danish

A spectogram-based approach to identifying langauges most commonly spoken in Pakistan.
This includes Urdu, English, Arabic, Pashto, and Sindhi. The model is contructed as CNN with LSTM layers
achieving an accuracy of 95%.

## Usage

import the file ```languageLocalization.py``` to use in your script

## Functions
```
def languageLocalize(inputFile, extension, chunk_file):
```
	
	Parameters:
	
	inputFile -- path to file for audio
	extension -- audio file extension
	chunk_file -- path to folder where the model stores preprocessing data

	Return Value:

	pred -- python array of strings containing the languauges predicted at positional interval
		pred[i] is the language detected between (i - 1) and (i)th second
		Example:
			pred[3] = 'english' means english detected between 2-3 sec of audio
 
## Live Audio Detection
the notebook contains the script to run live detection on audio.

## Dependancies
Tensorflow
Keras
Numpy
Scipy
Pydub
Librosa
