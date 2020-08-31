# LibriAdapt Dataset

These are the instructions to download and use the LibriAdapt dataset released at ICASSP 2020. 

The dataset is released under CC BY 4.0 license. If you use it in your work, please cite the following paper:

*Akhil Mathur, Fahim Kawsar, Nadia Berthouze and Nicholas D. Lane, "Libri-Adapt: a New Speech Dataset for Unsupervised Domain Adaptation," 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 7439-7443, doi: 10.1109/ICASSP40776.2020.9053074*

## Dataset Overview 

The LibriAdapt dataset is built on top of the LibriSpeech dataset [1], specifically using the `train-clean-100` partition for training data and `test-clean` partition for test data. 

It is primarily designed to faciliate domain adaptation research for ASR models, and contains the following three types of domain shifts in the data (please refer to the paper for details on the data collection methodology): 

**Domain shift due to Microphones**

Both training and test sets are recorded simultaneously on six off-the-shelf microphones that are available in the market to develop speech products. They are as follows:


|      Device     | Channels |                                Use-cases                                |
|:---------------:|:--------:|:-----------------------------------------------------------------------:|
|   Matrix Voice  |     7    | Internet of Things (IoT) systems, voice assistants, smart home products |
|    ReSpeaker    |     7    | Internet of Things (IoT) systems, voice assistants, smart home products |
| PlayStation Eye |     4    |                             Gaming Consoles                             |
|  USB microphone |     1    |           Embedded-scale IoT systems (e.g., with Raspberry pi)          |
|  Google Nexus 6 |     3    |                 Smartphone interaction, voice assistants                |
|    Shure MV5    |     1    |           Desktop microphone for podcasts, video conferencing           |


**Domain shift due to Speaker Accents**

The dataset contains the LibriSpeech texts spoken in three accents: US-English (`en-us`), Indian-English (`en-in`) and British-English (`en-gb`). For `en-us`, we replayed the original LibriSpeech audios and recorded them on the six microphones. The other accented speech files are generated using the Google Wavenet TTS model: for this purporse, we passed the Librispeech transcripts to the TTS model and obtained the accented speech. 

Since the TTS model has a limited number of speaking styles for each accent, it effectively reduces the speaker variability in the `en-in` and `en-gb` partitions. Hence, the ASR task becomes somewhat easier for them. 

**Domain shift due to Acoustic Environemnt**

Finally, the dataset contains three simulated background noise conditions: `Rain`, `Wind`, `Laughter`, in addition to the `Clean` condition. For the three noisy environments, we recorded samples of background noise again on all six microphones, and augmented them with the speech files. 

## Dataset Description 

📦libriadapt
 ┣ 📂en-us
 ┃ ┣ 📂matrix
 ┃ ┣ 📂nexus6
 ┃ ┣ 📂pseye
 ┃ ┣ 📂respeaker
 ┃ ┣ 📂shure
 ┃ ┗ 📂usb
 ┣ 📂en-in
 ┃ ┣ 📂matrix
 ┃ ┣ 📂nexus6
 ┃ ┣ 📂pseye
 ┃ ┣ 📂respeaker
 ┃ ┣ 📂shure
 ┃ ┗ 📂usb
 ┣ 📂en-gb
 ┃ ┣ 📂matrix
 ┃ ┣ 📂nexus6
 ┃ ┣ 📂pseye
 ┃ ┣ 📂respeaker
 ┃ ┣ 📂shure
 ┃ ┗ 📂usb
 ┗ 📂noise
 ┃ ┣ 📂matrix
 ┃ ┣ 📂nexus6
 ┃ ┣ 📂pseye
 ┃ ┣ 📂respeaker
 ┃ ┣ 📂shure
 ┃ ┗ 📂usb


## Quantification of Domain Shift


|           | Matrix   | ReSpeaker | USB      | Nexus    | Shure    | PS Eye   |
|-----------|----------|-----------|----------|----------|----------|----------|
| Matrix    | 0.055215 | 0.155436  | 0.073249 | 0.110685 | 0.069024 | 0.119291 |
| ReSpeaker | 0.807440 | 0.056819  | 0.154067 | 0.158762 | 0.127232 | 0.144229 |
| USB       | 0.312770 | 0.098500  | 0.044086 | 0.094666 | 0.055685 | 0.096603 |
| Nexus     | 0.461204 | 0.108495  | 0.092945 | 0.081738 | 0.054355 | 0.087136 |
| Shure     | 0.622235 | 0.126587  | 0.257692 | 0.115106 | 0.040585 | 0.088368 |
| PS Eye    | 0.612455 | 0.119135  | 0.257711 | 0.110959 | 0.055802 | 0.043578 |




[1] LibriSpeech ASR Corpus http://www.openslr.org/12