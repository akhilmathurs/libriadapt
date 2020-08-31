# LibriAdapt Dataset

These are the instructions to download and use the LibriAdapt dataset released at ICASSP 2020. 

The dataset is released under CC BY 4.0 license. If you use it in your work, please cite the following paper:

*Akhil Mathur, Fahim Kawsar, Nadia Berthouze and Nicholas D. Lane, "Libri-Adapt: a New Speech Dataset for Unsupervised Domain Adaptation," 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 7439-7443, doi: 10.1109/ICASSP40776.2020.9053074*

## Download URLs

Partition 1: US-English speech recorded on six microphones `en-us` [http://www.google.com]

Partition 2: Indian-English speech recorded on six microphones `en-in` [http://www.google.com]

Partition 3: British-English speech recorded on six microphones `en-gb` [http://www.google.com]

Partition 4: Noise recordings from six microphones `noise` [http://www.google.com]

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

**Domain shift due to Acoustic Environemnts**

Finally, the dataset contains three simulated background noise conditions: `Rain`, `Wind`, `Laughter`, in addition to the `Clean` condition. For the three noisy environments, we recorded samples of background noise on all six microphones, and augmented them with the microphone-specific speech files. 

## Dataset Description 

The dataset is offered in four partitions. You can download the appropriate partition depending on the domain shift that you want to study. The partitions are `en-us`, `en-in`, `en-gb` and `noise`. 

Inside each partition, there are subdirectories for the six microphones. Inside each microphone subdirectory, there are separate directories for training data and test data. 

We also provide a CSV file which lists all the files inside each microphone subdirectory. The CSV files contains 3 columns: `wav_filename, wav_filesize, transcript`, and its formatting is compatible with Mozilla DeepSpeech2 model [2] on which all the experiments are done so far. 

### Warnings
The authors have manually verified hundreds of speech recordings, but there is always the possibility that some (or many) of the speech recordings are incomplete or noisy. Please make sure to test for such cases in your data pipelines. 

```
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
```

## Experimenting with the dataset

The dataset could be used to evaluate the performance of ASR models under the presence of domain shift. Let us take the open-sourced Mozilla DeepSpeech2 ASR model as an example. 

1. Follow the instructions here [https://deepspeech.readthedocs.io/en/latest/TRAINING.html] and clone the DeepSpeech2 repo.  

2. Create a Docker container to setup the training environment by following these instructions: [https://deepspeech.readthedocs.io/en/latest/TRAINING.html#basic-dockerfile-for-training]. 

3. Download the DS2 TensorFlow checkpoint and scorer version 0.8.2 from [https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2]    

4. Run the docker container in interactive mode. 

5. Fine-tune the DS2 model for a specific domain (e.g., `en-us` and `ReSpeaker` microphone). 

```python
python3 DeepSpeech.py --n_hidden 2048 --es_epochs 5 --checkpoint_dir /path/to/checkpoints/deepspeech-0.8.2-checkpoint --epochs 15 --save_checkpoint_dir /path/to/checkpoints/en-us/clean/respeaker/ --train_files /path/to/libriadapt/en-us/clean/train_files_respeaker.csv --learning_rate 0.0001 --train_batch_size 16 --scorer_path /path/to/scorer/deepspeech-0.8.2-models.scorer --load_cudnn
```

This will load the DS2 pre-trained checkpoint, fine-tune it for `15` epochs on the .wav files inside `/path/to/libriadapt/en-us/clean/train_files_respeaker.csv` and save the checkpoints inside `/path/to/checkpoints/en-us/clean/respeaker/` 

6. Test the trained model  on a target domain (e.g., `en-us` and `pseye` microphone)

```python
python3 DeepSpeech.py --n_hidden 2048 --load_checkpoint_dir /path/to/checkpoints/en-us/clean/respeaker/ --test_files //path/to/libriadapt/en-us/clean/test_files_pseye.csv --test_batch_size 16 --scorer_path /path/to/scorer/deepspeech-0.8.2-models.scorer --load_cudnn
```

7. There are a lot of DS2 hyperparameters to play with during the fine-tuning step. See here: [https://deepspeech.readthedocs.io/en/latest/Flags.html#training-flags]


### Results of the initial benchmarking

1. Impact of microphone-induced domain shift in the Indian-English `en-in` accented dataset. 

|           | Matrix   | ReSpeaker | USB      | Nexus    | Shure    | PS Eye   |
|-----------|----------|-----------|----------|----------|----------|----------|
| Matrix    | **0.055215** | 0.155436  | 0.073249 | 0.110685 | 0.069024 | 0.119291 |
| ReSpeaker | 0.807440 | **0.056819**  | 0.154067 | 0.158762 | 0.127232 | 0.144229 |
| USB       | 0.312770 | 0.098500  | **0.044086** | 0.094666 | 0.055685 | 0.096603 |
| Nexus     | 0.461204 | 0.108495  | 0.092945 | 0.081738 | **0.054355** | 0.087136 |
| Shure     | 0.622235 | 0.126587  | 0.257692 | 0.115106 | **0.040585** | 0.088368 |
| PS Eye    | 0.612455 | 0.119135  | 0.257711 | 0.110959 | 0.055802 | **0.043578** |

The table reports the WER obtained on the DS2 model. Here, rows correspond to the microphone on which DS2 is finetuned and columns correspond to the microphone on which the fine-tuned model is tested. As we can see, microphone variability has a significant impact on the WER of the model. 

2. Impact of microphone-induced domain shift in the US-English `en-us` accented dataset. 

Let us repeat the experiment with US-accented speech. Here we see slightly higher WERs and also observe the effect of microphone-induced domain shifts. 

3. We can create complex scenarios of domain shifts as well. Let us find the WER when DS2 is trained for `en-us, Clean, ReSpeaker` and tested on `en-gb, Clean, USB`. 


[1] LibriSpeech ASR Corpus http://www.openslr.org/12