# LibriAdapt Dataset

These are the instructions to download and use the [LibriAdapt dataset](https://ieeexplore.ieee.org/document/9053074) released at ICASSP 2020. 

The dataset is released under CC BY 4.0 license. If you use any part of the dataset in your work, please cite the following paper:

*Akhil Mathur, Fahim Kawsar, Nadia Berthouze and Nicholas D. Lane, "Libri-Adapt: a New Speech Dataset for Unsupervised Domain Adaptation," 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 7439-7443, doi: 10.1109/ICASSP40776.2020.9053074*


 - [Dataset Overview](https://github.com/akhilmathurs/libriadapt#dataset-overview)
 - [Dataset Description](https://github.com/akhilmathurs/libriadapt#dataset-description)
 - [Download](https://github.com/akhilmathurs/libriadapt#Download)
 - [Experimenting with the dataset](https://github.com/akhilmathurs/libriadapt#experimenting-with-the-dataset)
 - [Baseline results](https://github.com/akhilmathurs/libriadapt#baseline-results)
 - [Contact](https://github.com/akhilmathurs/libriadapt#contact)

## Dataset Overview 

The LibriAdapt dataset is built on top of the LibriSpeech dataset [[1]](#1), specifically using the `train-clean-100` partition for training data and `test-clean` partition for test data. 

It is primarily designed to faciliate domain adaptation research for ASR models, and contains the following three types of domain shifts in the data (please refer to our [paper](https://ieeexplore.ieee.org/document/9053074) for details on the data collection methodology): 

**Domain shift due to Microphones**

Both training and test sets are recorded simultaneously on six off-the-shelf microphones available in the market to develop speech products. They are as follows:


|      Device     | Channels |                                Use-cases                                |
|:---------------:|:--------:|:-----------------------------------------------------------------------:|
|   Matrix Voice  |     7    | Internet of Things (IoT) systems, voice assistants, smart home products |
|    ReSpeaker    |     7    | Internet of Things (IoT) systems, voice assistants, smart home products |
| PlayStation Eye |     4    |                             Gaming Consoles                             |
|  USB microphone |     1    |           Embedded-scale IoT systems (e.g., with Raspberry pi)          |
|  Google Nexus 6 |     3    |                 Smartphone interaction, voice assistants                |
|    Shure MV5    |     1    |           Desktop microphone for podcasts, video conferencing           |


For all the multi-channel microphones, we only release the merged speech output provided by the microphone manufacturers, after applying beamforming or other channel merging techniques. 

**Domain shift due to Speaker Accents**

The dataset contains the LibriSpeech texts spoken in three accents: US-English (`en-us`), Indian-English (`en-in`) and British-English (`en-gb`). For `en-us`, we replayed the original LibriSpeech audios and recorded them on the six microphones. The other accented speech files are generated using the Google Wavenet TTS model: for this purporse, we passed the Librispeech transcripts to the TTS model and obtained the accented speech. 

Since the TTS model has a limited number of speaking styles for each accent, it effectively reduces the speaker variability in the `en-in` and `en-gb` partitions. Hence, the ASR task becomes somewhat easier for them. 

**Domain shift due to Acoustic Environemnts**

Finally, the dataset contains three simulated background noise conditions: `Rain`, `Wind`, `Laughter`, in addition to the `Clean` condition. For the three noisy environments, we recorded samples of background noise on all six microphones, and augmented them with the microphone-specific speech files. 

## Dataset Description 

The dataset is offered in four partitions. You can download the appropriate partition depending on the domain shift that you want to study. The partitions are `en-us`, `en-in`, `en-gb` and `noise`. 

Inside each partition, there are subdirectories for the six microphones. Inside each microphone subdirectory, there are separate directories for training data and test data. 

We also provide a CSV file which lists all the .wav files inside each microphone subdirectory. The CSV files contains 3 columns: `wav_filename, wav_filesize, transcript`, and their formatting is compatible with the format expected by the Mozilla DeepSpeech2 model [[2]](#2).  

```
📦libriadapt
 ┣ 📂en-gb
 ┃ ┗ 📂clean
 ┃ ┃ ┣ 📂matrix
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂nexus6
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂pseye
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂respeaker
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂shure
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┗ 📂usb
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┣ 📂en-in
 ┃ ┗ 📂clean
 ┃ ┃ ┣ 📂matrix
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂nexus6
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂pseye
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂respeaker
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂shure
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┗ 📂usb
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┣ 📂en-us
 ┃ ┗ 📂clean
 ┃ ┃ ┣ 📂matrix
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂nexus6
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂pseye
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂respeaker
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂shure
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┗ 📂usb
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┗ 📂noise
 ┃ ┣ 📂matrix
 ┃ ┣ 📂nexus6
 ┃ ┣ 📂pseye
 ┃ ┣ 📂respeaker
 ┃ ┣ 📂shure
 ┃ ┗ 📂usb
```



### Download

**Clean US-English speech recorded on six microphones (`en-us`)** 

[Part 1 (12GB)](https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_aa) 

[Part 2 (12GB)](https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ab) 

[Part 3 (12GB)](https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ac) 

[Part 4 (12GB)](https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ad) 

[Part 5 (6.4GB)](https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ae)


**Clean Indian-English speech recorded on six microphones (`en-in`)**

[Part 1 (10GB)](https://sensix.tech/libriadapt/libriadapt-en-in.tar.gz.part_aa)

[Part 2 (8.6GB)](https://sensix.tech/libriadapt/libriadapt-en-in.tar.gz.part_ab)


**Clean British-English speech recorded on six microphones (`en-gb`)**

[Part 1 (10GB)](https://sensix.tech/libriadapt/libriadapt-en-gb.tar.gz.part_aa)

[Part 2 (9.5GB)](https://sensix.tech/libriadapt/libriadapt-en-gb.tar.gz.part_ab)

**Noise recordings of rain, wind, laughter from six microphones (`noise`)** 

[Part 1 (24 MB)](https://sensix.tech/libriadapt/libriadapt-noise.tar.gz)


#### Merge the tar.gz.part_* files

Once the compressed tar files are downloaded, they need to be merged using `cat` and then uncompressed. For example:

```sh
cat libriadapt-en-in.tar.gz.part_a* > libriadapt-en-in.tar.gz
tar -zxvf libriadapt-en-in.tar.gz
```


#### Generating the noisy datasets

We provide a script to augment the clean speech files with the noise samples, and generate a noisy-version of the dataset. Check [augment_noise.py](https://github.com/akhilmathurs/libriadapt/blob/master/augment_noise.py). 



## Experimenting with the dataset

The dataset could be used to evaluate the performance of ASR models under the presence of domain shift. Let us take the open-sourced Mozilla DeepSpeech2 (DS2) ASR model [[2]](#2) as an example. 

1. Follow the instructions here [https://deepspeech.readthedocs.io/en/latest/TRAINING.html] and clone the DeepSpeech2 repo.  

2. Create a Docker container to setup the training environment by following these instructions: [https://deepspeech.readthedocs.io/en/latest/TRAINING.html#basic-dockerfile-for-training]. 

3. Download the DS2 TensorFlow checkpoint and scorer version 0.8.2 from [https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2]    

4. Run the docker container in interactive mode. 

5. Fine-tune the DS2 model for a specific domain (e.g., `en-us`, `Clean`, and `ReSpeaker` microphone). 

```python
python3 DeepSpeech.py --n_hidden 2048 --es_epochs 5 --epochs 15 \
--checkpoint_dir /path/to/checkpoints/deepspeech-0.8.2-checkpoint \
--save_checkpoint_dir /path/to/checkpoints/en-us/clean/respeaker/ \ 
--train_files /path/to/libriadapt/en-us/clean/train_files_respeaker.csv \ 
--scorer_path /path/to/scorer/deepspeech-0.8.2-models.scorer \
--learning_rate 0.0001 --train_batch_size 16 --load_cudnn
```

This will load the DS2 pre-trained checkpoint, fine-tune it for `15` epochs on the .wav files listed in `/path/to/libriadapt/en-us/clean/train_files_respeaker.csv` and save the checkpoints at `/path/to/checkpoints/en-us/clean/respeaker/` 

6. Test the finetuned model on a target domain (e.g., `en-us`, `Clean`, and `pseye` microphone)

```python
python3 DeepSpeech.py --n_hidden 2048 --test_batch_size 16 --load_cudnn \
--load_checkpoint_dir /path/to/checkpoints/en-us/clean/respeaker/ \
--test_files /path/to/libriadapt/en-us/clean/test_files_pseye.csv \
--scorer_path /path/to/scorer/deepspeech-0.8.2-models.scorer 
```

7. There are many hyperparameters to play with during the fine-tuning step of DS2. See here: [https://deepspeech.readthedocs.io/en/latest/Flags.html#training-flags]


## Baseline results

Note that we use the latest version of DeepSpeech2 (0.8.2) for the experiments below. Hence the results differ from those reported in our paper which were obtained on an older version of DS2. 

For these baseline experiments, the raw speech files in .wav format are directly fed to DS2 without doing any additional pre-processing. 

### Performance of pre-trained DS2 on Librispeech test sets recorded on different microphones ###

Below we compare the Word Error Rate (WER) of the pre-trained DS2 (0.8.2) model on the test datasets from different microphones. No finetuning is done on the model in this experiment. 

DS2 has an advertised WER of 0.0597 on the original Librispeech-clean test corpus (`en-us`). However, when the same test corpus is recorded on different microphones in the LibriAdapt dataset, the WER increases significantly (as high as 4.5x). 

|                   | Librispeech-clean-test |  Matrix  |  Nexus6  |  PS Eye  | ReSpeaker | Shure    | USB      |
|:-----------------:|:----------------------:|:--------:|:--------:|:--------:|-----------|----------|----------|
| DeepSpeech2 0.8.2 |       **0.0597**       | 0.276390 | 0.106245 | 0.116866 | 0.127056  | 0.082481 | 0.169147 |

This increase could be partly attributed to the data collection methodology of LibriAdapt where we replay the speech files using a speaker. Nevertheless, the variations in WER across microphones is interesting and present clear opportunities for applying domain adaptation. 

### Microphone-induced domain shift in the Indian-English accented dataset (`en-in`) ###

Let us finetune the DS2 model on the Indian-English dataset obtained from the six microphones, and study the generalization performance of the model. 

The following table reports the WER for different experiment settings. Here, rows correspond to the microphone on which DS2 is finetuned and columns correspond to the microphone on which the fine-tuned model is tested. As we can see, microphone variability has a significant impact on the WER of the model. 

|           | Matrix   | ReSpeaker | USB      | Nexus    | Shure    | PS Eye   |
|-----------|----------|-----------|----------|----------|----------|----------|
| Matrix    | **0.055215** | 0.155436  | 0.073249 | 0.110685 | 0.069024 | 0.119291 |
| ReSpeaker | 0.807440 | **0.056819**  | 0.154067 | 0.158762 | 0.127232 | 0.144229 |
| USB       | 0.312770 | 0.098500  | **0.044086** | 0.094666 | 0.055685 | 0.096603 |
| Nexus     | 0.461204 | 0.108495  | 0.092945 | 0.081738 | **0.054355** | 0.087136 |
| Shure     | 0.622235 | 0.126587  | 0.257692 | 0.115106 | **0.040585** | 0.088368 |
| PS Eye    | 0.612455 | 0.119135  | 0.257711 | 0.110959 | 0.055802 | **0.043578** |


### Microphone-induced domain shift in the US-English accented dataset (`en-us`) ###

Let us repeat the experiment with US-accented speech and finetune the DS2 model on `en-us` dataset for 20 epochs. Here we see slightly higher WERs and again observe the effect of microphone-induced domain shift. 


|           | Matrix   | ReSpeaker    | USB          | Nexus        | Shure        | PS Eye       |
|-----------|----------|--------------|--------------|--------------|--------------|--------------|
| Matrix    | 0.121247 | 0.133706     | 0.135427     | **0.112250** | 0.115262     | 0.129755     |
| ReSpeaker | 0.250220 | **0.106883** | 0.134528     | 0.106695     | 0.113619     | 0.115829     |
| USB       | 0.206544 | 0.128601     | **0.107419** | 0.112133     | 0.122108     | 0.179083     |
| Nexus     | 0.211630 | 0.121345     | 0.126900     | **0.094334** | 0.097893     | 0.125178     |
| Shure     | 0.245232 | 0.140493     | 0.152346     | 0.108534     | **0.086843** | 0.130538     |
| PS Eye    | 0.245721 | 0.133022     | 0.177597     | 0.128504     | 0.111741     | **0.096407** |


### Study more complex scenarios by mixing various domain shifts ###

LibriAdapt allows simulating multiple domain shifts in the data. 

Let us find the WER when DS2 is finetuned on the training data from `{en-us, Clean, ReSpeaker}`, and tested on 

1. `{en-us, Clean, ReSpeaker}` (i.e., no domain shift), 

2. `{en-gb, Clean, ReSpeaker}` (i.e., accent shift),

3. `{en-gb, Clean, PS Eye}` (i.e., accent and microphone shift),

4. `{en-gb, Rain, PS Eye}` (i.e., accent, microphone, background noise shift).


|                         | en-us, Clean, ReSpeaker | en-gb, Clean, ReSpeaker | en-gb, Clean, PS Eye | en-gb, Rain, PS Eye |
|:-----------------------:|:-----------------------:|:-------------------:|:--------------------:|:-------------------:|
| en-us, Clean, ReSpeaker |       **0.106883**      |       0.148786      |       0.195826       |       0.256049      |


## Contact

If you have any questions or find any errors in the dataset, please reach out to akhilmathurs{at}gmail{dot}{com}.  

### Disclaimer
The authors have manually verified hundreds of speech recordings, but there is always the possibility that some (or many) of the speech recordings are incomplete or noisy. Please make sure to test for such cases in your data pipelines. 


<a id="1">[1]</a> LibriSpeech ASR Corpus http://www.openslr.org/12

<a id="2">[2]</a> https://github.com/mozilla/DeepSpeech 
