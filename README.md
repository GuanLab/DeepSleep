## DeepSleep: Near-perfect Detection of Sleep Arousals at Millisecond Resolution by Deep Learning

This is the package of our winning algorithm in the 2018 "You Snooze, You Win" PhysioNet Challenge. 

background: [2018 PhysioNet Challenge](https://physionet.org/challenge/2018/)

Please contact (hyangl@umich.edu or gyuanfan@umich.edu) if you have any questions or suggestions.

![Figure1](figure/Fig_1.png?raw=true "Title")

---

## Installation
Git clone a copy of code:
```
git clone https://github.com/GuanLab/DeepSleep.git
```
## Required dependencies

* [python](https://www.python.org) (3.6.5)
* [numpy](http://www.numpy.org/) (1.13.3). It comes pre-packaged in Anaconda.
* [scikit-learn](http://scikit-learn.org) (0.19.0) A popular machine learning package. It can be installed by:
```
pip install -U scikit-learn
```
* [tensorflow](https://www.tensorflow.org/) (1.8.0)
* [keras](https://keras.io/) (2.0.4)

## Dataset

We used the public training data from the challenge, which contains 994 polysomnographic recordings. Each polysomnogram has 13 channels, corresponding to 6 EEG (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1), 3 EMG (chin, abdominal, chest), Airflow, Saturation of oxygen, and ECG. These polysomnogram have been manually labeled by sleep experts, including sleep stages, arousals, apneas and other sleep events. Download link and details can be found [HERE](https://physionet.org/physiobank/database/challenge/2018/).

## Model development

### 1. prepare polysomnogram data

First download the data and put them into the folder "./data/training/". The training dataset is approximately 135 GB in size.
Or you can try our code using the examples provided but unzip them first:
```
cd ./data/training/
unzip tr03-0078.zip
unzip tr03-0079.zip
cd ../../
```

### 2. preprocessing

Since the lengths of sleep recordings are different, we first make uniform these recordings to the same 8-million length (2^23 = 8,388,608) by padding zeros at both the beginning and the end. 
```
unzip ref555.zip
python uniform_8m.py
```

### 3. prediction

Now you can run predictions by providing the sample id:
```
python predict.py tr03-0078
```
It will generate a file called "tr03-0078.vec", each line corresponds to the prediction for each time point in the original polysomnogram.

If you want to run multiple predictions, you can try:

```
python predict.py tr03-0078 tr03-0079
```


### 4. scoring

The AUPRC and AUROC can be calculated by running:
```
python score2018.py *vec -r ./data/training/
```
It will print out the results like this:
```
record          AUROC     AUPRC
_______________________________
tr03-0078    0.932568  0.554207
tr03-0079    0.975179  0.314851
_______________________________
Overall      0.950086  0.492553

```

### 5. example
A 50-second example with 13 physiological signals and the corresponding binary label of sleep or arousal:

![Figure2](figure/example.gif?raw=true "Title")


