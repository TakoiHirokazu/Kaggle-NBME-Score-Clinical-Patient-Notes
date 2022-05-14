# Kaggle-NBME-Score-Clinical-Patient-Notes

<img width="1328" alt="スクリーンショット 2022-05-14 20 27 38" src="https://user-images.githubusercontent.com/30746280/168423739-557f36aa-8e52-4006-8c49-03dbafc107de.png">

# Takoi's part
## Hardware
Google Cloud Platform
- Debian 10.12
- a2-highgpu-1g (vCPU x 12, memory 85 GB)
- 1 x NVIDIA Tesla A100

#### Go to ./takoi directory and do the following.
## Data download
- Download data to ./data from https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/data and unzip it.
- Download data to ./deberta-v2-3-fast-tokenizer  https://www.kaggle.com/datasets/nbroad/deberta-v2-3-fast-tokenizer and unzip it.

## Environment
```
docker-compose up --build
```

## Preprocess
Please run all notebooks in the ``` ./fe ```

## Pretrain(MLM)
Please run the following notebooks in th ``` ./exp ```
- mln_deberta_v3_large.ipynb
- mlm_deberta_v2_xlarge.ipynb
- mlm_deberta_v2_xlxarge.ipynb

## Training(No pseudo label)
Please run the following notebooks in th ``` ./exp ```. 
Sometimes the training of deberta-v2-xlarge fails (score goes to 0), so please retrain the failed fold.
- exp029_deberta_v3_large.ipynb
- exp032_deberta_v2_xlarge.ipynb

## Pseudo label
Please run the following notebooks in th ``` ./exp ```. 
- exp029_deberta_v3_large_make_pseudo.ipynb
- exp032_deberta_v2_xlarge_make_pseudo.ipynb

## Training(With pseudo label) 
Please run the following notebooks in th ``` ./exp ```. 
I used preemptable instances, so one part of the model has separate training code for each fold.
Sometimes the training of deberta-v2-xlarge and deberta-v2-xxlarge fails (score goes to 0), so please retrain the failed fold. Please note that deberta-v2-xxlarge in particular often fails to train.
- deberta-v3-large
    - exp038_deberta_v3_large_with_pseudo.ipynb
- deberta-v2-xlarge
    - exp041_deberta_v2_xlarge_with_pseudo_fold0.ipynb
    - exp041_deberta_v2_xlarge_with_pseudo_fold1_4.ipynb
        - I failed to learn fold 3, so I had to train it again. If you don't fail, you don't need to train only fold 3 below. The training results of each fold need to be combined, so please do that.
    - exp041_deberta_v2_xlarge_with_pseudo_fold3.ipynb
- deberta-v2-xxlarge
    - exp051_deberta_v2_xxlarge_with_pseudo_fold0.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold1.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold2.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold3.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold4.ipynb

## Threshold optimization
Please store the prediction of nakama's exp141 in ``` ./output/nakama ``` . I could not read the pkl file due to  different version of pandas, so I converted it to csv and then stored it. Then please run the following notebooks in th ``` ./exp ```. 
- threshold_optimization_ensemble.ipynb

# Nakama's part
#### Go to ./nakama directory and do the following.
## Data download
- Download data to ./colab/input/nbme-score-clinical-patient-notes from https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/data.
- Download data to ./colab/input/deberta-v2-v3-tokenizer-fast https://www.kaggle.com/datasets/yasufuminakama/deberta-v2-v3-tokenizer-fast.

## Environment
- ./kaggle
    - Kaggle Notebook
    - Add Following Data 
        - https://www.kaggle.com/code/yasufuminakama/nbme-pip-wheels
        - https://www.kaggle.com/datasets/yasufuminakama/deberta-v2-v3-tokenizer-fast
- ./colab
    - Google Colaboratory
    - Note that ./colab/* notebooks are put in ./drive/MyDrive/Colab Notebooks/NBME

## Pretrain(MLM)
Please run the following notebooks using Kaggle Notebook.
- ./kaggle/nbme-deberta-v3-large-mlm-exp1.ipynb
    - Setting fold=0,1,2,3 but 12 hours runtime limit will be applied. You will get fold=0,1 model.
- ./kaggle/nbme-deberta-v3-large-mlm-exp1-fold2-3.ipynb
    - Same code as above except different fold setting.
    - Setting fold=2,3. You will get fold=2,3 model.
Then move pretrained 4 models to ./colab/mlm/kaggle-exp1

## Training(No pseudo label)
Please run the following notebook using Google Colaboratory. 
- ./colab/train/exp48/exp48.ipynb

## Prepare pseudo label dataset from unlabelled data
Please run the following notebook using Google Colaboratory. 
- ./colab/pseudo/make_pseudo_dataset.ipynb

## Training with pseudo label(1st time)
Please run the following notebooks using Google Colaboratory. 
- ./colab/pseudo/exp48/exp48.ipynb
- ./colab/train/exp135/exp135.ipynb

## Training with pseudo label(2nd time)
Please run the following notebooks using Google Colaboratory. 
- ./colab/pseudo/exp135/exp135.ipynb
- ./colab/train/exp141/exp141.ipynb