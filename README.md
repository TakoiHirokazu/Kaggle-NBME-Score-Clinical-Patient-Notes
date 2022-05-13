# Kaggle-NBME-Score-Clinical-Patient-Notes

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
        - Training of fold3 failed, so only fold3 was trained again.
    - exp041_deberta_v2_xlarge_with_pseudo_fold3.ipynb
- deberta-v2-xxlarge
    - exp051_deberta_v2_xxlarge_with_pseudo_fold0.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold1.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold2.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold3.ipynb
    - exp051_deberta_v2_xxlarge_with_pseudo_fold4.ipynb
