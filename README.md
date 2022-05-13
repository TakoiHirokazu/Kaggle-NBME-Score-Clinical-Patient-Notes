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