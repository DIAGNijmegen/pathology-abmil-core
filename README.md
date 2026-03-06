# CLAM and Additive MIL

## Status

This repository accompanies a manuscript currently under review. The codebase, documentation and dataset access instructions may be updated during the peer-review and publication process. Additional resources, including the full dataset and reproducibility scripts, will be finalized upon acceptance of the manuscript. 

## About
This repository integrates [**CLAM**](https://github.com/mahmoodlab/CLAM) with **Additive Multiple Instance Learning [(AddMIL)](https://openreview.net/forum?id=5dHQyEcYDgA)** [2]. It was developed to support the study:

**Title** 
: Automated detection of cutaneous squamous cell carcinoma (CSCC) in whole slide images of skin biopsies using weakly-supervised learning approaches

**Authors**
: Catherine Chia, Stephan Dooper, Antien Mooyaart, Avital Amir, Marlies Wakkee, and Geert Litjens

## Public datasets
Two internal datasets are used in this study, and both are hosted on an AWS S3 bucket:

`s3://cobra-pathology/`

#### COBRA
The COBRA dataset is publicly accessible via the `bcc` directory:

##### Instruction
1. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Bucket name: `s3://cobra-pathology/packages/bcc/`
3. To browse: `aws s3 ls --no-sign-request s3://cobra-pathology/packages/bcc/`
4. To download: `aws s3 cp --no-sign-request s3://cobra-pathology/packages/bcc/ <destination_path>`

Relevant directory structure
```
cobra-pathology
└── packages
    ├── bcc
        ├── annotations
        ├── images
    ├── ood
```
#### CSCC 
The CSCC dataset consists of two batches. The batch dated 2016-2020 is accessible via the `ood` directory, whereas the later 2021-2024 batch will be made publicly available upon manuscript acceptance. 

##### Instruction
1. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Bucket name: `s3://cobra-pathology/packages/ood/`
3. To browse: `aws s3 ls --no-sign-request s3://cobra-pathology/packages/ood/`
4. To download: `aws s3 cp --no-sign-request s3://cobra-pathology/packages/ood/ <destination_path>`

Relevant directory structure
```
cobra-pathology
└── packages
    ├── bcc
    ├── ood
        ├── annotations
        ├── images
```







