# Evaluate bias by Gaussian

This code is for the paper: **Constructing Holistic Measures for Social Biases in Masked Language Models**.


## Prepare Datasets
You need to download [StereoSet (SS)](https://github.com/moinnadeem/StereoSet) and [CrowS-Pairs (CP)](https://github.com/nyu-mll/crows-pairs) datasets.

```
wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
```
Preprocess the data using the `preprocess.py` provided by Kaneko et al. (2022).
```
python code/preprocess.py --input crows_pairs --output ../data/paralled_cp.json
python code/preprocess.py --input stereoset --output ../data/paralled_ss.json
```
## Prepare Models
You need to download the model weights from Hugging Face to the models folder.

## Data Sampling
To perform random sampling of the data, you can use the following script:

```
python code/sampling.py --sample_rate 0.8 --input ../data/ --output ../data/
```

## Evaluation
You can evaluate the log-likelihood of MLMs using the following script and then save the results:

```
python code/evaluate.py --data [cp, ss] --input ../data/ --output ../result/output/ \
--model [bert, roberta, albert] --sample_rate 1 --method [aul, cps, sss, gms]
```
Alternatively, you can replace main(args) with the following method to calculate the log-likelihood of all MLMs and metrics:

```
models = ['bert', 'roberta', 'albert']
datasets = ['ss', 'cp']
methods = ['aul', 'cps', 'sss', 'gms']
for model in models:
    for data in datasets:
        for method in methods:
            args.model = model
            args.data = data
            args.method = method
            main(args)
```
## Scoring
You can calculate the bias score based on the log-likelihood of MLMs using the following script:

```
python code/scoring.py --sample_rate 1
```
## Optimization
If you have any questions, please contact us via email. 
My email address is `lauyon@tju.edu.cn`.



