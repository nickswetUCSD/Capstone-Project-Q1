# Test Time Adaptation
Oftentimes, neural networks are forced to react to data from a test domain that does not reflect the data that the model was trained on.
For example, a self-driving car that's trained on imagery taken on a sunny day (source domain) will need additional "adaptation" to react to imagery from foggy days, imagery with poor lighting, or imagery with glare (distribution-shifted test domains). Introducing Test-Time Adaptation!: the notion of updating a model at test-time with nothing but the model itself and unlabeled test data.

This is our replication of test-time adaptation code for this [excellent repo](https://github.com/locuslab/tta_conjugate) using a test-time adaptation technique known as "Conjugate Pseudolabeling". Credits to Sachin Goyal*, Mingjie Sun*, Aditi Raghunanthan, J. Zico Kolter. Check out their [paper](https://arxiv.org/pdf/2207.09640) on the subject, too.

## Installation
First clone the repository:
```bash
git clone https://github.com/nickswetUCSD/Capstone-Project-Q1.git
```
    
Then install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To train a model on the source domain, run:
```bash
python utilities/train_source.py --loss cross_entropy
```
`loss` can be either `cross_entropy` for cross-entropy loss or `poly` for polyloss.

Then, to run test-time adaptation on the target domain, run:
```bash
python utilities/test_pseudolabels.py --pseudo_label hard
```
`pseo_label` can be either `hard` for hard pseudolabels or `conjugate` for conjugate pseudolabels.

