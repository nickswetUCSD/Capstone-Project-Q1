# Test Time Adaptation
Oftentimes, neural networks are forced to react to data from a test domain that does not reflect the data that the model was trained on.
For example, a self-driving car that's trained on imagery taken on a sunny day (source domain) will need additional "adaptation" to react to imagery from foggy days, imagery with poor lighting, or imagery with glare (distribution-shifted test domains). Introducing Test-Time Adaptation!: the notion of updating a model at test-time with nothing but the model itself and unlabeled test data.

This is our replication of test-time adaptation code for this [excellent repo](https://github.com/locuslab/tta_conjugate) using a test-time adaptation technique known as "Conjugate Pseudolabeling". Credits to Sachin Goyal*, Mingjie Sun*, Aditi Raghunanthan, J. Zico Kolter. Check out their [paper](https://arxiv.org/pdf/2207.09640) on the subject, too.
