# Pripol

> Privacy policy coverage analyis using deep neural networks

Pripol allows users to perform a coverage anaylsis of a website's privacy policy through the use of a simple cli. The annotation scheme and dataset is based off of the [OPP-115 Corpus](https://usableprivacy.org/static/files/swilson_acl_2016.pdf).

This repository contains the code for my undergraduate research thesis: A Comparative Study of Sequence Classification Models for Privacy Policy Coverage Analysis. Awarded "Best Computer Science Poster" at the [2019 CEPS Undergraduate Poster Session](https://www.uoguelph.ca/ceps/events/2019/08/11th-annual-ceps-undergraduate-poster-session). The academic paper can be found [here](https://arxiv.org/abs/2003.04972).

## Built With

-   [TensorFlow](https://www.tensorflow.org/)

## Getting Started

Install the required packages using requirements.txt

```
pip install -r requirements.txt
```

CLI Arguments

```
Usage: python pripol.py --url [privacy policy url] --name [model name]

Options:
    -u, --url           The web page's privacy policy url.
    -n, --name          The model to use for the coverage analysis. Models
                        currently supported: lstm (Long Short-Term Memory),
                        cnn (Convolutional Neural Network).
    -t, --train         Boolean flag indicating you want to train/retrain
                        the model.
    -h, --help          Show help.
```
