# Pripol

> Privacy policy coverage analyiss using deep neural networks

This repository contains the code for my undergraduate research thesis: A Comparative Study of Sequence Classification Models for Privacy Policy Coverage Analysis. Awarded "Best Computer Science Poster" at the [2019 CEPS Undergraduate Poster Session](https://www.uoguelph.ca/ceps/events/2019/08/11th-annual-ceps-undergraduate-poster-session).

The academic paper can be found [here](https://arxiv.org/abs/2003.04972).

## Built With

-   [TensorFlow](https://www.tensorflow.org/)
-   [OPP-115 Dataset](https://usableprivacy.org/static/files/swilson_acl_2016.pdf)

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

## TODO

-   Create a frontend that displays the website's privacy policy along with the generated segment annotations
