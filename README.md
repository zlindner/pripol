# Pripol

> Privacy policy coverage analyis using deep neural networks

Pripol allows users to perform a coverage anaylsis of a website's privacy policy through a simple web interface. Pripol employs a Convolutional Neural Network (CNN) to predict the most likely corresponding data practices for a given segment/section of the given policy. The annotation scheme and dataset is based off of the [OPP-115 Corpus](https://usableprivacy.org/static/files/swilson_acl_2016.pdf).

I wrote a related paper: <em>[A Comparative Study of Sequence Classification Models for Privacy Policy Coverage Analysis](https://arxiv.org/abs/2003.04972)</em> for my undergraduate research thesis which compares sequence classification models (deep learning and classical machine learning) for the purpose of privacy policy coverage analysis, provides an overview of each model. The project/paper was awarded <em>Best Computer Science Project</em> at the [2019 CEPS Undergraduate Poster Session](https://www.uoguelph.ca/ceps/events/2019/08/11th-annual-ceps-undergraduate-poster-session).

## Built With

-   [Flask](https://flask.palletsprojects.com/en/1.1.x/)
-   [React](https://reactjs.org/)
-   [TensorFlow](https://www.tensorflow.org/)
-   [TypeScript](https://www.typescriptlang.org/)

## Getting Started

### Client

Install the required dependencies

```
npm install
```

Generate a full static production build

```
npm run build
```

### Server

Install the required dependencies (it is recommended to first create a virtual environment)

```
pip install -r requirements.txt
```

Start the server (runs at localhost:5000)

```
python server/app.py
```
