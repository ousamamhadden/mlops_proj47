# src

### Overall goal of the project
In this project, we aim to fine-tune a text2text model to do grammar correction on English sentences. By applying the MLOps methods and tools taught in the course, we want to create a ML project with a focus on maintainability, shareability, and reproducibility.
### What framework are you going to use and do you intend to include the framework into your project?
We intend to use the transformer framework from HuggingFace. The framework provides pre-trained models that we can fine tune for our specific use-case.
### What data are you going to run on (initially, may change)
We will train our models on a subset of C4 200M Grammar Error Correction dataset ([link](https://www.kaggle.com/datasets/dariocioni/c4200m)).
The reason for training only a subset is to limit the training time of the models and simplify the scope of the project. 
### What models do you expect to use
The idea is to use FLAN-T5, which  is an enhanced version of T5 that has been fine-tuned on a mixture of tasks that include question answering, summarization, translation, and grammatical error correction. T5 is based on the Transformer architecture, which is a type of neural network that has been proven to be highly effective in NLP tasks. T5 uses a Text-to-Text approach. It is pre-trained on a large text corpus (C4). This allows T5 to learn general language knowledge and transfer it to different tasks. T5 is available in different sizes, ranging from small (60 million parameters) to XXL (11 billion parameters). Our task is to further fine-tune FLAN-T5 on grammar error correction to improve its performance.


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
