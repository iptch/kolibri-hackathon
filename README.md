# kolihack

## About
Repository for the Mai 2023 hackathon, specifically for the task to improve search.

## Installation
### Prerequisites
- [pyenv](https://github.com/pyenv/pyenv) - to manage your local python versions like a boss, easy to install via [pyenv-installer](https://github.com/pyenv/pyenv-installer) (check common build problems if issues arise)
- [pipenv](https://github.com/pypa/pipenv) - to manage dependencies in virtual environments with joy

### Create pipenv environment and install dependencies
1. Clone the repo
```sh
git clone git@github.com:iptch/kolihack.git
```
2. Install python dependencies via `pipenv` (check prerequisites before)
```sh
cd kolihack
pipenv install
```
3. Active virtual environment
```sh
pipenv shell
```

### Getting started
#### Getting the data ready
- Copy the file `content.csv` which you downloaded from the [kaggle competition website](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data)
  into the folder `data`. That means you end up with a file called `content.csv` inside the `data` folder at the root of this repo. Note: You can also get it from the ipt oss4good GDrive when you don't have a Kaggle account and don't want to register.
- Note: `data/content.csv` will be ignored by git and not checked in into the repo.

#### Running the notebooks
There is a **notebooks** folder holding all the jupyter notebooks. Go run them:  
Having the pipenv environment activated by running
```
pipenv shell
```
you can then run the jupyter lab with
```
jupyter lab
```

#### Python code
General python code (e.g. `io.py` for data loading and storing) should preferably be located in `kolihack`
such that we don't mess up our notebooks. 
