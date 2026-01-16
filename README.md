# Demographic prediction with and without in-text mentions

This project investigates the effect of in-text mentions of demographic and personality traits on demographic prediction models such as TF-IDF with logistic regression and XLM-RoBERTa.
It includes preprocessing, model training of both models, and the output of evaluation metrics.

---

## Environment Setup

- Python version: 3.12.10  
- All dependencies are listed in `requirements.txt`.
- Install CUDA for faster model training.

---

## Required Files and Directory Setup

To use the code, the directory should look like this:

├── data<br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── birth_year.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── extrovert_introvert.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── feeling_thinking.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── gender.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── judging_perceiving.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── nationality.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── political_leaning.csv <br>
│  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── sensing_intuitive.csv <br>
├── .gitignore <br>
├── bert.py <br>
├── data_exploration.py <br>
├── preprocessing.py <br>
├── README.md <br>
├── requirements.txt <br>
├── tfidf.py <br>
└── WeightedLossTrainer.py <br>

The `data` folder will not be on this github page, as it is too large. It can either be downloaded via the link of the lecturer, or from this google drive folder: https://drive.google.com/file/d/142hV1UqWLTu3GfNeGo7xhj9CschjSh_G/view?usp=sharing

---

## Running the code

Once the directory matches the structure above, first, `preprocessing.py` should be run. This will create a `cleaned_data` folder with the preprocessed data. 
Then any of the other files can be run in any order, and the results will be returned in the terminal. 
Note that `WeightedLossTrainer.py` does not return anything. This is a helper class.

---
