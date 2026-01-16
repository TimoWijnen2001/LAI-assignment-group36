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
│   ├── birth_year.csv <br>
│   ├── extrovert_introvert.csv
│   ├── feeling_thinking.csv
│   ├── gender.csv
│   ├── judging_perceiving.csv
│   ├── nationality.csv
│   ├── political_leaning.csv
│   ├── sensing_intuitive.csv
├── .gitignore
├── bert.py
├── data_exploration.py
├── preprocessing.py
├── README.md
├── requirements.txt
├── tfidf.py
└── WeightedLossTrainer.py  

The `data` folder will not be on this github page, as it is too large. It can either be downloaded via the link of the lecturer, or from this google drive folder: https://drive.google.com/file/d/142hV1UqWLTu3GfNeGo7xhj9CschjSh_G/view?usp=sharing

---

## Running the code

Once the directory matches the structure above, first, `preprocessing.py` should be run. This will create a `cleaned_data` folder with the preprocessed data. 
Then any of the other files can be run in any order, and the results will be returned in the terminal. 
Note that `WeightedLossTrainer.py` does not return anything. This is a helper class.

---
