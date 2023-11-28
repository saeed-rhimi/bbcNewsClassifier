
# BBC News Classifier Project

## Project Overview
This project aims to classify news articles into different categories using machine learning techniques. It is based on the BBC news dataset and utilizes text processing and classification algorithms to achieve accurate categorization of news articles.

## Dependencies
To run this project, you need to install the following Python libraries:
- pandas
- numpy
- scikit-learn
- nltk

You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn nltk
```

Additionally, make sure to download the necessary NLTK datasets using the following commands in a Python environment:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Dataset
The dataset used in this project is the 'BBC News Dataset', which includes news articles categorized into several types. It has been preprocessed to remove stop words, non-alphabetic characters, and lemmatization techniques have been applied. The dataset is split into training and testing sets for model training and evaluation.

## Model Description
This project uses a Multinomial Naive Bayes classifier, which is a popular choice for text classification tasks. The model is trained using a TF-IDF vectorized representation of the text data.

## Usage Instructions
1. Clone the repository.
2. Install the necessary dependencies as listed in the 'Dependencies' section.
3. Run the Jupyter Notebook `main.ipynb` to train the model and evaluate its performance.

