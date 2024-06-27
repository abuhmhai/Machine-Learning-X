## Fake News Detection Model

### Project Overview
This project aims to build and compare the performance of various machine learning models to detect fake news. We employ traditional machine learning algorithms (Decision Tree, AdaBoost, Random Forest, Logistic Regression) as well as a neural network model. The models are trained and evaluated on a dataset containing labeled fake and real news articles.

### Table of Contents
1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [Models](#models)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)


### Dataset
The dataset used for this project not consists of labeled news for each data. Each article will be label as either fake or real. The dataset is preprocessed to remove punctuation, convert text to lowercase, and vectorize the text using TF-IDF.

### Preprocessing
1. **Load the dataset**: The dataset is loaded into a pandas DataFrame.
2. **Clean the text**: Convert text to lowercase and remove punctuation and numbers.
3. **Vectorize the text**: Convert text data to numerical data using TF-IDF vectorization.

### Models
We compare the performance of the following models:

#### Traditional Machine Learning Models
1. **Decision Tree**: A simple decision tree classifier.
2. **AdaBoost**: An ensemble method that uses multiple decision trees.
3. **Random Forest**: An ensemble of decision trees to improve performance.
4. **Logistic Regression**: A linear model for binary classification.

#### Neural Network
A simple neural network model with the following architecture:
- Input layer with the dimension equal to the number of features.
- Two hidden layers with ReLU activation and Dropout for regularization.
- Output layer with a sigmoid activation for binary classification.

### Evaluation
We evaluate the models using the following metrics:
1. **Accuracy**: The proportion of correctly classified instances.
2. **Confusion Matrix**: A matrix to visualize the performance of the classification.
3. **Classification Report**: A report showing precision, recall, and F1-score.
4. **ROC Curve**: A plot to visualize the performance of the classification model.


### Results
The results of the evaluation are compared to determine the best-performing model for fake news detection. The performance of each model is summarized and visualized using accuracy scores, confusion matrices, and ROC curves.

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### License
This project is licensed under the MIT License.

---
