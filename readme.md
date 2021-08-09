Machine Learning

- anaconda
- data analysis
- pandas, numpy
- data visualization matplotlib
- scikit-learn machine learning, models
- supervised learning ( classication, regression )
- neural networks ( transfer learning, deep learning, time series)
- tensor flow and keras - image classification
- data engineering - spark / hadoop
- realtime and batch
- storytelling and communication

 What is Machine Learning?

 machine learn to do tasks for us

 AI - machine acts like a human neuro ai 
 machine learning subset of ai - how to achieve AI 
 getting computers to act without explicitly code without if else

 deep learning subset of machine learning - one of techniques
 to implement machine learning

 Data science - analysing data and do something with it

 Data Engineering


 Teachable machine
 https://teachablemachine.withgoogle.com/



 people look on spreadsheet
 machine learning look on big data nosql unstructured data

Very important pipeline
-----------------------------------

 Data collection ( this is the hardest part )
 Data modelling -> problem definition -> data -> eval -> features -> modelling -> experiments -> iterative process -> redefine problem
 =
 Deployment 

 https://ml-playground.com/#



 K nearest neighbors
 perceptron
 support vector machine
 artificial neural network
 Decision tree

 Machine learning predicting results based on incoming data
                                                    rewards punish
labels               no labels(csv)                 try and error
 supervised          unsupervised                   reinforcement

classification       clustering                     skill acquisi
regression           association rule learning      real time learn



Machine learning is using an algorithm / computer program to 
learn about different patterns in data and make predictions 
regarding future similar data


Input output

Machine learning models

chicken recipe


Problem definition
------------------

when not to use machine learning

- will a simple hand-coded instruction based system work?

when to use machine learning

- supervised learning
samples data and labels
i know my inputs and outputs
classification - is one thing or another ( binary , multi-class)
regression - a number will go up or down sell price of a house
- unsupervised learning 
I am not sure of the outputs but I have inputs
samples data but no labels 
clustering -> you provide labels, putting samples into groups
- transfer learning
i think that my problem may be similar to something else
pass knowledge from model to another
- reinforcement learning
rewarding and punishing, update a score, maximise the score

Matching your problem

Types of data
----------------

The more data the better
- structured -> excel csv
- unstructured - images, audio files, natural language texts

Streaming  - based on news change in stock
data-> jupyter notebooks-> pandas-> matplotlib-> skikit learn

Types of evaluation
--------------------

what defines success for us

machine learning model with % accuracy

Classification 
- accuracy
- precision
- recall

Regression
- Mean absolute error
- Mean squared  error
- Root mean squared error

Recommendation
- Precision at K

Features in data
----------------

feature is another word for different forms of data
feature variables to predict the target variable

heart disease
weight, sex, heart rate , chest pain
numerical or categorical features

derived feature -> visit in the last year for example

feature engineering

feature coverage -> ideally every sample has the same features
it may be possible for some samples to miss some features ( columns )

Modelling
-------------

Part 1 - 3 sets based on our problem and data what machine learning model should we use  - splitting data 

Choosing and training a model
tuning a model
model comparison

The training, validation(tuning) and test set

Split data into 3 different set

Course materials ( training set ) 70-80%
Practice exam ( validation set )  10-15%
Final exam ( test set ) 10-15%


1. Choosing a model?

Structured data -> CatBoost, dmlcXGBoost, Random Forest
Unstructured data -> Deep learning, Transfer learning



based on training data

2. Tuning a model 
Validation data or on training data

Random forest
Neural networks layers
models have hyperparameters you can adjust - this is tuning


3. Model comparison during experimentation
done on test data -- happens only on the testing set

how models perform in production



Data set | Performance ( underfitting )
----------------------
Training   64 %
Test       47 %     

Dat set | Performance ( overfitting )
-----------------------
Training  93%
Test      99%


Overfitting - data leakage
Underfitting - data mismatch, not same features

Solution for underfitting
--------------------------
- try a more advanced model
- increase model hyperparameters
- reduce amount of features
- traing longer

Solution for overfitting
------------------------
- collect more data
- try a less advanced model
- remember that no model is perfect



Experimentation
-------------

Data analysis - data/evaluation and features
Machine learn modelling
Experimenting


Data analysis - pandas, matplotlib, numpy
Data modelling - tensorflow, Pytorch, skikitlearn, dmlcXGBoost, CatBoost
Jupyter Notebooks, Anaconda

which tool to use for what kind of problem!! not knowing everything 
by heart

Anaconda
------------------
anaconda-navigator

Anaconda and miniconda - hardware store and work bench for machine learning
conda - data scientist personal assistant comes with miniconda

conda is like a package manager


conda create --prefix ./anacondaenv pandas numpy matplotlib scikit-learn jupyter


anaconda shell
--------------

conda activate /Users/emanueldumitru/env
conda deactivate
conda env list
conda install jupyter
jupyter notebook


conda create -n py33
















