NY Times Articles


An Analysis on Articles and Resultant Comments
  An analysis of engagement
 
 Kaggle New York Times Data
- https://www.kaggle.com/benjaminawd/new-york-times-articles-comments-2020

Important Links

NY Times Developer Instructions
- https://developer.nytimes.com/get-started

    Create an Account
    If you don't already have an account create one:

    Click Sign In.
    Click Create account and follow prompts to register.

    Sign In
    To sign in to the portal:

    Click Sign In.
    Enter your email address and password.
    
    
    Click Sign In.
    Register apps
    To register an app:

    Select My Apps from the user drop-down.
    Click + New App to create a new app.
    Enter a name and description for the app in the New App dialog.
    Click Create.
    Click the APIs tab.
    Click the access toggle to enable or disable access to an API product from the app.
    Access the API keys
    Select My Apps from th user drop-down.
    Click the app in the list.
    View the API key on the App Details tab.
    Confirm that the status of the API key is Approved.
    APIs
    The APIs page has information on the different APIs. The documentation for each API includes an interactive reference for trying out the API.

NY Times Scraper (originated the data)
- https://github.com/ietz/nytimes-scraper



# Workflow

## What business problems could we solve?
  - What drives high engagement (number of comments)
  - Result to Sell ads at a higher price

## Preparation/Preprocessing/Feature
    - Corpus is the column
    - Could be the entire document
  
    - Cleanup Text / Preprocessing
      - Tokenize /Remove punctuation, symbols
        - chop up text into pieces and throw out punctuation
          - RegexpTokenizer
      - Normalization - transformation to root form
        - **Lemma**tization - dictionary form*- morphological analysis - context, lexical library..generally faster
          - Part of speech tagging
        - **Stem**ming Lowercase - crude, predefined rules
      - Stopwords - remove low value terms
        - Use nltk stopwords, then add your own based on domain of text
          - Consider adding New York Times, newspaper, NYT, article
      ?# t?
      - Count Vectorization
        - Token Counts Matrix

      - TF-IDF Vectorization
        - sklearn
        - Does TF_IDF apply to all comments or only comments of its own type
    
      - EDA
      - Word Cloud
        - per topic
## Feature Engineering
    - Vectorization
      - LDA (requires bag of words)

    - Cosine Similarity
    - Reduce Dimensions
    - Similar Headlines

     - Topic Modeling
      - LDA - Latent Dirichlet allocation extract categories of words

    -  N-GRAMS
    - Build a pipeline
    - **Named Entity Recognition - people places things**
    - Liguistic Feature Extraction

## Dimension Reduction

## EDA 2
    - EDA: Frequency Timeline
    - EDA: Frequency Heatmaps

## Supervised 
- Text Classification
  - Supervised Learning
  - SVM
  - Lime - Local Interpretable Model-Agnostic Explanations (Anchor)
    - Used to Explain Classification Results
  - Named Entity R
  - XGBoost


## Unsupervised


## Other Models
- Stochastic Gradient Descent Classifier?
- SVM
- NaiveBayes
- Logistic



Questions to Answer
-   What drives high number of comments?
    -   Is it the nouns and the verbs, what makes this happen?
    -   Is it the topic and content?
-   Predict n comments in articles
-   Predict editorsSelection in Comments
-   Predict number of replies

## OTHER BUSINESS NOTES
- Noun counts per verbs
  - How many counts words
  - A given article has lots of comments
  - Low Medium High
  - WHAT DRIVES ENGAGEMENT

Avoid
    - Deep Learning


https://www.kdnuggets.com/2017/11/framework-approaching-textual-data-tasks.html

Data Collection or Assembly
- Build Corpus, entire set of everything
  - Is that set everything? Or just the set of classified topics?

Data Preprocessing
- tokenization, normalization, substitution
  
Data Exploration and Visualization
- word clouds
- word counts and distributions
- generating wordclouds
- performing distance measures

Model Building
- Finite state machines, Markov models, vector space modeling of word meanings
- Classifiers, Naive Bayes, Logistic Regression, Decision Trees, SVM,neural networks
- 


Model Evaluation


