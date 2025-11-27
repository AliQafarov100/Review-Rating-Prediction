# Review Rating Prediction 
A multi-class sentiment analysis project for predicting rating from the customer review from *Yelp review dataset*.
In this project we evaluate performance of *BERT* transformer model. Specifically, we're compared customer *BERT* built from scratch and pre-trained *BERT*.

---

##  Dataset
We experiment our models in *yelp_review_full* dataset from Hugging Face datasets. The dataset contain in generally 700K rows of data from which 650K are training examples, and remaining 50K are test examples.
Dataset contain next rows:
 - **text** -- customer reviews text
 - **label** -- corresponds to the score associated with review (between 1 and 5)

Goal: Predict rating based on customer review.

## Packages Used

  This project uses the following Python packages:

  - **torch** – for building neural networks, model training, and tensor operations
  - **transformers** - HuggingFace transformers for importing appropriate pre-trained models, tokenizer
  - **scikit-learn (sklearn)** – for evaluation metrics, and preprocessing
  - **numpy** – for numerical computations and array manipulation
  - **pandas** – for data loading, cleaning, and analysis
  - **matplotlib & seaborn** – for data visualization and exploratory analysis

## Results
In this project we compared the results of *BERT* model from scratch and pre-trained fine-tuned *BERT* from HuggingFace transformers. 
Because of memory issue and computational resources constraint was choosen BERTbase which have 110M parameters. 
Models were trained on NVIDIA L4 GPU, number of epochs set as 3 and learning rate 2e-5 with linear weight decay which equal to 0.01, other hyperparameters also was choosen from the original paper.
Results of the models we can see in the table below:

| Model | Number of parameters | Accuracy | Notes |
|-------|----------------|----------|--------|
| **Pre-trained BERT** | 110M | **68%** | Best performing model |
| **BERT from scratch** | 110M | **58%** | A quite worse than pre-trained |

That is not end of the project, we'll continiue try to experiment with new technologies and models to get higher result!
