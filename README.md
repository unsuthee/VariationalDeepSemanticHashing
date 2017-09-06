# Variational Deep Semantic Hashing (SIGIR'2017)
The implementation of the models and experiments of [Variational Deep Semantic Hashing](http://students.engr.scu.edu/~schaidar/paper/Variational_Deep_Hashing_for_Text_Documents.pdf) (SIGIR 2017).

Author: Suthee Chaidaroon

# Prepare dataset
The model expects the input document to be in a bag-of-words format. I provided sample dataset under dataset directory. If you want to use a new text collection, the input document collection to our model should be a matrix where each row represents one document and each column represents one unique word in the corpus. 

# To get the best performance
TFIDF turns out to be the best representation for our models according to our empirical results.

# Training the model
The component collapsing is common in variational autoencoder framework where the KL regularizer shuts off some latent dimensions (by setting the weights to zero). We use weight annealing technique [1] to mitigate this issue during the training. 

# References
[1] https://arxiv.org/abs/1602.02282
