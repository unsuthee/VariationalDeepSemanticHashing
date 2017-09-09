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

# Bibtex
```
@inproceedings{Chaidaroon:2017:VDS:3077136.3080816,
 author = {Chaidaroon, Suthee and Fang, Yi},
 title = {Variational Deep Semantic Hashing for Text Documents},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {75--84},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3077136.3080816},
 doi = {10.1145/3077136.3080816},
 acmid = {3080816},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, semantic hashing, variational autoencoder},
}
```
