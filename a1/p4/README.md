* run `enwik9_cleaner` to creater corpus
* run `a1_p4.py` to train model on corpus
* run `tensorboard --logdir=processed --port=6006` to view embeddings


Some new challenges I faced/things I learned:
 * [not enough RAM to keep data all in memory][1]
 * [word tokenizers tend to need prior sentence tokenization (see my comment to alvas's answer)][2]

[1]: https://stackoverflow.com/questions/45340148/python-memory-usage-txt-file-much-smaller-than-python-list-containing-file-text
[2]: https://stackoverflow.com/questions/45339229/nltk-word-tokenize-why-do-sentence-tokenization-before-word-tokenization
