# SafeSpace

**SafeSpace** is end to end content moderation tool, using an efficient ML regression tool built on a dataset that was scraped, cleaned and processed from scratch, and has been implemented on Discord.

- Checkout [this video](https://todo) for a working demo. 

So, go ahead! Use SafeSpace to make your chat group a safe and welcoming place for everyone.

### Introduction

Equity is one of the core goals of humanity as a whole. Equity involves everyone in the decision-making process, and the most equal world is one where everyone understands each other. This however, is not possible unless there is a safe space for ideas to flow.

The flaw with present day moderation system is that it is word dependent, as in, it moderates explicit words. However, use of f-words or such do not inherently make a conversation uncomfortable. The context around a word matters more than what it says exactly. This should also include innuendos, which don't use any vulgarity at all, on the surface.

Thus, we've used a Support Vector Classifier on a dataset generated from scratch, by scraping and cleaning a website with sexual content.

### About Discord

Among the several instant messaging platforms available, **Discord** is one of the most popular ones. Because of its several innovative features like server-channel systems, awesome call quality, permission management and tools to integrate bots, Discord has become a major platform for people to collaborate, converse and share ideas.

### Technologies Involved

- Python 3 and associated libraries (scikit-learn, nltk, pandas etc)
- `Discord.py` for functionality of discord bot

### Dataset

http://textfiles.com/sex/EROTICA/ this was used to scrape from, as a general purpose text corpus. We used multithreading in wget to speed up the download, and then cleaned the entire data, putting all content in UTF-8 encoding and stripping punctuations, newlines and multiple small changes.

## Machine Learning

We first use the Bag-Of-Words Approach from scikit-learn and apply the TF-IDF Statistic to convert the textual input into Meaningful Machine Readable Vectors. After that, we compared the results with 3 different Classification Algorithms: Logistic Regression, Support Vector Classifiers and Gaussian flavor of the Naive Bayes Classifier. Logistic Regression gave us the best F1 score, and we have thus used it for the Sexual Content Moderation Bot.

Here is our classification matrix. 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/GJQ5C7v/unknown.png" alt="unknown" border="0"></a>
