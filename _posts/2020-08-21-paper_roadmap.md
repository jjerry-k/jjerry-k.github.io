---
layout: post
title: Paper Reading Roadmap
category: [DeepLearning]
tags: [Paper]
sitemap :
changefreq : daily
---

> 본 포스팅은 [고려대학교 산업경영공학부 Data Science & Business Analytics 연구실](http://dsba.korea.ac.kr)의 [강필성 교수님의 자료](https://www.facebook.com/groups/TensorFlowKR/permalink/1275047779502944/)를 정리한 포스팅입니다. 

---

# Contents of Posting
- [Paper Reading Roadmap](#paper-reading-roadmap)
  - [ML Basics](#ml-basics)
  - [Data Mining](#data-mining)
    - [General](#general)
    - [Patter Mining](#patter-mining)
    - [Clustering](#clustering)
  - [Artificial Intelligence](#artificial-intelligence)
    - [General](#general-1)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Transfer Learning](#transfer-learning)
  - [Supervised Learning](#supervised-learning)
    - [Kernel Machines](#kernel-machines)
    - [Ensemble](#ensemble)
  - [Semi-supervvised Learning](#semi-supervvised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Neural Network](#neural-network)
    - [General](#general-2)
    - [Structure](#structure)
    - [Learning Strategies](#learning-strategies)
  - [NLP](#nlp)
    - [General](#general-3)
    - [Topic Modeling](#topic-modeling)
    - [Repersentation Learning](#repersentation-learning)
    - [Classification](#classification)
    - [Summarization](#summarization)
    - [Machine Translation](#machine-translation)
    - [Question Answering](#question-answering)
  - [Vision](#vision)
    - [Classification](#classification-1)
    - [Object Detection](#object-detection)
    - [Localization & Segmentation](#localization--segmentation)

# Paper Reading Roadmap

## ML Basics
- The matrix calculus you need for deep learning 
- Statistical Modeling: The Two Cultures 
- Machine learning: Trends, perspectives, and prospects 
- An introduction to ROC analysis 
- Learning from imbalanced data 
- Variational inference: A review for statisticians 
- The expectation-maximization algorithm 
- Dimension Reduction: A Guided Tour

## Data Mining

### General
- [Interestingness Measures for Data Mining: A Survey](https://dl.acm.org/doi/10.1145/1132960.1132963)
- [The PageRank citation ranking: Bringing order to the web](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)
- [Process Mining Manifesto](https://link.springer.com/content/pdf/10.1007/978-3-642-28108-2_19.pdf)
- [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)

### Patter Mining
- [Fast Algorithm for Mining Association Rules](https://web.stanford.edu/class/cs345d-01/rl/ar-mining.pdf)
- [A survey of sequential pattern mining](https://dl.acm.org/doi/10.1145/3314107)
- [A Survey of Parallel Sequential Pattern Mining](https://arxiv.org/pdf/1805.10515.pdf)

### Clustering
- [A density-based algorithm for discovering clusters in large spatial databases with noise](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [Data Clustering: A Review](http://users.eecs.northwestern.edu/~yingliu/datamining_papers/survey.pdf)
- [Techniques of Cluster Algorithms in Data Mining](https://cs.nju.edu.cn/zhouzh/zhouzh.files/course/dm/reading/reading06/grabmeier_dmkd02.pdf)
- [Survey of Clustering Data Mining Techniques](https://www.cc.gatech.edu/~isbell/reading/papers/berkhin02survey.pdf)
- [On Clustering Validation Techniques](https://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf)
- [clValid: An R Package for Cluster Validation](https://cran.r-project.org/web/packages/clValid/vignettes/clValid.pdf)
  
## Artificial Intelligence

### General
- [Learning Deep Architectures for AI](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
- [Representation learning: A review and new perspectives](https://arxiv.org/pdf/1206.5538.pdf)
- [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- [From evolutionary computation to the evolution of things](https://www.nature.com/articles/nature14544)
- [Probabilistic machine learning and artificial intelligence](https://www.nature.com/articles/nature14541)
- [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/pdf/1908.00709.pdf)

### Reinforcement Learning
- [Human-level control through deep reinforcement](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
- [Mastering the game of Go with deep neural networks and tree search](https://deepmind.com/research/publications/mastering-game-go-deep-neural-networks-tree-search)
- [An Introduction to Deep Reinforcement Learning](https://arxiv.org/pdf/1811.12560.pdf)
- [World Models](https://arxiv.org/pdf/1803.10122.pdf)

### Transfer Learning
- [Zero-shot learning through cross-modal transfer](https://papers.nips.cc/paper/5027-zero-shot-learning-through-cross-modal-transfer.pdf)
- [Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/pdf/1708.01547.pdf)

## Supervised Learning

### Kernel Machines
- An Introduction to Kernel-based Learning Algorithms 
- A Tutorial on Support Vector Machine for Pattern Recognition 
- A Tutorial on Support Vector Regression 
- A Tutorial on nu-Support Vector Machines

### Ensemble
- Bagging Predictors 
- Random Forests 
- A short introduction to boosting 
- Greedy Function Approximation: A Gradient Boosting Machine 
- Gradient Boosting Machine, A Tutorial 
- XGBoost: A Scalable Tree Boosting System 
- LightGBM: A Highly Efficient Gradient Boosting Decision Tree 
- CatBoost : unbiased boosting with categorical features

## Semi-supervvised Learning
- Combining Labeled and Unlabeled Data with Co-Training 
- Semi-supervised Learning with Deep Generative Models 
- Semi-Supervised Classification with Graph Convolutional Networks 
- MixMatch: A Holistic Approach to Semi-Supervised Learning 
- ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring 
- FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

## Unsupervised Learning
- Anomaly Detection: A Survey 
- Deep Learning for Anomaly Detection: A Survey 
- A Review of Novelty Detection 
- LOF: Identifying Density-Based Local Outliers 
- Support Vector Data Description 
- Isolation Forest 
- Isolation-based Anomaly Detection 
- DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning
  
## Neural Network

### General
- Deep learning

### Structure
- Long Short-Term Memory
- LSTM: A Search Space Odyssey
- Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling Sequence to sequence learning with neural networks
- Memory Networks
- End-To-End Memory Networks
- WaveNet: A Generative Model for Raw Audio
- An Introduction to Variational Autoencoders
- A Comprehensive Survey on Graph Neural Networks

### Learning Strategies
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
- Dropout: A Simple Way to Prevent Neural Networks from Overtting
- ADAM: A Method for Stochastic Optimization
- An overview of gradient descent optimization algorithms
- Layer normalization Group normalization

## NLP

### General
- Natural Language Processing (Almost) from Scratch
- Advances in natural language processing
- Recent trends in deep learning based natural language processing

### Topic Modeling
- An introduction to latent semantic analysis 
- Probabilistic latent semantic analysis 
- Probabilistic topic models
- Latent Dirichlet Allocation

### Repersentation Learning
- A Neural Probabilistic Language Model
- Distributed representations of words and phrases and their compositionality
- Efficient Estimation of Word Representations in Vector Space
- Glove: Global vectors for word representation
- Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation Enriching word vectors with subword information
- Bert: Pre-training of deep bidirectional transformers for language understanding
- Deep contextualized word representations
- Improving language understanding by generative pre-training
- Language models are unsupervised multitask learners
- Language Models are Few-Shot LearnersA Neural Probabilistic Language Model

### Classification
- Convolutional neural networks for sentence classification 
- Deep learning for sentiment analysis: A survey

### Summarization
- TextRank: Bringing Order into Texts
- A Neural Attention Model for Abstractive Sentence Summarization


### Machine Translation
- On the Properties of Neural Machine Translation: Encoder-Decoder Approaches
- Effective Approaches to Attention-based Neural Machine Translation
- Neural Machine Translation by Jointly Learning to Aligh and Translate
- Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation 
- Attention is all you need

### Question Answering
- VQA: Visual Question Answering
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing 
- Squad: 100,000+ questions for machine comprehension of text
- Know what you don't know: Unanswerable questions for SQuAD

## Vision

### Classification
- [Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Visualizing and understanding convolutional networks](https://arxiv.org/pdf/1311.2901.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
- [Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

### Object Detection
- [Overfeat: Integrated recognition, localization and detection using convolutional networks](https://arxiv.org/pdf/1312.6229.pdf) 
- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
- [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

### Localization & Segmentation
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [Learning deep features for discriminative localization](https://arxiv.org/pdf/1512.04150.pdf)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)