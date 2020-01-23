+++
abstract = "A key feature of neural networks, particularly deep convolutional neural networks, is their ability to 'learn' useful representations from data. The very last layer of a neural network is then simply a linear model trained on these 'learned' representations. Despite their numerous applications in other tasks such as classification, retrieval, clustering etc., a.k.a. transfer learning, not much work has been published that investigates the structure of these representations or whether structure can be imposed on them during the training process. In this paper, we study the dimensionality of the learned representations by models that have proved highly succesful for image classification. We focus on ResNet-18, ResNet-50 and VGG-19 and observe that when trained on CIFAR10 or CIFAR100 datasets, the learned representations exhibit a fairly low rank structure. We propose a modification to the training procedure, which further encourages low rank representations of activations at various stages in the neural network. Empirically, we show that this has implications for compression and robustness to adversarial examples."
authors = ["Amartya Sanyal", "Varun Kanade", "Philip H.S. Torr"]
date = "2018-04-01"
math = true
draft = "false"
publication_types = ["3"]
publication = "In submission"
featured = true
title = "Learning Low Rank Representations"
url_pdf = "https://arxiv.org/pdf/1804.07090"
url_preprint = "https://arxiv.org/abs/1804.07090"
+++