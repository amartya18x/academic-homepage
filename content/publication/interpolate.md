+++
abstract = "When trained with SGD, deep neural networks essentially achieve zero training error, even in the presence of label noise, while also exhibiting good generalization on natural test data, something referred to as benign overfitting [2, 8]. However, these models are vulnerable to adversarial attacks. We identify label noise as one of the causes for adversarial vulnerability, and provide theoretical and empirical evidence in support of this. Surprisingly, we find several instances of label noise in datasets such as MNIST and CIFAR, and that robustly trained models incur training error on some of these, i.e. they don’t fit the noise. We believe this highlights the importance of removing label noise from dataset as well as protecting the integrity of the dataset curation process.By means of simple theoretical setups, we show how the choice of representation can drastically affect adversarial robustness. We also provide some experimental evidence how incorporating better inductive biases can help improve robustness." 
authors = ["Amartya Sanyal", "Varun Kanade", "Puneet K. Dokania", "Philip H.S. Torr"]
date = "2020-12-01"
math = true
draft = "false"
publication_types = ["3"]
publication = "Accepted at NeurIPS '20 Workshop on Dataset Curation and Security"
featured = false
title = "Interpolating noisy datasets hurts adversarial robustness "
url_pdf = "http://securedata.lol/camera_ready/15.pdf"
url_project = "http://securedata.lol/"
+++