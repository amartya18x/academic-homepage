+++
abstract = "We investigate two causes for adversarial vulnerability in deep neural networks: bad data and (poorly) trained models. When trained with SGD, deep neural networks essentially achieve zero training error, even in the presence of label noise, while also exhibiting good generalization on natural test data, something referred to as benign overfitting. However, these models are vulnerable to adversarial attacks. We identify label noise as one of the causes for adversarial vulnerability, and provide theoretical and empirical evidence in support of this. Surprisingly, we find several instances of label noise in datasets such as MNIST and CIFAR, and that robustly trained models incur training error on some of these, i.e. they don't fit the noise. However, removing noisy labels alone does not suffice to achieve adversarial robustness. Standard training procedures bias neural networks towards learning 'simple' classification boundaries, which may be less robust than more complex ones. We observe that adversarial training does produce more complex decision boundaries. We conjecture that in part the need for complex decision boundaries arises from sub-optimal representation learning. By means of simple toy examples, we show theoretically how the choice of representation can drastically affect adversarial robustness." 
authors = ["Pau De Jorge", "Amartya Sanyal", "Harkirat Behl",  "Philip H.S. Torr", "Gregory Rogez", "Puneet K. Dokania"]
date = "2020-06-29"
math = true
draft = "false"
publication_types = ["3"]
publication = "In submission"
featured = true
title = "Progressive Skeletonization: Trimming more fat from a network at initialization"
url_pdf = "https://arxiv.org/pdf/2006.09081"
url_preprint = "https://arxiv.org/abs/2006.09081"
url_project="https://europe.naverlabs.com/blog/force-enables-extreme-pruning-of-artificial-neural-networks-at-initialization-a-new-method-called-force-achieves-extreme-sparsity-in-artificial/"
+++