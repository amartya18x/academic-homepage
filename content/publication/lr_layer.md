+++
abstract = "We investigate the effect of the dimensionality of the representations learned in Deep Neural Networks (DNNs) on their robustness to input perturbations, both adversarial and random. To achieve low dimensionality of learned representations, we propose an easy-to-use, end-to-end trainable, low-rank regularizer (LR) that can be applied to any intermediate layer representation of a DNN. This regularizer forces the feature representations to (mostly) lie in a low-dimensional linear subspace. We perform a wide range of experiments that demonstrate that the LR indeed induces low rank on the representations, while providing modest improvements to accuracy as an added benefit. Furthermore, the learned features make the trained model significantly more robust to input perturbations such as Gaussian and adversarial noise (even without adversarial training). Lastly, the low-dimensionality means that the learned features are highly compressible; thus discriminative features of the data can be stored using very little memory. Our experiments indicate that models trained using the LR learn robust classifiers by discovering subspaces that avoid non-robust features. Algorithmically, the LR is scalable, generic, and straightforward to implement into existing deep learning frameworks."
authors = ["Amartya Sanyal", "Varun Kanade", "Puneet K. Dokania", "Philip H.S. Torr"]
date = "2020-01-01"
math = true
draft = "false"
publication_types = ["3"]
publication = "Accepted at ICML 2018 Workshop on Theoretical Foundations and Applications  of Deep Generative Models"
featured = true
title = "Robustness via Deep Low Rank Representations"
url_pdf = "https://arxiv.org/pdf/1804.07090v5"
url_preprint = "https://arxiv.org/abs/1804.07090v5"
url_project = "https://amartya18x.github.io/post/lr_layer/"
+++