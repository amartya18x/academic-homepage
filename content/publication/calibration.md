+++
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title = "Calibrating Deep Neural Networks using Focal Loss"
summary= "Miscalibration -- a mismatch between a model's confidence and its correctness -- of Deep Neural Networks (DNNs) makes their predictions hard to rely on. Ideally, we want networks to be accurate, calibrated and confident. We show that, as opposed to the standard cross-entropy loss, focal loss (Lin et al., 2017) allows us to learn models that are already very well calibrated. When combined with temperature scaling, whilst preserving accuracy, it yields state-of-the-art calibrated models. We provide a thorough analysis of the factors causing miscalibration, and use the insights we glean from this to justify the empirically excellent performance of focal loss. To facilitate the use of focal loss in practice, we also provide a principled approach to automatically select the hyperparameter involved in the loss function. We perform extensive experiments on a variety of computer vision and NLP datasets, and with a wide variety of network architectures, and show that our approach achieves state-of-the-art accuracy and calibration in almost all cases."
authors= ["Jishnu Mukhoti", "Viveka Kulharia", "Amartya Sanyal", "Stuart Golodetz", "Philip H.S. Torr", "Puneet K. Dokania"]
math = true
draft = "false"
publication_types = ["3"]
publication = "Workshop on Uncertainty and Robustness in Deep Learnning , ICML 2020. (Spotlight Paper) "
date= 2020-02-21
featured= false

url_pdf = "https://arxiv.org/pdf/2002.09437"
url_preprint = "https://arxiv.org/abs/2002.09437"
url_project=""
+++
