---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "How Benign is Benign Overfitting ?"
subtitle: ""
summary: "We investigate two causes for adversarial vulnerability in deep neural networks: bad data and (poorly) trained models."
authors: ["Amartya Sanyal", "Puneet K. Dokania", "Varun Kanade", "Philip H.S. Torr"]
tags: []
categories: []
date: 2020-07-08T23:55:20+01:00
lastmod: 2020-07-08T23:55:20+01:00
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

A large body of research has been devoted to crafting defences to protect neural networks from adversarial attacks. However, such defences have usually been broken by future attacks. This arms race between attacks and defences suggests that to create a truly robust model would require a deeper understanding of the source of this vulnerability.
Our goal in this paper is not to propose new defences, but to provide better answers to the question: _What causes adversarial vulnerability?_

In doing so, we also seek to understand how existing methods designed to achieve adversarial robustness overcome some of the hurdles pointed out by our work. We identify two sources of vulnerability that, to the best of our knowledge, have not been properly studied before:
1) Memorization of label noise, and
2) Implicit bias towards simpler decision boundaries of neural networks trained with stochastic gradient descent (SGD).


In short, we make the following theoretical and empirical contributions in the paper.

### Summary of Theoretical Contributions ###


1. We provide simple sufficient conditions on the data distribution under
which any classifier that fits the training data with label noise perfectly is adversarially vulnerable.
 Let $c$ be the target classifier, and let $\mathcal{D}$ be a distribution over
  $\br{\mathbf{x},y}$, such that $y=c\br{\mathbf{x}}$ in its support. Using the
  notation $\mathbb{P}_D[A]$ to denote $\mathbb{P}_{(\mathbf{x}, y) \sim \mathcal{D}}[ \mathbf{x} \in A]$
  for any measurable subset $A \subseteq \mathbb{R}^d$, suppose that there
  exist $c_1 \geq c_2 > 0$, $\rho>0$, and a finite set $\zeta \subset
  \mathbb{R}^d$ satisfying
  %%%
  \begin{equation}
    \label{eq:balls_density}
	 \mathbb{P}_\mathcal{D}\bs{\bigcup_{\vec{s}\in\zeta}\cB_\rho^p\br{\vec{s}}}\ge c_1\quad\text{and}\quad\forall \vec{s}\in\zeta,~\mathbb{P}_\mathcal{D}\bs{\cB_{\rho}^p\br{\vec{s}}}\ge\frac{c_2}{\abs{\zeta}}
  \end{equation}
  where $\cB_\rho^p\br{\vec{s}}$ represents a $\ell_p$-ball of radius $\rho$
  around $\vec{s}$. Further, suppose that each of these balls contain points from a single class
  i.e. for all $\vec{s}\in\zeta$, for all
  $\mathbf{x},\vec{z}\in\cB_{\rho}^p\br{\vec{s}}:
  c\br{\mathbf{x}}=c\br{\vec{z}}$.

  Let $\cS_m$ be a dataset of $m$ i.i.d. samples  drawn from
  $\mathcal{D}$, which subsequently has each label flipped independently with probability $\eta$. 
  For any classifier $f$ that \emph{perfectly} fits the training data $\cS_m$
  i.e. $\forall~\mathbf{x},y\in\cS_m, f\br{\mathbf{x}}=y$, 
  $\forall \delta>0$ and $m\ge\frac{\abs{\zeta}}{\eta
    c_2}\log\br{\frac{\abs{\zeta}}{\delta}}$, with probability at least $1-\delta$, $\radv{2\rho}{f;\cD}\ge c_1$.
2. The choice of the representation~(and hence the shape of the decision
		boundary) can be important for adversarial accuracy even when it doesn't
		affect natural test accuracy.
3.  There exists data distributions and training algorithms, which when
		trained with (some fraction of) random label noise have the following
		property:
    1. Using one representation, it is possible to have high
		natural and robust test accuracies but at the cost of having training error;
    2. Using another representation, it is possible to have no training
		error (including fitting noise) and high test accuracy, but low robust
		accuracy. Furthermore, any classifier that has no training error must
		have low robust accuracy.

The last example shows that the choice of representation matters significantly when it comes to adversarial accuracy, and that memorizing label noise directly leads to loss of robust accuracy. The proofs for the results are not technically complicated, but we skip them in this blog for brevity.

### Summary of Experimental Contributions ###
1. As predicted theoretically, neural nets trained to convergence with
		label noise have greater adversarial vulnerability.
2. Robust training methods, such as \AT and TRADES that have higher
		robust accuracy, avoid overfitting (some) label noise. This behaviour is
		also partly responsible for their decrease in natural test accuracy.
3. Even in the absence of any label noise, methods like \AT and TRADES
		have higher robust accuracy due to more complex decision
		boundaries.
4.When trained with more fine-grained labels, subclasses
          within each class, leads to higher robust accuracy.



Our theoretical 


{{<figure src="/benign/mislabelled.png">}}

Surprisingly, as shown above, we find several instances of label noise in datasets such as MNIST and CIFAR, and that robustly trained models incur training error on these mislabelled points, i.e. they don’t fit the noise. 