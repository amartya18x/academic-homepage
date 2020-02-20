---
title: "Robustness via Deep Low-Rank Representations"
subtitle: "Amartya Sanyal, Varun Kanade, Puneet K. Dokania, Philip HS. Torr"
date: 2020-01-23T12:40:23Z
draft: false
image:
---

In this paper ([Full Paper here](https://arxiv.org/abs/1804.07090v5)), we investigate the relation of the _intrinsic_ dimension of the representation space of deep networks with its robustness. To do this, we introduce an easy-to-implement, end-to-end trainable, scalable  regularizer that enforces low rank structure in the representation space of deep networks; we conduct a variety of experiments to show that the resultant network is largely robust to adversarial and random perturbations. In addition, the low _intrinsic_ dimension also means that the representations (and the model) can be compressed significantly without significant loss in accuracy.

### Low Rank Representations

{{% alert hint %}}
For most models trained in a supervised fashion, the vector of
activations in the penultimate layer~(or a layer close to the penultimate layer) is a *learned* representation of the raw input.
{{% /alert %}}


The remarkable success of DNNs is primarily attributed to the discriminative quality of this learned representation space. However, despite their impressive performance, DNNs are known to be brittle to input perturbations. This raises concerns regarding the robustness of the factors captured by the learned representation space of DNNs. But we know that the factors captured by dimensionality reduction techniques, while being discriminative, are robust to input perturbations. This motivates the thesis behind this work:

{{% alert hint %}}
If we enforce DNNs to learn representations that lie in a low-dimensional subspace (for the entire dataset), we would obtain more robust classifiers while preserving their discriminative power.
{{% /alert %}}

Precisely, we propose a low-rank regularizor (LR) that

1. does not put any restriction on the network architecture.
2. is end-to-end trainable
3. is efficient in that it allows mini-batch training

### Problem Formulation

Consider $f: \mathbb{R}^p \rightarrow \mathbb{R}^k$ to be a feed-forward
multilayer NN that maps $p$ dimensional input $x$ to a $k$
dimensional output $y$. We can decompose this into two
sub-networks, one consisting of the layers before the $\ell^{\it th}$
layer and one after i.e.  $f(x) = f_\ell^{+}(f^{-}_\ell(x;
  \phi) ; \theta)$, where $f^{-}_\ell (.;\phi)$, parameterized by
$\phi$, represents the part of the network up to layer
$\ell$ and, $f^{+}_\ell(.;\theta)$ represents the part of the
network thereafter. With this notation, the $m$ dimensional
representation (or the activations) of any layer $\ell$ can simply be
written as $a = f^{-}_\ell(x; \phi) \in \mathbb{R}^m$.


Let $X = \\{x_i\\}\_{i=1}^n$ and $Y = \\{y_i\\}\_{i=1}^n$ be the set of inputs and outputs
of a given training dataset. By slight abuse of notation, we
define $A_\ell = f^{-}_\ell(X; \phi) =[a_1,\cdots,a_n]^\top\in \mathbb{R}^{n
  \times m}$ to be the activation matrix of the entire dataset, so
that $a_i$ is the activation vector of the $i$-th sample. Note
that for most practical purposes $n\gg m$.

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="731px" viewBox="-0.5 -0.5 731 207" content="&lt;mxfile host=&quot;www.draw.io&quot; modified=&quot;2020-02-18T11:21:30.017Z&quot; agent=&quot;Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36&quot; etag=&quot;DT0IjDWKjBIrWZAHeLs-&quot; version=&quot;12.7.1&quot; type=&quot;device&quot;&gt;&lt;diagram id=&quot;C5RBs43oDa-KdzZeNtuy&quot; name=&quot;Page-1&quot;&gt;7Vpbd+I2EP41Pmf7kBxsg2MegYTsttl2t5Am+6jYwriRLUcWCe6vr4Tki4wAQ3DCXl4SazQa2ZpvPs3MwbBH0fKagGT+GfsQGVbHXxr2pWFZfddmf7kgEwLHdIUgIKEvRGYpmIT/QSnsSOki9GGqKFKMEQ0TVejhOIYeVWSAEPyiqs0wUndNQADXBBMPoHXpXejTuZC61kUp/wjDYJ7vbDp9MROBXFl+SToHPn6piOwrwx4RjKl4ipYjiPjZ5edy9ym7QzePzvXvX9MncDv8Y/rnP2fC2HifJcUnEBjT45q2hOlngBbyvOS30iw/QIIXsQ+5kY5hD+c0QuzRZI//Qkoz6XCwoJiJMKFzHOAYoBuME6k3wzGVaiYfw9gfcMey8QPC3qMQjUOE5B5sJPVdNkopwY+F77iBwhFcGYEHiIbAewxWLzrCCBM2FeMYclM+A4P8lvLlrkrpsOHZSh+keEE8uEVPRgsFJIDb7DlCj79fBafSc9cQR5CSjCkQiAANn1UwAxkTQaFX+p09SNfvAQNbAwMHUek8BQ/O0wLnE2fpyk0DpmBZybKc5IEKPHXB6PMtE08gCWfs/984AnF1gRPw//f5vuwzxNZCvgWVHBEv85DCSSK2fGE8piK1ikB2yMMAgTSV+NkBr/3g8QwJhcutDpWzPUkqklVdOXwpKcrMGXReoadupyUE9HYTwe5Y+q6p4shE4DQkgouGRJCp12pjXpCWvuBwFci5Cp7NUvZeddgUGx6OJOfoXJJTgGLlMH6ZbeQXIU4XD4XoyjLcrtG3K0uq08WKJBedKZqF+AN7HS4cjQ3X+a0xwc1x9LBId5ObAnFucgyiEHG4fIToGdLQAxoKBCgMYjbwGNoh0ccO2zKMAzZyytF0Fatn3RapscC4hHyv14wb+21x48UpI3pwdESfytWrYHmEo2TBkWp1eIlC4uJE7OF4miV8g6BFUNrO+13Y3lPfG4He9CEckOXNNLNuhpPiCw4Hpel+nzRrWENjO9FeGUP3hyPaNaw2hfnmHNRSMe1oMN3RYNptC9MNqtEfPAmFy5Dey2X8+Rtfdd6To8tlviMfZEqJUksgj5jJXrxTQlrqSLx2+ypeC3DmNsSnyFVbDNWBv2ZIZO1rhhgMQFZRS7hC+sokWhsJJ1OQ/3U7/XI7PbXUoC1GdC5O7JZvUpaX9OTxEww99ZgPoRR2hiS7rw4qq/iwXLYatUBF25KealG9OXzWvVxxY0/jxd4rqSynqVpvp1dHR2Oa6qiGuvUKZwNNHYuEdJX8ZvDJG8wH6bxggAoMXwWN93JlvRQ166lPU1euGapjomVX6krYn8uVTv3OP5or64ZadqX7y5V1Vx5KsDsx0bIr+z+9K7tm99xxFCcU8bWvNzW23tqhpq6C3bMro20VXgIKmibhLCemtc7FKpuulZozVsDWRGu9Cp5hhx5AAzkRhb6/KrJ1qb1aeFeye+791lJ2t6d63FpP2bsa/FptZezm60s3PQQGHoszFms4Tn8hQdui7VvnrnNiaOiuM/qeHSxtj0rTy9KUfGWZ960yoy/59F5qsZSTVLmzlhOKb/DLCP3uugpctqQT3jDZFM4z2RzmAc2mQJSsDtW2eUfXixZ8pWy9ENl6UXUM0cxm2OURdpaKEOPmYkwigLZ33cXLvbbl/XahqyZiRQ7QMAvZw+1sWP5kS9z75e/e7Kv/AQ==&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://www.draw.io/?client=1&amp;lightbox=1&amp;edit=_blank');}}})(this);" style="cursor:pointer;max-width:100%;max-height:207px;"><defs/><g><path d="M 110 45 L 149.88 45" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 158.88 45 L 149.88 49.5 L 149.88 40.5 Z" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="0" y="25" width="110" height="40" rx="6" ry="6" fill="#ffffff" stroke="#000000" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 108px; height: 1px; padding-top: 45px; margin-left: 1px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 22px" face="CMU Serif Roman">X</font></div></div></div></foreignObject><text x="55" y="49" fill="#000000" font-family="Helvetica" font-size="12px" text-anchor="middle">X</text></switch></g><path d="M 270 45 L 299.88 45" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 308.88 45 L 299.88 49.5 L 299.88 40.5 Z" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 215 0 L 270 45 L 215 90 L 160 45 Z" fill="#ffffff" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 100px; height: 1px; padding-top: 43px; margin-left: 165px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 22px"><font face="CMU Serif Roman">f</font><sub>ℓ</sub><sup>-</sup>( ;φ)</font></div></div></div></foreignObject><text x="215" y="47" fill="#000000" font-family="Helvetica" font-size="12px" text-anchor="middle">fℓ-( ;φ)</text></switch></g><rect x="310" y="25" width="110" height="40" rx="6" ry="6" fill="#ffffff" stroke="#000000" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 108px; height: 1px; padding-top: 45px; margin-left: 311px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Computer Modern Roman; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 22px"><font face="CMU Serif Roman">A</font><sub>ℓ</sub></font></div></div></div></foreignObject><text x="365" y="49" fill="#000000" font-family="Computer Modern Roman" font-size="12px" text-anchor="middle">Aℓ</text></switch></g><path d="M 520 5 L 570 45 L 520 85 L 470 45 Z" fill="#ffffff" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 90px; height: 1px; padding-top: 43px; margin-left: 475px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 18px"><font face="CMU Serif Roman">f</font><sub>ℓ</sub><sup>+</sup>( ;θ)</font></div></div></div></foreignObject><text x="520" y="47" fill="#000000" font-family="Helvetica" font-size="12px" text-anchor="middle">fℓ+( ;θ)</text></switch></g><path d="M 420 45 L 459.88 45" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 468.88 45 L 459.88 49.5 L 459.88 40.5 Z" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="620" y="25" width="110" height="40" rx="6" ry="6" fill="#ffffff" stroke="#000000" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 108px; height: 1px; padding-top: 45px; margin-left: 621px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 22px" face="CMU Serif Roman">OUTPUT</font></div></div></div></foreignObject><text x="675" y="49" fill="#000000" font-family="Helvetica" font-size="12px" text-anchor="middle">OUTPUT</text></switch></g><path d="M 570 45 L 613.63 45" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 618.88 45 L 611.88 48.5 L 613.63 45 L 611.88 41.5 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 160 125 L 160 85" fill="none" stroke="#000000" stroke-miterlimit="10" stroke-dasharray="3 3" pointer-events="stroke"/><path d="M 570 125 L 160 125" fill="none" stroke="#000000" stroke-miterlimit="10" stroke-dasharray="3 3" pointer-events="stroke"/><path d="M 570 85 L 570 125" fill="none" stroke="#000000" stroke-miterlimit="10" stroke-dasharray="3 3" pointer-events="stroke"/><path d="M 364.66 155 L 364.66 125" fill="none" stroke="#000000" stroke-miterlimit="10" stroke-dasharray="3 3" pointer-events="stroke"/><rect x="35" y="65" width="40" height="20" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 75px; margin-left: 36px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 22px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 12px">Data</font></div></div></div></foreignObject><text x="55" y="82" fill="#000000" font-family="Helvetica" font-size="22px" text-anchor="middle">Data</text></switch></g><rect x="342.86" y="65" width="40" height="20" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 75px; margin-left: 344px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 22px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 12px">Activations</font></div></div></div></foreignObject><text x="363" y="82" fill="#000000" font-family="Helvetica" font-size="22px" text-anchor="middle">Acti...</text></switch></g><path d="M 55 85 L 55 85" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 55 85 L 55 85 L 55 85 L 55 85 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 365 155 L 390 180 L 365 205 L 340 180 Z" fill="#ffffff" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 48px; height: 1px; padding-top: 180px; margin-left: 341px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 22px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span style="font-family: &quot;cmu serif roman&quot; ; white-space: normal">f</span></div></div></div></foreignObject><text x="365" y="187" fill="#000000" font-family="Helvetica" font-size="22px" text-anchor="middle">f</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://desk.draw.io/support/solutions/articles/16000042487" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

In this setting,
the problem of learning low-rank representations can be
formulated as a constrained optimization problem as follows:
\begin{align}
  \label{eq:opt_prob}
	\min_{\theta, \phi}\mathcal{L}(X, Y; \theta, 
  \phi),~\text{s.t.}~~\mathrm{rank}(A_\ell) = r,\end{align} 
where $\mathcal{L}(.)$ is the loss function and $r < m$ is the
desired rank of the representations at layer $\ell$. The rank $r$ is a
hyperparameter. However, there are a few problems with this formulation.

First, the rank constraint is on a matrix
* whose size scales with dataset size.
* which is not a parameter;
* and it is not immediately clear if a Tikhonov regularizor exists that can achieve this.

Second, minimizing the restriction of $A_\ell$ on a minibatch can result in orthogonal low rank spaces for each minibatch thus having a high dimensional subspace when all the minibatches are combined.

To mitigate these issues, we augment the initial problem as follows:
#### Augmented problem (our low rank regularizer)


$$
  \min_{\theta, \phi, W, b} \mathcal{L}(X, Y; \theta, \phi) + \mathcal{L}_c(A_\ell; W,b) + \mathcal{L}_n(A_\ell)$$
  $$\text{s.t.,} W\in \mathbb{R}^{m\times m}, \mathrm{rank}(W) = r,~ b\in \mathbb{R}^m,A=f^{-}_\ell(X; \phi)
$$
where,
$$
  \underbrace{\mathcal{L}_c(A; W, b) = \frac{1}{n} \sum_{i=1}^{n} \Big\\|W^\top(a_i+b) - (a_i+b)\Big\\|_2^2}_{\Large\text{Projection Loss}}$$
 $$ \text{and} \quad \underbrace{\mathcal{L}_n(A) = \frac{1}{n}\sum_{i=1}^n \Big|1 - \\|a_i\\| \Big|}_{\Large \text{Norm Loss}}$$

__Projection Loss__: Minimizing the projection loss $\mathcal{L}_c$
ensures that the affine low-rank mappings ($AW$) of the activations
are close to the original ones i.e. $AW \approx A$. As $W$ is
low-rank, recalling sub-multiplicity of rank - $\mathrm{rank}(AW)\le \mathrm{min}(\mathrm{rank}(A),\mathrm{rank}(W))$,  $AW$ is also low-rank; thus implicitly~(due to
$AW\approx A$) it forces the original activations $A$ to
be low-rank. The bias $b$ allows for the activations to be translated before
projection.


__Norm Loss__: However note that setting $A$ and $b$ close to zero trivially minimizes $\mathcal{L}_c$, especially when the activation dimension is large. We observed this to happen in practice as it is easier for the network to learn $\phi$ such that the activations and the bias are very small in order to minimize $\mathcal{L}_c$. To prevent this, we use $\mathcal{L}_n$ that acts as a norm constraint on the activation vector to keep the activations sufficiently large.

Intuitively, we learn a _virtual layer i.e._($W,b$) during training that does two things.

1. Learns a **very** low dimensional subspace that captures a large fraction of the information present in the representations/activations of the network on the training data.

2. Learns to encourage the representations/activations to lie as much as possible entirely on this low dimensional subspace.
#### Optimizing the Loss

We optimize the loss using an alternate minimization scheme. During forward pass, the loss from the three components $\mathcal{L}, \mathcal{L}_c,\mathcal{L}_n$ are back-propagated through the network. Every $10$ iteration, $W$ is rank thresholded using  column-sampled ensembled Nystr&ouml;m SVD, which is essentially an approximate SVD.

{{<figure src="/lr_layer/LR_fw_back.png" title="Forward and backward pass through our regularizor.">}}

### Adversarial Robustness of our method

We recall that adversarial perturbations are well crafted~(almost
imperceptible) input perturbations that, when added to a clean input, flips the prediction
of the model on the input to an incorrect
one.

{{<figure src="/lr_layer/adv_pig.png" title="An example of an image of a pig that is initially correctly classified by a classifier. On adding a small impertible perturbation, the same classifier mis-classifies it as an airliner. <a href='#adv_ref'>[a]</a>" lightbpx="true">}}

We look at the adversarial robustness of our model as compared to a vanilla model i.e. one without our regularizor; and also with some other methods that impose constraints on the parameter space. We test it against two main adversaries -
1. one that is computationally constrained where we measure the success of the adversarial attack for different computational budgets,
2. computationally unconstrained where we measure the amount of perturbation necessary for $99\%$ mis-classification.

**Computationally Constrained Adversary**

In the table below, we measure the adversarial accuracy of a ResNet50 model trained on the CIFAR10 dataset against an untargeted $\ell_{\infty}$  PGD adversary with the computational constraints (i.e. attack budgets measured in terms of $\ell_\infty$ radius and att. steps. Higher these values, stronger is the adversary.) The table  shows that our Low Rank regularizer has a much higher test accuracy than any of the other methods used.

<table  class="table"  fgcolor="white" >
<caption >ResNet50 trained on CIFAR10. Low Rank uses our regularizor; SNIP <a href="#section1">[1]</a> is a pruning technique to to  enforce sparsity in parameters and SRN<a href="#section2"> [2]</a> is a technique to introduce a soft low rank structure in the parameter space.</caption>
  <thead>
    <tr bgcolor="white">
      <th style="width:12.5%" scope="col">Algorithm</th>
      <th class="text-center" colspan=8 style="width:87.5%" scope="col">Adversarial Test Accuracy</th>
    </tr>
    <tr bgcolor="white">
      <th style="width:12.5%" scope="col">$\ell_\infty$ radius</th>
      <th class="text-center" colspan=2 style="width:21.875%" scope="col"><sup>8</sup>&frasl;<sub>255</sub></th>
      <th class="text-center" colspan=2 style="width:21.875%" scope="col"><sup>10</sup>&frasl;<sub>255</sub></th>
      <th class="text-center" colspan=2 style="width:21.875%" scope="col"><sup>16</sup>&frasl;<sub>255</sub></th>
      <th class="text-center" colspan=2 style="width:21.875%" scope="col"><sup>20</sup>&frasl;<sub>255</sub></th>
     </tr>
     <tr bgcolor="white">	
      <th style="width:12.5%" scope="col">Att. steps</th>
      <th class="text-center" style="width:10.9375%" scope="col">7</th>
      <th class="text-center" style="width:10.9375%" scope="col">20</th>
      <th class="text-center" style="width:10.9375%" scope="col">7</th>
      <th class="text-center" style="width:10.9375%" scope="col">20</th>
      <th class="text-center" style="width:10.9375%" scope="col">7</th>
      <th class="text-center" style="width:10.9375%" scope="col">20</th>
      <th class="text-center" style="width:10.9375%" scope="col">7</th>
      <th class="text-center" style="width:10.9375%" scope="col">20</th>
     </tr>
  </thead>
  <tbody>
    <tr bgcolor="white">
  <th  scope="row">Vanilla</th>
  <td> 43.1 </td>
  <td> 31.0  </td>
  <td> 38.5 </td>
  <td> 21.8 </td>
  <td> 31.2 </td>
  <td> 7.8   </td>
  <td> 28.9 </td>
  <td> 4.5 </td> 
    </tr>
    <tr  bgcolor="white">
      <th scope="row">SNIP <a href="#section1">[1]</a> </th>
     <td>   29.4 </td>
     <td> 14.5  </td>
     <td> 25.0 </td>
     <td> 8.0 </td>
     <td> 18.5 </td>
     <td> 1.3  </td>
     <td> 16.2 </td>
     <td> 0.4 </td>
    </tr>
     <tr  bgcolor="white">
      <th scope="row">SRN <a href="#section1">[2]</a> </th>
    <td> 47.8</td>
    <td> 37.6</td>
    <td> 44.4</td>
    <td> 31.4</td>
    <td> 39.8</td>
    <td>  21.3</td>
    <td> 37.5</td>
    <td> 18.4</td>
    </tr>
     <tr bgcolor="white"> 
      <th scope="row">Low Rank (Ours)</th>
    <td> $\mathbf{79.1}$ </td>
    <td> $\mathbf{78.5}$  </td>
    <td> $\mathbf{78.6}$ </td>
    <td> $\mathbf{78.1}$  </td>
    <td> $\mathbf{77.9}$ </td>
    <td> $\mathbf{77.0}$  </td>
    <td> $\mathbf{77.1}$ </td>
    <td> $\mathbf{76.6}$ </td>
    </tr>
   
  </tbody>
</table>

**Computationally UnConstrained Adversary (Deepfool) <a href="#section3"> [3]</a>**

We also look at the amount of adversarial perturbation that is necessary to fool our classifier as opposed to a vanilla model. Measured numerically, ours requires an order of magnitude more than normal models. Please refer to our paper for the exact values. If visualized, the images appear as below and the adversarial images for teh Low-Rank model (LR) are significantly more different from the original model than those for the vanilla model. 


Original Image|  Vanilla Model | Low Rank Model |
:--------:|:--------:|:--------:|
{{< figure src="/lr_layer/orig_1.png" lightbox="true" >}}  |  {{< figure src="/lr_layer/N-LR_1.png"  lightbox="true" >}} |  {{< figure src="/lr_layer/2-LR_1.png" lightbox="true" >}}|
{{< figure src="/lr_layer/orig_3.png"  lightbox="true" >}} |  {{< figure src="/lr_layer/N-LR_3.png"  lightbox="true" >}} |  {{< figure src="/lr_layer/2-LR_3.png" lightbox="true" >}}|

### Noise Stability


We explain this robustness by showing that the noise-stability of the low rank representations are much more than those of vanilla models. More specifically, we show two different experiments

1. <b> Random Pixel Perturbation: </b> When a random subset (each pixel is chosen iid with probability $p$)  of the pixels of a image are perturbed with a random multi-variate gaussian noise, the test accuracy of our low-rank models drops much slower than that of a vanilla model.

Pert. Prob. $p$|  $0.4$ | $0.6$ | $0.8$ | $1.0$ |
----|------------|------------|------------|------------|
Vanilla | $69.7$ | $26.1$  | $12.6$ | $11.3$|
Low-Rank (Ours) | $\mathbf{75.1}$ | $\mathbf{34.2}$ | $\mathbf{15.8}$ | $\mathbf{13.0}$|


2.<b> Noise-Stability of Representations:</b> We measure the ratio of the norm of the perturbation induced in the representation space with to the norm of the perturbation in the input space for adversarial perturbations. In the following diagram, x-axis measures $$\text{x-axis}:\dfrac{\\|\delta\\|_2^2}{\\|x\\|_2^2}\quad \text{y-axis}:\dfrac{\\|f_\ell^{-}(x+\delta) - f_\ell^{-}(x)\\|_2^2}{\\|f_\ell^{-}(x)\\|_2^2}$$
White box attacks are attacks where the perturbation is constructed using the model to be attacked, black box models are attacks where the vanilla model is used to construct the attack.

||White Box Attack|Black Box Attack|
|:--:|:-----:|:-----:|
{{< figure src="/lr_layer/pert_legend.png">}}|{{< figure src="/lr_layer/pert_comp_fig.png"   >}} | {{< figure src="/lr_layer/bb_pert_comp_fig.png"  >}}

### Further observations in the paper

1. **Layer Cusion**: Arora et. al  <a href="#section4"> [4]</a> define a quantity called layer cushion, which is intuitively the reciprocal of noise-sensitivity to random gaussian noise measured on the real dataset. We measure this quantity for all our networks and show that this quantity is much higher for our networks.
2. **Compressing Representations**: Due to the low *intrinsic dimension* of the low rank representations, even after aggressive compression ($400x$) outhe low rank model looses only $6\\%$ in accuracy as opposed to the vanilla model which looses more than $27\\%$.
3. **Compressing Models**: We also throw away the latter parts of the model (after the low rank representation) and train a small linear model ($8M$ replaced with $160k$ parameters) yielding less than $1\\%$ drop in accuracy.
4. **Discriminative Representations**: When visualized with PCA (and colored according to classes), the low rank model yields representations that are more discriminative in nature in the sense that a low dimensional lienar classifier can classify it with a much larger margin than for vanilla models.

  Vanilla Model | Low Rank Model |
:--------:|:--------:|
  {{< figure src="/lr_layer/coarse_label_PCA_NLR.png"  lightbox="true" >}} |   {{< figure src="/lr_layer/coarse_label_PCA.png"  lightbox="true" >}} |

For the full paper refer to https://arxiv.org/abs/1804.07090.
<p style="font-size:11px" id="adv_ref"> [a] Picture taken from https://medium.com/@smkirthishankar/the-unusual-effectiveness-of-adversarial-attacks-e1314d0fa4d3 </p>
<p style="font-size:11px"  id="section1">[1] Lee, Namhoon, Thalaiyasingam Ajanthan, and Philip HS Torr. "Snip: Single-shot network pruning based on connection sensitivity." International Conference on Learning Representations (2019). </p>

<p style="font-size:11px"  id="section1">[2] Sanyal, Amartya, Philip HS Torr, and Puneet K. Dokania. "Stable Rank Normalization for Improved Generalization in Neural Networks and GANs."  International Conference on Learning Representations (2020). </p>

<p style="font-size:11px"  id="section3">[3] Moosavi-Dezfooli, Seyed-Mohsen, Alhussein Fawzi, and Pascal Frossard. "Deepfool: a simple and accurate method to fool deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. </p>


<p style="font-size:11px"  id="section4">[4] Arora, Sanjeev, et al. "Stronger generalization bounds for deep nets via a compression approach." arXiv preprint arXiv:1802.05296 (2018). </p>