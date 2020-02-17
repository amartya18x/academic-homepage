---
title: "Robustness via Deep Low-Rank Representations"
date: 2020-01-23T12:40:23Z
draft: true
image:
---

In this paper ([Full Paper here](https://arxiv.org/abs/1804.07090)), we investigate the relation of the _intrinsic_ dimension of the representation space of deep networks with its robustness. To do this, we introduce an easy-to-implement, end-to-end trainable, scalable  regularizer that enforces low rank structure in the representation space of deep networks; we conduct a variety of experiments to show that the resultant network is largely robust to adversarial and random perturbations. In addition, the low _intrinsic_ dimension also means that the representations (and the model) can be compressed significantly without significant loss in accuracy.

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

 and provides benefits;
 1. it increases the robustness against black box and white box version of various adversarial attacks.
 2. the generated representations  are discriminative and can be compressed due to it having a low _intrinsic_ dimension.

### Problem Formulation

Let $X = \\{x_i\\}\_{i=1}^n$ and $Y = \\{y_i\\}\_{i=1}^n$ be the set of inputs and outputs
of a given training dataset. By slight abuse of notation, we
define $A_\ell = f^{-}_\ell(X; \phi) =[a_1,\cdots,a_n]^\top\in \mathbb{R}^{n
  \times m}$ to be the activation matrix of the entire dataset, so
that $a_i$ is the activation vector of the $i$-th sample. Note
that for most practical purposes $n\gg m$. In this setting,
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
#### Augemented problem (our low rank regularizer)


$$
  \min_{\theta, \phi, W, b} \mathcal{L}(X, Y; \theta, \phi) + \mathcal{L}_c(A_\ell; W,b) + \mathcal{L}_n(A_\ell)$$
  $$\text{s.t.,} W\in \mathbb{R}^{m\times m}, \mathrm{rank}(W) = r,~ b\in \mathbb{R}^m,A=f^{-}_\ell(X; \phi)
$$
where,
$$
  \underbrace{\mathcal{L}_c(A; W, b) = \frac{1}{n} \sum_{i=1}^{n} \Big\\|W^\top(a_i+b) - (a_i+b)\Big\\|_2^2}_{\Large\text{Projection Loss}}$$
 $$ \text{and} \quad \underbrace{\mathcal{L}_n(A) = \frac{1}{n}\sum_{i=1}^n \Big|1 - \\|a_i\\| \Big|}_{\Large \text{Norm Loss}}$$

__Projetion Loss__: Minimizing the projection loss $\mathcal{L}_c$
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