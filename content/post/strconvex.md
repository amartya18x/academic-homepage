+++
date = "2016-09-15T23:49:40+05:30"
title = "Strong Convexity and Strong Smoothness"
description = "Let's talk about Strongly convex and strongly smooth functions and how gradient descent performs on them."
tags=["academic"]
+++


I am going to , inspired by the course on optimization that I am doing this semester, talk a bit about strong convexity and strong smoothness and our very popular __gradient descent__ works on them. So, before going right into the details let's have a quick chat about convexity in general and we do have a few ways of going about it. 

I will go about with talking about two definitions of convex functions, the first one being general more than the second. This is not to say that there aren't more general or equivalent definitions, which there are, but that I like these better and this is probably what you will use more in your life if you have to use one at all.

## Convex Function

$$\frac{f(y)+f(x)}{2} \ge f(\frac{x+y}{2}) $$
Note that this does not require the function to be differentiable. But, with differetiable functions, one can actually get an easier and more productive definition.

### Convex differetiable functions

$$ f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle $$

### Subgradients 
The interesting thing about this is that this can tolerate functions, which are not differntiable in a finite number of points. We need to define __subgradients__ for that though.

$$ g(x) = \\{ g(x) | \langle g(x), x_0 - x\rangle \forall y \\} $$

And hence let's do the intuitive thing of replacing  $\nabla f(x)$ with $g(x)$. So, now we have 

$$ f(y) \ge f(x) + \langle  g(x), y - x \rangle $$

## Strongly Convex function

Now that we know of convex functions mathematically, it is intuitively a function such that if we draw a tangent plane(line in case of one variable), the function __lies above the tangent plane__ at all points. Mathematically, that would be 

$$ f(y) \ge f(x) + \langle g(x), y - x \rangle + \frac{\alpha}{2} \\| y - x\\|^2 $$ and __$\alpha$__ is known as the strong convexity parameter. One can also, for doubly differntiable functions, say $ \nabla^2 f(x) \succeq \alpha I  $, where $\nabla^2 f(x)$ is the hessian matrix and $ I $ is the identity matrix.


## Strongly Smooth function

A strongly smooth function is just the opposite of the strongly convex function i.e. a function which __lies below the tangent plane__ at all points. Mathematically, that would be 

$$ f(y) \le f(x) + \langle g(x), y - x \rangle + \frac{\beta}{2} \\| y - x\\|^2 $$ and __$\beta$__ is known as the strong smoothness parameter. One can also, for doubly differntiable functions, say $ \beta I \succeq \nabla^2 f(x)  $.

# Gradient descent for strongly convex functions

We know for a strongly convex function $f(x)$, $$ f(y) \ge f(x) + \langle g(x), y - x \rangle + \frac{\alpha}{2} \\| y - x\\|^2 $$ A very nice property of these functions is that, we can actually bound $\\|f( x^{\*}) - f(x)\\|$ where $(x^{\*})$ is the optimal point , with the norm of the gradient ($\\|\nabla f(x)\\|$).
Let, 
$$ 
z = \underset{y}{argmin }\hspace{5pt}  \\{ f(x) + \langle g(x), y - x \rangle + \frac{\alpha}{2} \\| y - x\\|^2 \\}
$$ 

The definition of strong convexity is applicable for all y.
 
\begin{align\*}
z &= x - \frac{\nabla f(x)}{\alpha} \\\\\
f(y) &\ge f(x) + \langle g(x), y - x \rangle + \frac{\alpha}{2} \\| y - x\\|^2  \\\\\
&\ge f(x) + \langle g(x), \frac{\nabla f(x)}{\alpha}  \rangle + \frac{\alpha}{2} \\| \frac{\nabla f(x)}{\alpha}  \\|^2 \\\\\
&\ge f(x) - \frac{1}{2\alpha} \| \nabla f(x)\|^2 \\\\\
\therefore f(x^{\*}) &\ge f(x) - \frac{1}{2\alpha} \\| \nabla f(x)\\|^2
\end{align\*}

This interesting bound also gives us a good convergence criterion. Set $f(x) - f(x^{\*}) = \epsilon$ i.e. the desired error. We only need to do the gradient descent until the gradient has reached $2\alpha \epsilon$ from the previous inequality. 
$$
    2\alpha \epsilon \le \frac{1}{2\alpha} \\| \nabla f(x)\\|^2
$$

Like the function value above, we can also get a bound on the distance of the current point from the optimal point in terms of gradient.

\begin{align\*}
    f(x^{\*})  &\ge f(x) + \langle g(x), x^{\*} - x \rangle + \frac{\alpha}{2} \\| x^{\*} - x\\|^2  \\\\\
     &\ge f(x) - \underbrace{( \\{ \\| g(x)\\|\\| x^{\*} - x \\| - \frac{\alpha}{2} \\| x^{\*} - x\\|^2 \\}) }_a && \text{(By Cauchy's inequality)}\\\\\
\end{align\*}
As we know that $f(x^{\*})$ is optimal $ f(x^{\*}) \le f(x)$, a \ge 0 must be true.

$$ \\| g(x)\\|\\| x^{\*} - x \\| \ge \frac{\alpha}{2} \\| x^{\*} - x\\|^2  $$
Or, 
$$\frac{2}{\alpha} \\| g(x)\\| \ge   \\| x^{\*} - x\\|  $$

## Analyzing the gradient descent

Define, 
$$
    \Phi_t = f(x^t) - f(x^{\*})
$$  
- the lyapunov function. Decrease in the value of the lyapunov function means that the function is nearing its optima. Let's also define another distance.
$$
    D_t = \\| x^{\*} - x_t \\|_2
$$
\begin{align\*}
    \label{eq:phi}
     f(x^{\*})  &\ge f(x_t) + \langle \nabla f(x_t), x^{\*} - x_t \rangle + \frac{\alpha}{2} \\| x^{\*} - x_t\\|^2  \\\\\
     f(x^{\*}) -  f(x_t)  &\ge \hspace{2pt} \langle\nabla f(x_t), x^{\*} - x_t \rangle + \frac{\alpha}{2} \\| x^{\*} - x_t\\|^2 \\\\\
\end{align\*}

Hence, we have the following inequality with the lyapunov function.
\begin{equation}
     \Phi_t \le -\langle g(x_t), x^{\*} - x_t \rangle - \frac{\alpha}{2} \\| x^{\*} - x_t\\|^2
\end{equation}
Lets work with $D_t$ and then try to relate it with $\Phi_t$.
\begin{align\*}
    D^2\_{t+1} &= \\|x^{t+1} - x^{\*} \\|\_2^2 \\\\\
    &= \\| x^t - \eta\_t g\_t -   x^{\*}\\|\_2^2 \\\\\
    & = \\|x^t - x^{\*} \\|\_2^2 + \eta\_t^2 \\|g\_t(x_t)\\|^2  - 2 \eta\_t   \langle x^t - x^{\*}, g\_t\rangle \\\\\
    D^2\_{t+1} - D^2\_{t} &= \eta\_t^2 \\|g\_t(x_t)\\|^2  - 2 \eta\_t   \langle x^t - x^{\*}, g\_t\rangle \\\\\
    \langle x^t - x^{\*}, g\_t\rangle &= \frac{D^2\_t - D^2\_{t+1}}{2\eta\_t} + \frac{\eta_t \\|g\_t\\|^2}{2}
\end{align\*}
Let's plugin in this onto the previous definition of the lyapunov function.
\begin{align\*}
    \Phi_t &\le -\langle g(x_t), x^{\*} - x_t \rangle - \frac{\alpha}{2} \\| x^{\*} - x_t\\|^2 \\\\\
    &\le \frac{D^2\_t - D^2\_{t+1}}{2\eta\_t} + \frac{\eta_t \\|g\_t\\|^2}{2} - \frac{\alpha}{2} \\| x^{\*} - x_t\\|^2 \\\\\
    &\le  \frac{D^2\_t - D^2\_{t+1}}{2\eta\_t} + \frac{\eta_t \\|g\_t\\|^2}{2} - \frac{\alpha}{2} D^2\_t
\end{align\*}
It is difficult to show that $\Phi\_t$ is going to zero. It is considerably easier to use the sum for that. We will see a trick with jensen's inequality. 
\begin{align\*}
    \sum\_{t=0}^T \Phi\_t &\le \sum\_{t=0}^T (\frac{D^2\_t - D^2\_{t+1}}{2\eta\_t} + \frac{\eta_t \\|g\_t\\|^2}{2} - \frac{\alpha}{2} D^2\_t) \\\\\
    &\le  \sum\_{t=0}^T (\frac{D^2\_t}{2\eta\_t} + \frac{\eta_t \\|g\_t\\|^2}{2} - \frac{\alpha}{2} D^2\_t) -  \sum\_{t=1} \frac{D^2\_{t}}{2\eta\_{t-1}} && \text{(A bit of change of variable)} \\\\\
    &\le D\_0^2(\frac{1}{2\eta\_0} - \frac{\alpha}{2}) +  \sum\_{t=1}^T D^2\_t(\frac{1}{2\eta\_t} - \frac{\alpha}{2} + \frac{1}{2\eta\_{t-1}}) + \sum\_{t=0}^T\frac{\eta_t G^2}{2} &&\text{(Assuming bounded gradients again)}
\end{align\*}

Now, the first term is a constant and the third term is a sum of constants weighed by a parameter that we are fixing. So, the major problem is with the second term. Why note set it to zero ? We can do that by setting `$\eta\_t = \frac{1}{\alpha t}$`
\begin{align\*}
    \sum\_{t=0}^T \Phi\_t &\le D\_0^2(\underbrace{\frac{1}{2\eta\_0} - \frac{\alpha}{2}}\_{\le 0}) +  \sum\_{t=1}^T D^2\_t( \underbrace{\frac{1}{2\eta\_t} - \frac{\alpha}{2} + \frac{1}{2\eta\_{t-1}}}\_0) + \sum\_{t=0}^T\frac{ G^2}{2\alpha t} &&\text{(Assuming bounded gradients again)} \\\\\
    \frac{1}{T}\sum\_{t=0}^T {\Phi\_t} &\le \frac{G^2 log(T)}{2T} \\\\\
    \sum\_{t=0}^T  \frac{f(x^t) - f(x^{\*})}{T} &\le \frac{G^2 log(T)}{2T}  \\\\\
    \end{align\*}
Applying Jensen's inequality because $\Phi\_t$ is convex
         $$f(\frac{\sum\_{t=0}^T x^t}{T}) \le \sum\_{t=0}^T  \frac{f(x^t) }{T} - f(x^{\*}) \le \frac{G^2 log(T)}{2T} $$
To remove the $log(T)$, try doing a weighed sum of the $\Phi\_t$ with $t\Phi\_(t)$ and then don't forget to divide by $\sum\_{t=1}^{T}t$. It will work out smooth

# Gradient descent for strongly smooth functions

We will work with similar __Lyapunov function__ `$\Phi\_t$` and `$D\_t$`

As $f(x)$ is strong smooth,
$$ f(x^{t+1}) \le f(x^t) + \langle \nabla f(x^t), x^{t+1} - x^t \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 $$
and as $f(x)$ is convex
$$ f(x^{\*}) \ge f(x) + \langle \nabla f(x), x^{\*} - x \rangle  $$
By rearranging terms we get, 
$$  f(x) \le f(x^{\*})  + \langle \nabla f(x), x - x^{\*}  \rangle  $$
Plugging in this inequality into the strong smoothness inequality and replacing the $\Phi\_{t+1}$ and $D\_t$ in the correct places gives us the following 

\begin{align\*}
\Phi\_{t+1} &\le   \langle \nabla f(x), x^t - x^{\*}\rangle + \langle \nabla f(x^t), x^{t+1} - x^t \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
&\le  \langle \nabla f(x), x^t - x^{\*} +  x^{t+1} - x^t \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
&\le  \langle \nabla f(x),  x^{t+1} - x^{\*} \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
\end{align\*}

Now, we know the following `$x^{t+1} = x^t - \eta\_t \nabla f(x^t)$` which gives us `$\nabla f(x^t) = \frac{x^t - x^{t+1}}{\eta\_t} $`

\begin{align\*}
\Phi\_{t+1} &\le  \langle \nabla f(x),  x^{t+1} - x^{\*} \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
    &\le  \frac{1}{\eta\_t}\langle x^{t} - x^{t+1},  x^{t+1} - x^{\*} \rangle + \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
\end{align\*}
Now, we need to evaluate the term `$\langle x^{t} - x^{t+1},  x^{t+1} - x^{*} \rangle $`
\begin{align\*}
\\|x^t - x^{\*}\\|^2 &= \\|x^t - x^{t+1} + x^{t+1} - x^{\*}\\| \\\\\
    &=\\|x^t - x^{t+1}\\|^2 + \\|x^{t+1} - x^{\*}\\| + 2\langle x^{t} - x^{t+1},  x^{t+1} - x^{\*} \rangle \\\\\
\langle x^{t} - x^{t+1},  x^{t+1} - x^{*} \rangle &= \frac{1}{2}( D\_{t}^2 - D\_{t+1}^2 -  \\|x^{t+1} - x^{\*}\\| )
\end{align\*}
Plugging this back into the inequality we have above, we get the following

\begin{align\*}
\Phi\_{t+1}     &\le  \frac{1}{2\eta\_t} (D\_{t}^2 - D\_{t+1}^2 -  \frac{1}{2}\\|x^{t+1} - x^{\*}\\| )+ \frac{\beta}{2} \\| x^{t+1} - x^t \\|^2 \\\\\
&\le \frac{1}{2\eta\_t} (D\_{t}^2 - D\_{t+1}^2) - \\| x^{t+1} - x^t \\|^2 (\frac{1}{2\eta\_t} - \frac{\beta}{2})
\end{align\*}
Set  $\eta\_t = \frac{c}{\beta} $  where, $c\in [0,1]$ . We get($0 \ge k\le 1) $
$$ (\frac{1}{2\eta\_t} - \frac{\beta}{2}) = \frac{\beta}{2} (\frac{1}{k} - 1)\ge 0  $$

\begin{align\*}
\sum\_{t=0}^T \Phi\_{t+1} &\le \sum\_{t=0}^T \frac{\beta}{2c} (D\_{t}^2 - D\_{t+1}^2)\\\\\
\frac{1}{T} \sum\_{t=0}^T \Phi\_{t+1}  &\le \frac{\beta}{2cT} (D\_{0}^2 - D\_{T+1}^2) 
\end{align\*}
Applying Jensen's inequality because $\Phi\_t$ is convex
         $$f(\frac{\sum\_{t=0}^T x^t}{T}) \le \sum\_{t=0}^T  \frac{f(x^t) }{T} - f(x^{\*}) \le \frac{\beta}{2cT} (D\_{0}^2 ) $$ 
         
For both the strongly convex and the strong smooth functions, we have seen the average selector. Is it possible that some other selector can give us better bounds ? Selector refers to the particular way of choosing the $x$, which we want the function to return. 

