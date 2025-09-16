Published as a conference paper at ICLR 2017

## SGDR: S TOCHASTIC G RADIENT D ESCENT WITH W ARM R ESTARTS


**Ilya Loshchilov & Frank Hutter**
University of Freiburg
Freiburg, Germany,
_{_ ilya,fh _}_ @cs.uni-freiburg.de


A BSTRACT


Restart techniques are common in gradient-free optimization to deal with multimodal functions. Partial warm restarts are also gaining popularity in gradientbased optimization to improve the rate of convergence in accelerated gradient
schemes to deal with ill-conditioned functions. In this paper, we propose a simple warm restart technique for stochastic gradient descent to improve its anytime
performance when training deep neural networks. We empirically study its performance on the CIFAR-10 and CIFAR-100 datasets, where we demonstrate new
state-of-the-art results at 3.14% and 16.21%, respectively. We also demonstrate
its advantages on a dataset of EEG recordings and on a downsampled version of
the ImageNet dataset. Our source code is available at
[https://github.com/loshchil/SGDR](https://github.com/loshchil/SGDR)


1 I NTRODUCTION


Deep neural networks (DNNs) are currently the best-performing method for many classification
problems, such as object recognition from images (Krizhevsky et al., 2012a; Donahue et al., 2014)
or speech recognition from audio data (Deng et al., 2013). Their training on large datasets (where
DNNs perform particularly well) is the main computational bottleneck: it often requires several
days, even on high-performance GPUs, and any speedups would be of substantial value.


The training of a DNN with _n_ free parameters can be formulated as the problem of minimizing a
function _f_ : IR _[n]_ _→_ IR. The commonly used procedure to optimize _f_ is to iteratively adjust _**x**_ _t_ _∈_ IR _[n]_

(the parameter vector at time step _t_ ) using gradient information _∇f_ _t_ ( _**x**_ _t_ ) obtained on a relatively
small _t_ -th batch of _b_ datapoints. The Stochastic Gradient Descent (SGD) procedure then becomes
an extension of the Gradient Descent (GD) to stochastic optimization of _f_ as follows:


_**x**_ _t_ +1 = _**x**_ _t_ _−_ _η_ _t_ _∇f_ _t_ ( _**x**_ _t_ ) _,_ (1)


where _η_ _t_ is a learning rate. One would like to consider second-order information


_**x**_ _t_ +1 = _**x**_ _t_ _−_ _η_ _t_ _**H**_ _[−]_ _t_ [1] _[∇][f]_ _[t]_ [(] _**[x]**_ _[t]_ [)] _[,]_ (2)


but this is often infeasible since the computation and storage of the inverse Hessian _**H**_ _[−]_ _t_ [1] is intractable for large _n_ . The usual way to deal with this problem by using limited-memory quasiNewton methods such as L-BFGS (Liu & Nocedal, 1989) is not currently in favor in deep learning,
not the least due to (i) the stochasticity of _∇f_ _t_ ( _**x**_ _t_ ), (ii) ill-conditioning of _f_ and (iii) the presence
of saddle points as a result of the hierarchical geometric structure of the parameter space (Fukumizu
& Amari, 2000). Despite some recent progress in understanding and addressing the latter problems
(Bordes et al., 2009; Dauphin et al., 2014; Choromanska et al., 2014; Dauphin et al., 2015), state-ofthe-art optimization techniques attempt to approximate the inverse Hessian in a reduced way, e.g.,
by considering only its diagonal to achieve adaptive learning rates. AdaDelta (Zeiler, 2012) and
Adam (Kingma & Ba, 2014) are notable examples of such methods.


1


Published as a conference paper at ICLR 2017



10 0


10 −1


10 −2



Learning rate schedule



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-1-0.png)





![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-1-1.png)

10 −3


10 −4


20 40 60 80 100 120 140 160 180 200

Epochs


Figure 1: Alternative schedule schemes of learning rate _η_ _t_ over batch index _t_ : default schemes
with _η_ 0 = 0 _._ 1 (blue line) and _η_ 0 = 0 _._ 05 (red line) as used by Zagoruyko & Komodakis (2016);
warm restarts simulated every _T_ 0 = 50 (green line), _T_ 0 = 100 (black line) and _T_ 0 = 200 (grey line)
epochs with _η_ _t_ decaying during _i_ -th run from _η_ _max_ _[i]_ [= 0] _[.]_ [05][ to] _[ η]_ _min_ _[i]_ [= 0][ according to eq. (5); warm]
restarts starting from epoch _T_ 0 = 1 (dark green line) and _T_ 0 = 10 (magenta line) with doubling
( _T_ _mult_ = 2) periods _T_ _i_ at every new warm restart.


Intriguingly enough, the current state-of-the-art results on CIFAR-10, CIFAR-100, SVHN, ImageNet, PASCAL VOC and MS COCO datasets were obtained by Residual Neural Networks
(He et al., 2015; Huang et al., 2016c; He et al., 2016; Zagoruyko & Komodakis, 2016) trained without the use of advanced methods such as AdaDelta and Adam. Instead, they simply use SGD with
momentum [1] :


_**v**_ _t_ +1 = _µ_ _t_ _**v**_ _t_ _−_ _η_ _t_ _∇f_ _t_ ( _**x**_ _t_ ) _,_ (3)

_**x**_ _t_ +1 = _**x**_ _t_ + _**v**_ _t_ +1 _,_ (4)


where _**v**_ _t_ is a velocity vector initially set to _**0**_, _η_ _t_ is a decreasing learning rate and _µ_ _t_ is a momentum
rate which defines the trade-off between the current and past observations of _∇f_ _t_ ( _**x**_ _t_ ). The main
difficulty in training a DNN is then associated with the scheduling of the learning rate and the
amount of L2 weight decay regularization employed. A common learning rate schedule is to use a
constant learning rate and divide it by a fixed constant in (approximately) regular intervals. The blue
line in Figure 1 shows an example of such a schedule, as used by Zagoruyko & Komodakis (2016)
to obtain the state-of-the-art results on CIFAR-10, CIFAR-100 and SVHN datasets.


In this paper, we propose to periodically simulate warm restarts of SGD, where in each restart the
learning rate is initialized to some value and is scheduled to decrease. Four different instantiations
of this new learning rate schedule are visualized in Figure 1. Our empirical results suggest that SGD
with warm restarts requires 2 _×_ to 4 _×_ fewer epochs than the currently-used learning rate schedule
schemes to achieve comparable or even better results. Furthermore, combining the networks obtained right before restarts in an ensemble following the approach proposed by Huang et al. (2016a)
improves our results further to 3.14% for CIFAR-10 and 16.21% for CIFAR-100. We also demonstrate its advantages on a dataset of EEG recordings and on a downsampled version of the ImageNet
dataset.


1 More specifically, they employ Nesterov’s momentum (Nesterov, 1983; 2013)


2


Published as a conference paper at ICLR 2017


2 R ELATED W ORK


2.1 R ESTARTS IN GRADIENT - FREE OPTIMIZATION


When optimizing multimodal functions one may want to find all global and local optima. The
tractability of this task depends on the landscape of the function at hand and the budget of function evaluations. Gradient-free optimization approaches based on niching methods (Preuss, 2015)
usually can deal with this task by covering the search space with dynamically allocated niches of
local optimizers. However, these methods usually work only for relatively small search spaces,
e.g., _n <_ 10, and do not scale up due to the curse of dimensionality (Preuss, 2010). Instead, the
current state-of-the-art gradient-free optimizers employ various restart mechanisms (Hansen, 2009;
Loshchilov et al., 2012). One way to deal with multimodal functions is to iteratively sample a large
number _λ_ of candidate solutions, make a step towards better solutions and slowly shape the sampling
distribution to maximize the likelihood of successful steps to appear again (Hansen & Kern, 2004).
The larger the _λ_, the more global search is performed requiring more function evaluations. In order
to achieve good anytime performance, it is common to start with a small _λ_ and increase it (e.g., by
doubling) after each restart. This approach works best on multimodal functions with a global funnel
structure and also improves the results on ill-conditioned problems where numerical issues might
lead to premature convergence when _λ_ is small (Hansen, 2009).


2.2 R ESTARTS IN GRADIENT - BASED OPTIMIZATION


Gradient-based optimization algorithms such as BFGS can also perform restarts to deal with multimodal functions (Ros, 2009). In large-scale settings when the usual number of variables _n_ is on
the order of 10 [3] _−_ 10 [9], the availability of gradient information provides a speedup of a factor of
_n_ w.r.t. gradient-free approaches. Warm restarts are usually employed to improve the convergence
rate rather than to deal with multimodality: often it is sufficient to approach any local optimum to
a given precision and in many cases the problem at hand is unimodal. Fletcher & Reeves (1964)
proposed to flesh the history of conjugate gradient method every _n_ or ( _n_ + 1) iterations. Powell
(1977) proposed to check whether enough orthogonality between _∇f_ ( _**x**_ _t−_ 1 ) and _∇f_ ( _**x**_ _t_ ) has been
lost to warrant another warm restart. Recently, O’Donoghue & Candes (2012) noted that the iterates
of accelerated gradient schemes proposed by Nesterov (1983; 2013) exhibit a periodic behavior if
momentum is overused. The period of the oscillations is proportional to the square root of the local
condition number of the (smooth convex) objective function. The authors showed that fixed warm
restarts of the algorithm with a period proportional to the conditional number achieves the optimal
linear convergence rate of the original accelerated gradient scheme. Since the condition number is
an unknown parameter and its value may vary during the search, they proposed two adaptive warm
restart techniques (O’Donoghue & Candes, 2012):


_•_ **The function scheme** restarts whenever the objective function increases.


_•_ **The gradient scheme** restarts whenever the angle between the momentum term and the
negative gradient is obtuse, i.e, when the momentum seems to be taking us in a bad direction, as measured by the negative gradient at that point. This scheme resembles the one of
Powell (1977) for the conjugate gradient method.


O’Donoghue & Candes (2012) showed (and it was confirmed in a set of follow-up works) that these
simple schemes provide an acceleration on smooth functions and can be adjusted to accelerate stateof-the-art methods such as FISTA on nonsmooth functions.


Smith (2015; 2016) recently introduced cyclical learning rates for deep learning, his approach is
closely-related to our approach in its spirit and formulation but does not focus on restarts.


Yang & Lin (2015) showed that Stochastic subGradient Descent with restarts can achieve a _linear_
_convergence rate_ for a class of non-smooth and non-strongly convex optimization problems where
the epigraph of the objective function is a polyhedron. In contrast to our work, they never increase
the learning rate to perform restarts but decrease it geometrically at each epoch. To perform restarts,
they periodically reset the current solution to the averaged solution from the previous epoch.


3


Published as a conference paper at ICLR 2017


3 S TOCHASTIC G RADIENT D ESCENT WITH WARM RESTARTS (SGDR)


The existing restart techniques can also be used for stochastic gradient descent if the stochasticity
is taken into account. Since gradients and loss values can vary widely from one batch of the data
to another, one should denoise the incoming information: by considering averaged gradients and
losses, e.g., once per epoch, the above-mentioned restart techniques can be used again.


In this work, we consider one of the simplest warm restart approaches. We simulate a new warmstarted run / restart of SGD once _T_ _i_ epochs are performed, where _i_ is the index of the run. Importantly, the restarts are not performed from scratch but emulated by increasing the learning rate _η_ _t_
while the old value of _**x**_ _t_ is used as an initial solution. The amount of this increase controls to which
extent the previously acquired information (e.g., momentum) is used.


Within the _i_ -th run, we decay the learning rate with a cosine annealing for each batch as follows:



_max_ _[−]_ _[η]_ _min_ _[i]_ [)(1 + cos(] _[T]_ _[cur]_
2 [(] _[η]_ _[i]_ _T_



_η_ _t_ = _η_ _min_ _[i]_ [+ 1]



_π_ )) _,_ (5)
_T_ _i_



where _η_ _min_ _[i]_ [and] _[ η]_ _max_ _[i]_ [are ranges for the learning rate, and] _[ T]_ _[cur]_ [accounts for how many epochs]
have been performed since the last restart. Since _T_ _cur_ is updated at each batch iteration _t_, it can
take discredited values such as 0.1, 0.2, etc. Thus, _η_ _t_ = _η_ _max_ _[i]_ [when] _[ t]_ [ = 0][ and] _[ T]_ _[cur]_ [= 0][. Once]
_T_ _cur_ = _T_ _i_, the cos function will output _−_ 1 and thus _η_ _t_ = _η_ _min_ _[i]_ [. The decrease of the learning rate]
is shown in Figure 1 for fixed _T_ _i_ = 50, _T_ _i_ = 100 and _T_ _i_ = 200; note that the logarithmic axis
obfuscates the typical shape of the cosine function.


In order to improve anytime performance, we suggest an option to start with an initially small _T_ _i_
and increase it by a factor of _T_ _mult_ at every restart (see, e.g., Figure 1 for _T_ 0 = 1 _, T_ _mult_ = 2 and
_T_ 0 = 10 _, T_ _mult_ = 2). It might be of great interest to decrease _η_ _max_ _[i]_ [and] _[ η]_ _min_ _[i]_ [at every new restart.]
However, for the sake of simplicity, here, we keep _η_ _max_ _[i]_ [and] _[ η]_ _min_ _[i]_ [the same for every] _[ i]_ [ to reduce the]
number of hyperparameters involved.


Since our simulated warm restarts (the increase of the learning rate) often temporarily worsen performance, we do not always use the last _**x**_ _t_ as our recommendation for the best solution (also called
the _incumbent solution_ ). While our recommendation during the first run (before the first restart) is
indeed the last _**x**_ _t_, our recommendation after this is _a solution obtained at the end of the last per-_
_formed run at η_ _t_ = _η_ _min_ _[i]_ [. We emphasize that with the help of this strategy, our method does not]
require a separate validation data set to determine a recommendation.


4 E XPERIMENTAL RESULTS


4.1 E XPERIMENTAL SETTINGS


We consider the problem of training Wide Residual Neural Networks (WRNs; see Zagoruyko &
Komodakis (2016) for details) on the CIFAR-10 and CIFAR-100 datasets (Krizhevsky, 2009). We
will use the abbreviation WRN- _d_ - _k_ to denote a WRN with depth _d_ and width _k_ . Zagoruyko &
Komodakis (2016) obtained the best results with a WRN-28-10 architecture, i.e., a Residual Neural
Network with _d_ = 28 layers and _k_ = 10 times more filters per layer than used in the original
Residual Neural Networks (He et al., 2015; 2016).


The CIFAR-10 and CIFAR-100 datasets (Krizhevsky, 2009) consist of 32 _×_ 32 color images drawn
from 10 and 100 classes, respectively, split into 50,000 train and 10,000 test images. For image
preprocessing Zagoruyko & Komodakis (2016) performed global contrast normalization and ZCA
whitening. For data augmentation they performed horizontal flips and random crops from the image
padded by 4 pixels on each side, filling missing pixels with reflections of the original image.


For training, Zagoruyko & Komodakis (2016) used SGD with Nesterov’s momentum with initial
learning rate set to _η_ 0 = 0 _._ 1, weight decay to 0.0005, dampening to 0, momentum to 0.9 and
minibatch size to 128. The learning rate is dropped by a factor of 0.2 at 60, 120 and 160 epochs,
with a total budget of 200 epochs. We reproduce the results of Zagoruyko & Komodakis (2016) with
the same settings except that i) we subtract per-pixel mean only and do not use ZCA whitening; ii)
we use SGD with momentum as described by eq. (3-4) and not Nesterov’s momentum.


4


Published as a conference paper at ICLR 2017


WRN−28−10 on CIFAR−10



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-0.png)

25


20


15


10


5





0
50 100 150 200

Epochs


WRN−28−10 on CIFAR−10


5


4.5


4


3.5


3

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-2.png)
50 100 150 200

Epochs


WRN−28−20 on CIFAR−10


5


4.5


4


3.5


3

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-4.png)
50 100 150 200

Epochs



WRN−28−10 on CIFAR−100


50


40


30


20


10


0

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-1.png)
50 100 150 200

Epochs


WRN−28−10 on CIFAR−100


21


20.5


20


19.5


19


18.5


18

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-3.png)
50 100 150 200

Epochs


WRN−28−20 on CIFAR−100


21


20.5


20


19.5


19


18.5


18

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-4-5.png)
50 100 150 200

Epochs



Figure 2: Test errors on CIFAR-10 (left column) and CIFAR-100 (right column) datasets. Note that
for SGDR we only plot the recommended solutions. The top and middle rows show the same results
on WRN-28-10, with the middle row zooming into the good performance region of low test error.
The bottom row shows performance with a wider network, WRN-28-20.
The results of the default learning rate schedules of Zagoruyko & Komodakis (2016) with _η_ 0 = 0 _._ 1
and _η_ 0 = 0 _._ 05 are depicted by the blue and red lines, respectively. The schedules of _η_ _t_ used in SGDR
are shown with i) restarts every _T_ 0 = 50 epochs (green line); ii) restarts every _T_ 0 = 100 epochs
(black line); iii) restarts every _T_ 0 = 200 epochs (gray line); iv) restarts with doubling ( _T_ _mult_ = 2)
periods of restarts starting from the first epoch ( _T_ 0 = 1, dark green line); and v) restarts with
doubling ( _T_ _mult_ = 2) periods of restarts starting from the tenth epoch ( _T_ 0 = 10, magenta line).


The schedule of _η_ _t_ used by Zagoruyko & Komodakis (2016) is depicted by the blue line in Figure 1.
The same schedule but with _η_ 0 = 0 _._ 05 is depicted by the red line. The schedule of _η_ _t_ used in SGDR
is also shown in Figure 1, with two initial learning rates _T_ 0 and two restart doubling periods.


5


Published as a conference paper at ICLR 2017

|Col1|depth-k|# params|# runs|CIFAR-10|CIFAR-100|
|---|---|---|---|---|---|
|original-ResNet (He et al., 2015)|110<br>1202|1.7M<br>10.2M|mean of 5<br>mean of 5|6.43<br>7.93|25.16<br>27.82|
|stoc-depth (Huang et al., 2016c)|110<br>1202|1.7M<br>10.2M|1 run<br>1 run|5.23<br>4.91|24.58<br>n/a|
|pre-act-ResNet (He et al., 2016)|110<br>164<br>1001|1.7M<br>1.7M<br>10.2M|med. of 5<br>med. of 5<br>med. of 5|6.37<br>5.46<br>4.62|n/a<br>24.33<br>22.71|
|WRN (Zagoruyko & Komodakis, 2016)<br>with dropout|16-8<br>28-10<br>28-10|11.0M<br>36.5M<br>36.5M|1 run<br>1 run<br>1 run|4.81<br>4.17<br>n/a|22.07<br>20.50<br>20.04|
|WRN (ours)<br>default with_ η_0 = 0_._1<br>default with_ η_0 = 0_._05<br>_T_0 = 50_, Tmult_ = 1<br>_T_0 = 100_, Tmult_ = 1<br>_T_0 = 200_, Tmult_ = 1<br>_T_0 = 1_, Tmult_ = 2<br>_T_0 = 10_, Tmult_ = 2|28-10<br>28-10<br>28-10<br>28-10<br>28-10<br>28-10<br>28-10|36.5M<br>36.5M<br>36.5M<br>36.5M<br>36.5M<br>36.5M<br>36.5M|med. of 5<br>med. of 5<br>med. of 5<br>med. of 5<br>med. of 5<br>med. of 5<br>med. of 5|4.24<br>4.13<br>4.17<br>4.07<br>3.86<br>4.09<br>4.03|20.33<br>20.21<br>19.99<br>19.87<br>19.98<br>19.74<br>19.58|
|default with_ η_0 = 0_._1<br>default with_ η_0 = 0_._05<br>_T_0 = 50_, Tmult_ = 1<br>_T_0 = 100_, Tmult_ = 1<br>_T_0 = 200_, Tmult_ = 1<br>_T_0 = 1_, Tmult_ = 2<br>_T_0 = 10_, Tmult_ = 2|28-20<br>28-20<br>28-20<br>28-20<br>28-20<br>28-20<br>28-20|145.8M<br>145.8M<br>145.8M<br>145.8M<br>145.8M<br>145.8M<br>145.8M|med. of 2<br>med. of 2<br>med. of 2<br>med. of 2<br>med. of 2<br>med. of 2<br>med. of 2|4.08<br>3.96<br>4.01<br>**3.77**<br>**3.66**<br>3.91<br>**3.74**|19.53<br>19.67<br>19.28<br>19.24<br>19.69<br>**18.90**<br>**18.70**|



Table 1: Test errors of different methods on CIFAR-10 and CIFAR-100 with moderate data augmentation (flip/translation). In the second column _k_ is a widening factor for WRNs. Note that the
computational and memory resources used to train all WRN-28-10 are the same. In all other cases
they are different, but WRNs are usually faster than original ResNets to achieve the same accuracy
(e.g., up to a factor of 8 according to Zagoruyko & Komodakis (2016)). Bold text is used only to
highlight better results and is not based on statistical tests (too few runs).


4.2 S INGLE -M ODEL R ESULTS


Table 1 shows that our experiments reproduce the results given by Zagoruyko & Komodakis (2016)
for WRN-28-10 both on CIFAR-10 and CIFAR-100. These “default” experiments with _η_ 0 = 0 _._ 1
and _η_ 0 = 0 _._ 05 correspond to the blue and red lines in Figure 2. The results for _η_ 0 = 0 _._ 05 show
better performance, and therefore we use _η_ 0 = 0 _._ 05 in our later experiments.


SGDR with _T_ 0 = 50, _T_ 0 = 100 and _T_ 0 = 200 for _T_ _mult_ = 1 perform warm restarts every 50, 100
and 200 epochs, respectively. A single run of SGD with the schedule given by eq. (5) for _T_ 0 = 200
shows the best results suggesting that the original schedule of WRNs might be suboptimal w.r.t. the
test error in these settings. However, the same setting with _T_ 0 = 200 leads to the worst anytime
performance except for the very last epochs.


SGDR with _T_ 0 = 1 _, T_ _mult_ = 2 and _T_ 0 = 10 _, T_ _mult_ = 2 performs its first restart after 1 and 10
epochs, respectively. Then, it doubles the maximum number of epochs for every new restart. The
main purpose of this doubling is to reach good test error as soon as possible, i.e., achieve good
anytime performance. Figure 2 shows that this is achieved and test errors around 4% on CIFAR-10
and around 20% on CIFAR-100 can be obtained about 2-4 times faster than with the default schedule
used by Zagoruyko & Komodakis (2016).


6


Published as a conference paper at ICLR 2017

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-6-0.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-6-1.png)


Figure 3: Test errors of ensemble models built from _N_ runs of SGDR on WRN-28-10 with _M_
model snapshots per run made at epochs 150, 70 and 30 (right before warm restarts of SGDR as
suggested by Huang et al. (2016a)). When _M_ =1 (respectively, _M_ =2), we aggregate probabilities of
softmax layers of snapshot models at epoch index 150 (respectively, at epoch indexes 150 and 70).

|Col1|CIFAR-10|CIFAR-100|
|---|---|---|
|_N_ = 1 run of WRN-28-10 with_ M_ = 1 snapshot (median of 16 runs)<br>_N_ = 1 run of WRN-28-10 with_ M_ = 3 snapshots per run<br>_N_ = 3 runs of WRN-28-10 with_ M_ = 3 snapshots per run<br>_N_ = 16 runs of WRN-28-10 with_ M_ = 3 snapshots per run|4.03<br>3.51<br>3.25<br>**3.14**|19.57<br>17.75<br>16.64<br>**16.21**|



Table 2: Test errors of ensemble models on CIFAR-10 and CIFAR-100 datasets.


Since SGDR achieves good performance faster, it may allow us to train larger networks. We therefore investigated whether results on CIFAR-10 and CIFAR-100 can be further improved by making
WRNs two times wider, i.e., by training WRN-28-20 instead of WRN-28-10. Table 1 shows that
the results indeed improved, by about 0.25% on CIFAR-10 and by about 0.5-1.0% on CIFAR-100.
While network architecture WRN-28-20 requires roughly three-four times more computation than
WRN-28-10, the aggressive learning rate reduction of SGDR nevertheless allowed us to achieve a
better error rate in the same time on WRN-28-20 as we spent on 200 epochs of training on WRN28-10. Specifically, Figure 2 (right middle and right bottom) show that after only 50 epochs, SGDR
(even without restarts, using _T_ 0 = 50 _, T_ _mult_ = 1) achieved an error rate below 19% (whereas none
of the other learning methods performed better than 19.5% on WRN-28-10). We therefore have hope
that – by enabling researchers to test new architectures faster – SGDR’s good anytime performance
may also lead to improvements of the state of the art.


In a final experiment for SGDR by itself, Figure 7 in the appendix compares SGDR and the default schedule with respect to training and test performance. As the figure shows, SGDR optimizes
training loss faster than the standard default schedule until about epoch 120. After this, the default
schedule overfits, as can be seen by an increase of the test error both on CIFAR-10 and CIFAR-100
(see, e.g., the right middle plot of Figure 7). In contrast, we only witnessed very mild overfitting for
SGDR.


4.3 E NSEMBLE R ESULTS


Our initial arXiv report on SGDR (Loshchilov & Hutter, 2016) inspired a follow-up study by Huang
et al. (2016a) in which the authors suggest to take _M_ snapshots of the models obtained by SGDR (in
their paper referred to as cyclical learning rate schedule and cosine annealing cycles) right before
_M_ last restarts and to use those to build an ensemble, thereby obtaining ensembles “for free” (in
contrast to having to perform multiple independent runs). The authors demonstrated new state-of

7


Published as a conference paper at ICLR 2017


the-art results on CIFAR datasets by making ensembles of DenseNet models (Huang et al., 2016b).
Here, we investigate whether their conclusions hold for WRNs used in our study. We used WRN28-10 trained by SGDR with _T_ 0 = 10 _, T_ _mult_ = 2 as our baseline model.


Figure 3 and Table 2 aggregate the results of our study. The original test error of 4.03% on CIFAR-10
and 19.57% on CIFAR-100 (median of 16 runs) can be improved to 3.51% on CIFAR-10 and 17.75%
on CIFAR-100 when _M_ = 3 snapshots are taken at epochs 30, 70 and 150: when the learning rate
of SGDR with _T_ 0 = 10 _, T_ _mult_ = 2 is scheduled to achieve 0 (see Figure 1) and the models are
used with uniform weights to build an ensemble. To achieve the same result, one would have to
aggregate _N_ = 3 models obtained at epoch 150 of _N_ = 3 independent runs (see _N_ = 3 _, M_ = 1 in
Figure 3). Thus, the aggregation from snapshots provides a 3-fold speedup in these settings because
additional ( _M >_ 1-th) snapshots from a single SGDR run are computationally free. Interestingly,
aggregation of models from independent runs (when _N >_ 1 and _M_ = 1) does not scale up as well
as from _M >_ 1 snapshots of independent runs when the same number of models is considered: the
case of _N_ = 3 and _M_ = 3 provides better performance than the cases of _M_ = 1 with _N_ = 18
and _N_ = 21. Not only the number of snapshots _M_ per run but also their origin is crucial. Thus,
naively building ensembles from models obtained at last epochs only (i.e., _M_ = 3 snapshots at
epochs 148, 149, 150) did not improve the results (i.e., the baseline of _M_ = 1 snapshot at 150)
thereby confirming the conclusion of Huang et al. (2016a) that snapshots of SGDR provide a useful
diversity of predictions for ensembles.


Three runs ( _N_ = 3) of SGDR with _M_ = 3 snapshots per run are sufficient to greatly improve the
results to 3.25% on CIFAR-10 and 16.64% on CIFAR-100 outperforming the results of Huang et al.
(2016a). By increasing _N_ to 16 one can achieve 3.14% on CIFAR-10 and 16.21% on CIFAR-100.
We believe that these results could be further improved by considering better baseline models than
WRN-28-10 (e.g., WRN-28-20).


4.4 E XPERIMENTS ON A DATASET OF EEG RECORDINGS


To demonstrate the generality of SGDR, we also considered a very different domain: a dataset of
electroencephalographic (EEG) recordings of brain activity for classification of actual right and left
hand and foot movements of 14 subjects with roughly 1000 trials per subject (Schirrmeister et al.,
2017). The best classification results obtained with the original pipeline based on convolutional neural networks designed by Schirrmeister et al. (2017) were used as our reference. First, we compared
the baseline learning rate schedule with different settings of the total number of epochs and initial
learning rates (see Figure 4). When 30 epochs were considered, we dropped the learning rate by
a factor of 10 at epoch indexes 10, 15 and 20. As expected, with more epochs used and a similar
(budget proportional) schedule better results can be achieved. Alternatively, one can consider SGDR
and get a similar final performance while having a better anytime performance without defining the
total budget of epochs beforehand.


Similarly to our results on the CIFAR datasets, our experiments with the EEG data confirm that
snapshots are useful and the median reference error (about 9%) can be improved i) by 1-2% when
model snapshots of a single run are considered, and ii) by 2-3% when model snapshots from both
hyperparameter settings are considered. The latter would correspond to _N_ = 2 in Section (4.3).


4.5 P RELIMINARY EXPERIMENTS ON A DOWNSAMPLED I MAGE N ET DATASET


In order to additionally validate our SGDR on a larger dataset, we constructed a downsampled
version of the ImageNet dataset [P. Chrabaszcz, I. Loshchilov and F. Hutter. A Downsampled Variant
of ImageNet as an Alternative to the CIFAR datasets., _in preparation_ ]. In contrast to earlier attempts
(Pouransari & Ghili, 2015), our downsampled ImageNet contains exactly the same images from
1000 classes as the original ImageNet but resized with box downsampling to 32 _×_ 32 pixels. Thus,
this dataset is substantially harder than the original ImageNet dataset because the average number
of pixels per image is now two orders of magnitude smaller. The new dataset is also more difficult
than the CIFAR datasets because more classes are used and the relevant objects to be classified often
cover only a tiny subspace of the image and not most of the image as in the CIFAR datasets.


We benchmarked SGD with momentum with the default learning rate schedule, SGDR with _T_ 0 =
1 _, T_ _mult_ = 2 and SGDR with _T_ 0 = 10 _, T_ _mult_ = 2 on WRN-28-10, all trained with 4 settings of


8


Published as a conference paper at ICLR 2017

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-8-0.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-8-1.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-8-2.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-8-3.png)


Figure 4: ( **Top** ) Improvements obtained by the baseline learning rate schedule and SGDR w.r.t. the
best known reference classification error on a dataset of electroencephalographic (EEG) recordings
of brain activity for classification of actual right and left hand and foot movements of 14 subjects
with roughly 1000 trials per subject. Both considered approaches were tested with the initial learning rate _lr_ = 0 _._ 025 ( **Top-Left** ) and _lr_ = 0 _._ 05 ( **Top-Right** ). Note that the baseline approach is
considered with different settings of the total number of epochs: 30, 60, _. . ._, 480. ( **Bottom** ) SGDR
with _lr_ = 0 _._ 025 and _lr_ = 0 _._ 05 without and with _M_ model snapshots taken at the last _M_ = _nr/_ 2
restarts, where _nr_ is the total number of restarts.


the initial learning rate _η_ _max_ _[i]_ [: 0.050, 0.025, 0.01 and 0.005. We used the same data augmentation]
procedure as for the CIFAR datasets. Similarly to the results on the CIFAR datasets, Figure 5 shows
that SGDR demonstrates better anytime performance. SGDR with _T_ 0 = 10 _, T_ _mult_ = 2 _, η_ _max_ _[i]_ [=]
0 _._ 01 achieves top-1 error of 39.24% and top-5 error of 17.17% matching the original results by
AlexNets (40.7% and 18.2%, respectively) obtained on the original ImageNet with full-size images
of ca. 50 times more pixels per image (Krizhevsky et al., 2012b). Interestingly, when the dataset is
permuted only within 10 subgroups each formed from 100 classes, SGDR also demonstrates better
results (see Figure 8 in the Supplementary Material). An interpretation of this might be that while
the initial learning rate seems to be very important, SGDR reduces the problem of improper selection
of the latter by scanning / annealing from the initial learning rate to 0.


Clearly, longer runs (more than 40 epochs considered in this preliminary experiment) and hyperparameter tuning of learning rates, regularization and other hyperparameters shall further improve the
results.


9


Published as a conference paper at ICLR 2017


WRN−28−10 on downsampled 32x32 ImageNet

60



WRN−28−10 on downsampled 32x32 ImageNet

|Col1|Col2|Col3|Col4|Col5|Col6|Defau<br>SGD<br>SGD|lt<br>R T = 1, T<br>0<br>R T = 10,<br>0|= 2<br>mult<br>T = 2<br>mult|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||



5 10 15 20 25 30 35

Epochs



40


35


30


25


20



55


50


45


40




|Col1|Col2|Col3|Col4|Col5|Col6|Defau<br>SGD<br>SGD|lt<br>R T = 1, T<br>0<br>R T = 10,<br>0|= 2<br>mult<br>T = 2<br>mult|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||



35
5 10 15 20 25 30 35

Epochs



Figure 5: Top-1 and Top-5 test errors obtained by SGD with momentum with the default learning
rate schedule, SGDR with _T_ 0 = 1 _, T_ _mult_ = 2 and SGDR with _T_ 0 = 10 _, T_ _mult_ = 2 on WRN-28-10
trained on a version of ImageNet, with all images from all 1000 classes downsampled to 32 _×_ 32
pixels. The same baseline data augmentation as for the CIFAR datasets is used. Four settings of the
initial learning rate are considered: 0.050, 0.025, 0.01 and 0.005.


5 D ISCUSSION


Our results suggest that _even without any restarts_ the proposed aggressive learning rate schedule
given by eq. (5) is competitive w.r.t. the default schedule when training WRNs on the CIFAR10 (e.g., for _T_ 0 = 200 _, T_ _mult_ = 1) and CIFAR-100 datasets. In practice, the proposed schedule
requires only two hyper-parameters to be defined: the initial learning rate and the total number of
epochs.


We found that the anytime performance of SGDR remain similar when shorter epochs are considered
(see section 8.1 in the Supplemenary Material).


One _should not_ suppose that the parameter values used in this study and many other works with
(Residual) Neural Networks are selected to demonstrate the fastest decrease of the training error.
Instead, the best validation or / and test errors are in focus. Notably, the validation error is rarely
used when training Residual Neural Networks because the recommendation is defined by the final
solution (in our approach, the final solution of each run). One could use the validation error to
determine the optimal initial learning rate and then run on the whole dataset; this could further
improve results.


The main purpose of our proposed warm restart scheme for SGD is to improve its anytime performance. While we mentioned that restarts can be useful to deal with multi-modal functions, _we do_
_not claim_ that we observe any effect related to multi-modality.


As we noted earlier, one could decrease _η_ _max_ _[i]_ [and] _[ η]_ _min_ _[i]_ [at every new warm restart to control the]
amount of divergence. If new restarts are worse than the old ones w.r.t. validation error, then one
might also consider going back to the last best solution and perform a new restart with adjusted
hyperparameters.


Our results reproduce the finding by Huang et al. (2016a) that intermediate models generated by
SGDR can be used to build efficient ensembles at no cost. This finding makes SGDR especially
attractive for scenarios when ensemble building is considered.


6 C ONCLUSION


In this paper, we investigated a simple warm restart mechanism for SGD to accelerate the training of
DNNs. Our SGDR simulates warm restarts by scheduling the learning rate to achieve competitive
results on CIFAR-10 and CIFAR-100 roughly two to four times faster. We also achieved new stateof-the-art results with SGDR, mainly by using even wider WRNs and ensembles of snapshots from


10


Published as a conference paper at ICLR 2017


SGDR’s trajectory. Future empirical studies should also consider the SVHN, ImageNet and MS
COCO datasets, for which Residual Neural Networks showed the best results so far. Our preliminary
results on a dataset of EEG recordings suggest that SGDR delivers better and better results as we
carry out more restarts and use more model snapshots. The results on our downsampled ImageNet
dataset suggest that SGDR might also reduce the problem of learning rate selection because the
annealing and restarts of SGDR scan / consider a range of learning rate values. Future work should
consider warm restarts for other popular training algorithms such as AdaDelta (Zeiler, 2012) and
Adam (Kingma & Ba, 2014).


Alternative network structures should be also considered; e.g., soon after our initial arXiv report
(Loshchilov & Hutter, 2016), Zhang et al. (2016); Huang et al. (2016b); Han et al. (2016) reported
that WRNs models can be replaced by more memory-efficient models. Thus, it should be tested
whether our results for individual models and ensembles can be further improved by using their
networks instead of WRNs. Deep compression methods (Han et al., 2015) can be used to reduce the
time and memory costs of DNNs and their ensembles.


7 A CKNOWLEDGMENTS


This work was supported by the German Research Foundation (DFG), under the BrainLinksBrainTools Cluster of Excellence (grant number EXC 1086). We thank Gao Huang, Kilian Quirin Weinberger, Jost Tobias Springenberg, Mark Schmidt and three anonymous reviewers for their helpful
comments and suggestions. We thank Robin Tibor Schirrmeister for providing his pipeline for the
EEG experiments and helping integrating SGDR.


R EFERENCES


Antoine Bordes, L´eon Bottou, and Patrick Gallinari. Sgd-qn: Careful quasi-newton stochastic gradient descent. _The Journal of Machine Learning Research_, 10:1737–1754, 2009.


Anna Choromanska, Mikael Henaff, Michael Mathieu, G´erard Ben Arous, and Yann LeCun. The
loss surface of multilayer networks. _arXiv preprint arXiv:1412.0233_, 2014.


Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua
Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In _Advances in Neural Information Processing Systems_, pp. 2933–2941, 2014.


Yann N Dauphin, Harm de Vries, Junyoung Chung, and Yoshua Bengio. Rmsprop and equilibrated
adaptive learning rates for non-convex optimization. _arXiv preprint arXiv:1502.04390_, 2015.


L. Deng, G. Hinton, and B. Kingsbury. New types of deep neural network learning for speech
recognition and related applications: An overview. In _Proc. of ICASSP’13_, 2013.


J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep
convolutional activation feature for generic visual recognition. In _Proc. of ICML’14_, 2014.


Reeves Fletcher and Colin M Reeves. Function minimization by conjugate gradients. _The computer_
_journal_, 7(2):149–154, 1964.


Kenji Fukumizu and Shun-ichi Amari. Local minima and plateaus in hierarchical structures of
multilayer perceptrons. _Neural Networks_, 13(3):317–327, 2000.


Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep pyramidal residual networks. _arXiv preprint_
_arXiv:1610.02915_, 2016.


Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks
with pruning, trained quantization and huffman coding. _arXiv preprint arXiv:1510.00149_, 2015.


Nikolaus Hansen. Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed. In
_Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computa-_
_tion Conference: Late Breaking Papers_, pp. 2389–2396. ACM, 2009.


11


Published as a conference paper at ICLR 2017


Nikolaus Hansen and Stefan Kern. Evaluating the cma evolution strategy on multimodal test
functions. In _International Conference on Parallel Problem Solving from Nature_, pp. 282–291.
Springer, 2004.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. _arXiv preprint arXiv:1512.03385_, 2015.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual
networks. _arXiv preprint arXiv:1603.05027_, 2016.


Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, and Kilian Q. Weinberger.
Snapshot ensembles: Train 1, get m for free. _ICLR 2017 submission_, 2016a.


Gao Huang, Zhuang Liu, and Kilian Q Weinberger. Densely connected convolutional networks.
_arXiv preprint arXiv:1608.06993_, 2016b.


Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Weinberger. Deep networks with stochastic depth. _arXiv preprint arXiv:1603.09382_, 2016c.


Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint_
_arXiv:1412.6980_, 2014.


A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural
networks. In _Proc. of NIPS’12_, pp. 1097–1105, 2012a.


Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.


Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In _Advances in neural information processing systems_, pp. 1097–1105,
2012b.


Dong C Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization.
_Mathematical programming_, 45(1-3):503–528, 1989.


Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent with Restarts. _arXiv preprint_
_arXiv:1608.03983_, 2016.


Ilya Loshchilov, Marc Schoenauer, and Michele Sebag. Alternative restart strategies for CMA-ES.
In _International Conference on Parallel Problem Solving from Nature_, pp. 296–305. Springer,
2012.


Yurii Nesterov. A method of solving a convex programming problem with convergence rate o (1/k2).
In _Soviet Mathematics Doklady_, volume 27, pp. 372–376, 1983.


Yurii Nesterov. _Introductory lectures on convex optimization: A basic course_, volume 87. Springer
Science & Business Media, 2013.


Brendan O’Donoghue and Emmanuel Candes. Adaptive restart for accelerated gradient schemes.
_arXiv preprint arXiv:1204.3982_, 2012.


Hadi Pouransari and Saman Ghili. Tiny imagenet visual recognition challenge. _CS231 course at_
_STANFORD_, 2015.


Michael James David Powell. Restart procedures for the conjugate gradient method. _Mathematical_
_programming_, 12(1):241–254, 1977.


Mike Preuss. Niching the CMA-ES via nearest-better clustering. In _Proceedings of the 12th annual_
_conference companion on Genetic and evolutionary computation_, pp. 1711–1718. ACM, 2010.


Mike Preuss. Niching methods and multimodal optimization performance. In _Multimodal Optimiza-_
_tion by Means of Evolutionary Algorithms_, pp. 115–137. Springer, 2015.


Raymond Ros. Benchmarking the bfgs algorithm on the bbob-2009 function testbed. In _Proceed-_
_ings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Con-_
_ference: Late Breaking Papers_, pp. 2409–2414. ACM, 2009.


12


Published as a conference paper at ICLR 2017


Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer, Martin
Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, and
Tonio Ball. Deep learning with convolutional neural networks for brain mapping and decoding of
movement-related information from the human eeg. _arXiv preprint arXiv:1703.05051_, 2017.


Leslie N Smith. No more pesky learning rate guessing games. _arXiv preprint arXiv:1506.01186_,
2015.


Leslie N Smith. Cyclical learning rates for training neural networks. _arXiv preprint_
_arXiv:1506.01186v3_, 2016.


Tianbao Yang and Qihang Lin. Stochastic subgradient methods with linear convergence for polyhedral convex optimization. _arXiv preprint arXiv:1510.01444_, 2015.


Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. _arXiv preprint_
_arXiv:1605.07146_, 2016.


Matthew D Zeiler. Adadelta: An adaptive learning rate method. _arXiv preprint arXiv:1212.5701_,
2012.


K. Zhang, M. Sun, T. X. Han, X. Yuan, L. Guo, and T. Liu. Residual Networks of Residual Networks: Multilevel Residual Networks. _ArXiv e-prints_, August 2016.


13


Published as a conference paper at ICLR 2017


8 S UPPLEMENTARY M ATERIAL


30


25


20


15


10



CIFAR−10



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-13-0.png)



5
0 20 40 60 80 100

Epochs


Figure 6: The median results of 5 runs for the best learning rate settings considered for WRN-28-1.


8.1 50 K VS 100 K EXAMPLES PER EPOCH


Our data augmentation procedure code is inherited from the Lasagne Recipe code for ResNets where
flipped images are added to the training set. This doubles the number of training examples per epoch
and thus might impact the results because hyperparameter values defined as a function of epoch
index have a different meaning. While our experimental results given in Table 1 reproduced the
results obtained by Zagoruyko & Komodakis (2016), here we test whether SGDR still makes sense
for WRN-28-1 (i.e., ResNet with 28 layers) where one epoch corresponds to 50k training examples.
We investigate different learning rate values for the default learning rate schedule (4 values out of

[0.01, 0.025, 0.05, 0.1]) and SGDR (3 values out of [0.025, 0.05, 0.1]). In line with the results given
in the main paper, Figure 6 suggests that SGDR is competitive in terms of anytime performance.


14


Published as a conference paper at ICLR 2017


WRN−28−10 on CIFAR−10



1


0.8


0.6


0.4


0.2



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-0.png)



0
50 100 150 200

Epochs


WRN−28−10 on CIFAR−10



0.2


0.19


0.18


0.17


0.16


0.15


5


4.5


4


3.5



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-2.png)

50 100 150 200

Epochs


WRN−28−10 on CIFAR−10



WRN−28−10 on CIFAR−100


1


0.8


0.6


0.4


0.2


0

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-1.png)
50 100 150 200

Epochs


WRN−28−10 on CIFAR−100


1.1


1


0.9


0.8


0.7

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-3.png)
50 100 150 200

Epochs


WRN−28−10 on CIFAR−100


21


20.5


20


19.5


19


18.5


18

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-5.png)
50 100 150 200

Epochs



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-14-4.png)

3
50 100 150 200

Epochs



Figure 7: Training cross-entropy + regularization loss (top row), test loss (middle row) and test
error (bottom row) on CIFAR-10 (left column) and CIFAR-100 (right column).


15


Published as a conference paper at ICLR 2017


WRN−28−10 on downsampled 32x32 ImageNet

100


95


90


85



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/COSINE/COSINE.pdf-15-0.png)

80


75





70

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||050<br>~~15~~|050<br>~~15~~|||||
||Defa<br>~~Defa~~|Defa<br>~~Defa~~|ult, lr=0.<br>~~ult, lr=0.~~|050<br>~~15~~|050<br>~~15~~|050<br>~~15~~|050<br>~~15~~|050<br>~~15~~|050<br>~~15~~|
||Defa<br>SGD<br>|Defa<br>SGD<br>|ult, lr=0.<br>R, lr=0.0<br>|005<br>50<br>||||||
||Defa<br>SGD<br>|Defa<br>SGD<br>|ult, lr=0.<br>R, lr=0.0<br>|005<br>50<br>||||||
||~~SGD~~<br>SGD|~~SGD~~<br>SGD|~~R, lr=0.0~~<br>R, lr=0.0|~~15~~<br>05|~~15~~<br>05|||||
|||||||||||

5 10 15 20 25 30 35 40

Epochs


Figure 8: Top-5 test errors obtained by SGD with momentum with the default learning rate schedule
and SGDR with _T_ 0 = 1 _, T_ _mult_ = 2 on WRN-28-10 trained on a version of ImageNet, with all
images from all 1000 classes downsampled to 32 _×_ 32 pixels. The same baseline data augmentation
as for the CIFAR datasets is used. Three settings of the initial learning rate are considered: 0.050,
0.015 and 0.005. In contrast to the experiments described in the main paper, here, the dataset is
permuted only within 10 subgroups each formed from 100 classes which makes good generalization
much harder to achieve for both algorithms. An interpretation of SGDR results given here might
be that while the initial learning rate seems to be very important, SGDR reduces the problem of
improper selection of the latter by scanning / annealing from the initial learning rate to 0.


16


