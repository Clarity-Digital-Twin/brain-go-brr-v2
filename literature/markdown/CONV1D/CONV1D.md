#### **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision** **Applications**

Andrew G. Howard Menglong Zhu Bo Chen Dmitry Kalenichenko
Weijun Wang Tobias Weyand Marco Andreetto Hartwig Adam


Google Inc.


_{_ howarda,menglong,bochen,dkalenichenko,weijunw,weyand,anm,hadam _}_ @google.com



**Abstract**


_We present a class of efficient models called MobileNets_
_for mobile and embedded vision applications. MobileNets_
_are based on a streamlined architecture that uses depth-_
_wise separable convolutions to build light weight deep_
_neural networks. We introduce two simple global hyper-_
_parameters that efficiently trade off between latency and_
_accuracy. These hyper-parameters allow the model builder_
_to choose the right sized model for their application based_
_on the constraints of the problem._ _We present extensive_
_experiments on resource and accuracy tradeoffs and show_
_strong performance compared to other popular models on_
_ImageNet classification. We then demonstrate the effective-_
_ness of MobileNets across a wide range of applications and_
_use cases including object detection, finegrain classifica-_
_tion, face attributes and large scale geo-localization._


**1. Introduction**


Convolutional neural networks have become ubiquitous
in computer vision ever since AlexNet [19] popularized
deep convolutional neural networks by winning the ImageNet Challenge: ILSVRC 2012 [24]. The general trend
has been to make deeper and more complicated networks
in order to achieve higher accuracy [27, 31, 29, 8]. However, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and
speed. In many real world applications such as robotics,
self-driving car and augmented reality, the recognition tasks
need to be carried out in a timely fashion on a computationally limited platform.
This paper describes an efficient network architecture
and a set of two hyper-parameters in order to build very
small, low latency models that can be easily matched to the
design requirements for mobile and embedded vision applications. Section 2 reviews prior work in building small



models. Section 3 describes the MobileNet architecture and

two hyper-parameters width multiplier and resolution multiplier to define smaller and more efficient MobileNets. Section 4 describes experiments on ImageNet as well a variety
of different applications and use cases. Section 5 closes
with a summary and conclusion.


**2. Prior Work**


There has been rising interest in building small and efficient neural networks in the recent literature, e.g. [16, 34,
12, 36, 22]. Many different approaches can be generally
categorized into either compressing pretrained networks or
training small networks directly. This paper proposes a
class of network architectures that allows a model devel
oper to specifically choose a small network that matches
the resource restrictions (latency, size) for their application.
MobileNets primarily focus on optimizing for latency but
also yield small networks. Many papers on small networks
focus only on size but do not consider speed.
MobileNets are built primarily from depthwise separable
convolutions initially introduced in [26] and subsequently
used in Inception models [13] to reduce the computation in
the first few layers. Flattened networks [16] build a network
out of fully factorized convolutions and showed the potential of extremely factorized networks. Independent of this
current paper, Factorized Networks[34] introduces a similar
factorized convolution as well as the use of topological connections. Subsequently, the Xception network [3] demonstrated how to scale up depthwise separable filters to out
perform Inception V3 networks. Another small network is
Squeezenet [12] which uses a bottleneck approach to design
a very small network. Other reduced computation networks
include structured transform networks [28] and deep fried
convnets [37].
A different approach for obtaining small networks is
shrinking, factorizing or compressing pretrained networks.
Compression based on product quantization [36], hashing



1


![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-4.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-10.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-0.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-3.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-9.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-1.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-2.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-6.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-5.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-7.png)

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-11.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-1-8.png)






[2], and pruning, vector quantization and Huffman coding

[5] have been proposed in the literature. Additionally various factorizations have been proposed to speed up pretrained networks [14, 20]. Another method for training
small networks is distillation [9] which uses a larger network to teach a smaller network. It is complementary to
our approach and is covered in some of our use cases in
section 4. Another emerging approach is low bit networks

[4, 22, 11].


**3. MobileNet Architecture**


In this section we first describe the core layers that MobileNet is built on which are depthwise separable filters.
We then describe the MobileNet network structure and con
clude with descriptions of the two model shrinking hyperparameters width multiplier and resolution multiplier.


**3.1. Depthwise Separable Convolution**


The MobileNet model is based on depthwise separable
convolutions which is a form of factorized convolutions

which factorize a standard convolution into a depthwise
convolution and a 1 _×_ 1 convolution called a pointwise convolution. For MobileNets the depthwise convolution applies a single filter to each input channel. The pointwise
convolution then applies a 1 _×_ 1 convolution to combine the
outputs the depthwise convolution. A standard convolution
both filters and combines inputs into a new set of outputs
in one step. The depthwise separable convolution splits this
into two layers, a separate layer for filtering and a separate
layer for combining. This factorization has the effect of
drastically reducing computation and model size. Figure 2
shows how a standard convolution 2(a) is factorized into a
depthwise convolution 2(b) and a 1 _×_ 1 pointwise convolution 2(c).
A standard convolutional layer takes as input a _D_ _F_ _×_



_D_ _F_ _× M_ feature map **F** and produces a _D_ _F_ _× D_ _F_ _× N_
feature map **G** where _D_ _F_ is the spatial width and height
of a square input feature map [1], _M_ is the number of input
channels (input depth), _D_ _G_ is the spatial width and height of
a square output feature map and _N_ is the number of output
channel (output depth).
The standard convolutional layer is parameterized by
convolution kernel **K** of size _D_ _K_ _×D_ _K_ _×M_ _×N_ where _D_ _K_
is the spatial dimension of the kernel assumed to be square
and _M_ is number of input channels and _N_ is the number of
output channels as defined previously.
The output feature map for standard convolution assuming stride one and padding is computed as:


**G** _k,l,n_ = � **K** _i,j,m,n_ _·_ **F** _k_ + _i−_ 1 _,l_ + _j−_ 1 _,m_ (1)

_i,j,m_


Standard convolutions have the computational cost of:


_D_ _K_ _· D_ _K_ _· M · N · D_ _F_ _· D_ _F_ (2)


where the computational cost depends multiplicatively on
the number of input channels _M_, the number of output
channels _N_ the kernel size _D_ _k_ _× D_ _k_ and the feature map
size _D_ _F_ _× D_ _F_ . MobileNet models address each of these
terms and their interactions. First it uses depthwise separable convolutions to break the interaction between the num
ber of output channels and the size of the kernel.
The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation.
The filtering and combination steps can be split into two
steps via the use of factorized convolutions called depthwise


1 We assume that the output feature map has the same spatial dimensions as the input and both feature maps are square. Our model shrinking
results generalize to feature maps with arbitrary sizes and aspect ratios.


separable convolutions for substantial reduction in computational cost.

Depthwise separable convolution are made up of two
layers: depthwise convolutions and pointwise convolutions.
We use depthwise convolutions to apply a single filter per
each input channel (input depth). Pointwise convolution, a
simple 1 _×_ 1 convolution, is then used to create a linear combination of the output of the depthwise layer. MobileNets
use both batchnorm and ReLU nonlinearities for both lay
ers.

Depthwise convolution with one filter per input channel
(input depth) can be written as:


ˆ ˆ
**G** _k,l,m_ = � **K** _i,j,m_ _·_ **F** _k_ + _i−_ 1 _,l_ + _j−_ 1 _,m_ (3)

_i,j_


where **K** [ˆ] is the depthwise convolutional kernel of size
_D_ _K_ _× D_ _K_ _× M_ where the _m_ _th_ filter in **K** [ˆ] is applied to
the _m_ _th_ channel in **F** to produce the _m_ _th_ channel of the
filtered output feature map **G** [ˆ] .
Depthwise convolution has a computational cost of:


_D_ _K_ _· D_ _K_ _· M · D_ _F_ _· D_ _F_ (4)


Depthwise convolution is extremely efficient relative to
standard convolution. However it only filters input channels, it does not combine them to create new features. So
an additional layer that computes a linear combination of
the output of depthwise convolution via 1 _×_ 1 convolution
is needed in order to generate these new features.
The combination of depthwise convolution and 1 _×_ 1
(pointwise) convolution is called depthwise separable convolution which was originally introduced in [26].
Depthwise separable convolutions cost:


_D_ _K_ _· D_ _K_ _· M · D_ _F_ _· D_ _F_ + _M · N · D_ _F_ _· D_ _F_ (5)


which is the sum of the depthwise and 1 _×_ 1 pointwise convolutions.

By expressing convolution as a two step process of filtering and combining we get a reduction in computation of:


_D_ _K_ _· D_ _K_ _· M · D_ _F_ _· D_ _F_ + _M · N · D_ _F_ _· D_ _F_


_D_ _K_ _· D_ _K_ _· M · N · D_ _F_ _· D_ _F_


1 1

=
_N_ [+] _D_ _K_ [2]


MobileNet uses 3 _×_ 3 depthwise separable convolutions
which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy as
seen in Section 4.

Additional factorization in spatial dimension such as in

[16, 31] does not save much additional computation as very
little computation is spent in depthwise convolutions.


|Col1|Col2|
|---|---|
|||
|||
|_N_|_N_|


|Col1|Col2|
|---|---|
|||
|||
|||
|||
|||
|_DK_|_DK_|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
||_M_|||
|||||



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-2-0.png)

(c) 1 _×_ 1 Convolutional Filters called Pointwise Convolution in the context of Depthwise Separable Convolution


Figure 2. The standard convolutional filters in (a) are replaced by
two layers: depthwise convolution in (b) and pointwise convolution in (c) to build a depthwise separable filter.


**3.2. Network Structure and Training**


The MobileNet structure is built on depthwise separable
convolutions as mentioned in the previous section except for
the first layer which is a full convolution. By defining the
network in such simple terms we are able to easily explore
network topologies to find a good network. The MobileNet
architecture is defined in Table 1. All layers are followed by
a batchnorm [13] and ReLU nonlinearity with the exception
of the final fully connected layer which has no nonlinearity
and feeds into a softmax layer for classification. Figure 3
contrasts a layer with regular convolutions, batchnorm and
ReLU nonlinearity to the factorized layer with depthwise
convolution, 1 _×_ 1 pointwise convolution as well as batchnorm and ReLU after each convolutional layer. Down sampling is handled with strided convolution in the depthwise
convolutions as well as in the first layer. A final average
pooling reduces the spatial resolution to 1 before the fully
connected layer. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.
It is not enough to simply define networks in terms of a
small number of Mult-Adds. It is also important to make
sure these operations can be efficiently implementable. For






















|3x3 Conv|Col2|
|---|---|
|||
|BN|BN|
|||
|ReLU|ReLU|


|3x3 Depthwise Conv|Col2|
|---|---|
|||
|BN|BN|
|||
|ReLU|ReLU|
|||
|1x1 Conv|1x1 Conv|
|||
|BN|BN|
|||
|ReLU|ReLU|



Figure 3. Left: Standard convolutional layer with batchnorm and
ReLU. Right: Depthwise Separable convolutions with Depthwise
and Pointwise layers followed by batchnorm and ReLU.


instance unstructured sparse matrix operations are not typically faster than dense matrix operations until a very high
level of sparsity. Our model structure puts nearly all of the
computation into dense 1 _×_ 1 convolutions. This can be implemented with highly optimized general matrix multiply
(GEMM) functions. Often convolutions are implemented
by a GEMM but require an initial reordering in memory
called im2col in order to map it to a GEMM. For instance,
this approach is used in the popular Caffe package [15].
1 _×_ 1 convolutions do not require this reordering in memory
and can be implemented directly with GEMM which is one
of the most optimized numerical linear algebra algorithms.
MobileNet spends 95% of it’s computation time in 1 _×_ 1
convolutions which also has 75% of the parameters as can
be seen in Table 2. Nearly all of the additional parameters
are in the fully connected layer.
MobileNet models were trained in TensorFlow [1] using RMSprop [33] with asynchronous gradient descent similar to Inception V3 [31]. However, contrary to training
large models we use less regularization and data augmentation techniques because small models have less trouble
with overfitting. When training MobileNets we do not use
side heads or label smoothing and additionally reduce the
amount image of distortions by limiting the size of small
crops that are used in large Inception training [31]. Additionally, we found that it was important to put very little or
no weight decay (l2 regularization) on the depthwise filters
since their are so few parameters in them. For the ImageNet
benchmarks in the next section all models were trained with

same training parameters regardless of the size of the model.


**3.3. Width Multiplier: Thinner Models**


Although the base MobileNet architecture is already
small and low latency, many times a specific use case or
application may require the model to be smaller and faster.
In order to construct these smaller and less computationally
expensive models we introduce a very simple parameter _α_
called width multiplier. The role of the width multiplier _α_ is
to thin a network uniformly at each layer. For a given layer



Table 1. MobileNet Body Architecture

|Type / Stride|Filter Shape|Input Size|
|---|---|---|
|Conv / s2|3_ ×_ 3_ ×_ 3_ ×_ 32|224_ ×_ 224_ ×_ 3|
|Conv dw / s1|3_ ×_ 3_ ×_ 32 dw|112_ ×_ 112_ ×_ 32|
|Conv / s1|1_ ×_ 1_ ×_ 32_ ×_ 64|112_ ×_ 112_ ×_ 32|
|Conv dw / s2|3_ ×_ 3_ ×_ 64 dw|112_ ×_ 112_ ×_ 64|
|Conv / s1|1_ ×_ 1_ ×_ 64_ ×_ 128|56_ ×_ 56_ ×_ 64|
|Conv dw / s1|3_ ×_ 3_ ×_ 128 dw<br>|56_ ×_ 56_ ×_ 128<br>|
|Conv / s1|1_ ×_ 1_ ×_ 128_ ×_ 128|56_ ×_ 56_ ×_ 128|
|Conv dw / s2|3_ ×_ 3_ ×_ 128 dw|56_ ×_ 56_ ×_ 128|
|Conv / s1|1_ ×_ 1_ ×_ 128_ ×_ 256|28_ ×_ 28_ ×_ 128|
|Conv dw / s1|3_ ×_ 3_ ×_ 256 dw|28_ ×_ 28_ ×_ 256|
|Conv / s1|1_ ×_ 1_ ×_ 256_ ×_ 256|28_ ×_ 28_ ×_ 256|
|Conv dw / s2|3_ ×_ 3_ ×_ 256 dw|28_ ×_ 28_ ×_ 256|
|Conv / s1|1_ ×_ 1_ ×_ 256_ ×_ 512|14_ ×_ 14_ ×_ 256|
|5_×_ Conv dw / s1<br>Conv / s1|3_ ×_ 3_ ×_ 512 dw<br>1_ ×_ 1_ ×_ 512_ ×_ 512|14_ ×_ 14_ ×_ 512<br>14_ ×_ 14_ ×_ 512|
|Conv dw / s2|3_ ×_ 3_ ×_ 512 dw|14_ ×_ 14_ ×_ 512|
|Conv / s1|1_ ×_ 1_ ×_ 512_ ×_ 1024|7_ ×_ 7_ ×_ 512|
|Conv dw / s2|3_ ×_ 3_ ×_ 1024 dw|7_ ×_ 7_ ×_ 1024|
|Conv / s1|1_ ×_ 1_ ×_ 1024_ ×_ 1024|7_ ×_ 7_ ×_ 1024|
|Avg Pool / s1|Pool 7_ ×_ 7|7_ ×_ 7_ ×_ 1024|
|FC / s1|1024_ ×_ 1000|1_ ×_ 1_ ×_ 1024|
|Softmax / s1|Classiﬁer|1_ ×_ 1_ ×_ 1000|



Table 2. Resource Per Layer Type

|Type|Mult-Adds|Parameters|
|---|---|---|
|Conv 1_ ×_ 1|94.86%|74.59%|
|Conv DW 3_ ×_ 3|3.06%|1.06%|
|Conv 3_ ×_ 3|1.19%|0.02%|
|Fully Connected|0.18%|24.33%|



and width multiplier _α_, the number of input channels _M_ becomes _αM_ and the number of output channels _N_ becomes
_αN_ .

The computational cost of a depthwise separable convolution with width multiplier _α_ is:


_D_ _K_ _· D_ _K_ _· αM · D_ _F_ _· D_ _F_ + _αM · αN · D_ _F_ _· D_ _F_ (6)


where _α ∈_ (0 _,_ 1] with typical settings of 1, 0.75, 0.5 and
0.25. _α_ = 1 is the baseline MobileNet and _α <_ 1 are

reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters
quadratically by roughly _α_ [2] . Width multiplier can be applied to any model structure to define a new smaller model
with a reasonable accuracy, latency and size trade off. It
is used to define a new reduced structure that needs to be
trained from scratch.


**3.4. Resolution Multiplier: Reduced Representa-**
**tion**


The second hyper-parameter to reduce the computational
cost of a neural network is a resolution multiplier _ρ_ . We ap

Table 3. Resource usage for modifications to standard convolution.
Note that each row is a cumulative effect adding on top of the
previous row. This example is for an internal MobileNet layer
with _D_ _K_ = 3, _M_ = 512, _N_ = 512, _D_ _F_ = 14.


Layer/Modification Million Million

Mult-Adds Parameters

Convolution 462 2.36

Depthwise Separable Conv 52.3 0.27
_α_ = 0 _._ 75 29.6 0.15

_ρ_ = 0 _._ 714 15.1 0.15


ply this to the input image and the internal representation of
every layer is subsequently reduced by the same multiplier.
In practice we implicitly set _ρ_ by setting the input resolution.

We can now express the computational cost for the core
layers of our network as depthwise separable convolutions
with width multiplier _α_ and resolution multiplier _ρ_ :


_D_ _K_ _· D_ _K_ _· αM · ρD_ _F_ _· ρD_ _F_ + _αM · αN · ρD_ _F_ _· ρD_ _F_ (7)


where _ρ ∈_ (0 _,_ 1] which is typically set implicitly so that
the input resolution of the network is 224, 192, 160 or 128.
_ρ_ = 1 is the baseline MobileNet and _ρ <_ 1 are reduced
computation MobileNets. Resolution multiplier has the effect of reducing computational cost by _ρ_ [2] .
As an example we can look at a typical layer in MobileNet and see how depthwise separable convolutions,
width multiplier and resolution multiplier reduce the cost
and parameters. Table 3 shows the computation and number
of parameters for a layer as architecture shrinking methods
are sequentially applied to the layer. The first row shows
the Mult-Adds and parameters for a full convolutional layer
with an input feature map of size 14 _×_ 14 _×_ 512 with a kernel _K_ of size 3 _×_ 3 _×_ 512 _×_ 512. We will look in detail

in the next section at the trade offs between resources and

accuracy.


**4. Experiments**


In this section we first investigate the effects of depthwise convolutions as well as the choice of shrinking by reducing the width of the network rather than the number of
layers. We then show the trade offs of reducing the network based on the two hyper-parameters: width multiplier
and resolution multiplier and compare results to a number
of popular models. We then investigate MobileNets applied
to a number of different applications.


**4.1. Model Choices**


First we show results for MobileNet with depthwise separable convolutions compared to a model built with full convolutions. In Table 4 we see that using depthwise separable convolutions compared to full convolutions only reduces



Table 4. Depthwise Separable vs Full Convolution MobileNet


Model ImageNet Million Million

Accuracy Mult-Adds Parameters
Conv MobileNet 71.7% 4866 29.3

MobileNet 70.6% 569 4.2


Table 5. Narrow vs Shallow MobileNet

Model ImageNet Million Million

Accuracy Mult-Adds Parameters
0.75 MobileNet 68.4% 325 2.6

Shallow MobileNet 65.3% 307 2.9


Table 6. MobileNet Width Multiplier

Width Multiplier ImageNet Million Million

Accuracy Mult-Adds Parameters
1.0 MobileNet-224 70.6% 569 4.2

0.75 MobileNet-224 68.4% 325 2.6

0.5 MobileNet-224 63.7% 149 1.3

0.25 MobileNet-224 50.6% 41 0.5


Table 7. MobileNet Resolution

Resolution ImageNet Million Million

Accuracy Mult-Adds Parameters
1.0 MobileNet-224 70.6% 569 4.2

1.0 MobileNet-192 69.1% 418 4.2

1.0 MobileNet-160 67.2% 290 4.2

1.0 MobileNet-128 64.4% 186 4.2


accuracy by 1% on ImageNet was saving tremendously on
mult-adds and parameters.
We next show results comparing thinner models with
width multiplier to shallower models using less layers. To
make MobileNet shallower, the 5 layers of separable filters
with feature size 14 _×_ 14 _×_ 512 in Table 1 are removed.

Table 5 shows that at similar computation and number of
parameters, that making MobileNets thinner is 3% better
than making them shallower.


**4.2. Model Shrinking Hyperparameters**


Table 6 shows the accuracy, computation and size trade
offs of shrinking the MobileNet architecture with the width
multiplier _α_ . Accuracy drops off smoothly until the architecture is made too small at _α_ = 0 _._ 25.

Table 7 shows the accuracy, computation and size trade
offs for different resolution multipliers by training MobileNets with reduced input resolutions. Accuracy drops
off smoothly across resolution.
Figure 4 shows the trade off between ImageNet Accuracy and computation for the 16 models made from the
cross product of width multiplier _α ∈{_ 1 _,_ 0 _._ 75 _,_ 0 _._ 5 _,_ 0 _._ 25 _}_
and resolutions _{_ 224 _,_ 192 _,_ 160 _,_ 128 _}_ . Results are log linear
with a jump when models get very small at _α_ = 0 _._ 25.


![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-5-0.png)

Figure 4. This figure shows the trade off between computation
(Mult-Adds) and accuracy on the ImageNet benchmark. Note the
log linear dependence between accuracy and computation.

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-5-1.png)


Figure 5. This figure shows the trade off between the number of
parameters and accuracy on the ImageNet benchmark. The colors
encode input resolutions. The number of parameters do not vary
based on the input resolution.


Figure 5 shows the trade off between ImageNet Accuracy and number of parameters for the 16 models
made from the cross product of width multiplier _α ∈_
_{_ 1 _,_ 0 _._ 75 _,_ 0 _._ 5 _,_ 0 _._ 25 _}_ and resolutions _{_ 224 _,_ 192 _,_ 160 _,_ 128 _}_ .


Table 8 compares full MobileNet to the original
GoogleNet [30] and VGG16 [27]. MobileNet is nearly
as accurate as VGG16 while being 32 times smaller and
27 times less compute intensive. It is more accurate than
GoogleNet while being smaller and more than 2.5 times less
computation.


Table 9 compares a reduced MobileNet with width multiplier _α_ = 0 _._ 5 and reduced resolution 160 _×_ 160. Reduced
MobileNet is 4% better than AlexNet [19] while being 45 _×_
smaller and 9 _._ 4 _×_ less compute than AlexNet. It is also 4%
better than Squeezenet [12] at about the same size and 22 _×_
less computation.



Table 8. MobileNet Comparison to Popular Models


Model ImageNet Million Million

Accuracy Mult-Adds Parameters
1.0 MobileNet-224 70.6% 569 4.2

GoogleNet 69.8% 1550 6.8
VGG 16 71.5% 15300 138


Table 9. Smaller MobileNet Comparison to Popular Models


Model ImageNet Million Million

Accuracy Mult-Adds Parameters
0.50 MobileNet-160 60.2% 76 1.32

Squeezenet 57.5% 1700 1.25
AlexNet 57.2% 720 60


Table 10. MobileNet for Stanford Dogs

Model Top-1 Million Million

Accuracy Mult-Adds Parameters
Inception V3 [18] 84% 5000 23.2
1.0 MobileNet-224 83.3% 569 3.3

0.75 MobileNet-224 81.9% 325 1.9

1.0 MobileNet-192 81.9% 418 3.3

0.75 MobileNet-192 80.5% 239 1.9


Table 11. Performance of PlaNet using the MobileNet architecture. Percentages are the fraction of the Im2GPS test dataset that
were localized within a certain distance from the ground truth. The
numbers for the original PlaNet model are based on an updated
version that has an improved architecture and training dataset.


Scale Im2GPS [7] PlaNet [35] PlaNet

MobileNet

Continent (2500 km) 51.9% 77.6% 79.3%
Country (750 km) 35.4% 64.0% 60.3%
Region (200 km) 32.1% 51.1% 45.2%
City (25 km) 21.9% 31.7% 31.7%
Street (1 km) 2.5% 11.0% 11.4%


**4.3. Fine Grained Recognition**


We train MobileNet for fine grained recognition on the
Stanford Dogs dataset [17]. We extend the approach of [18]
and collect an even larger but noisy training set than [18]
from the web. We use the noisy web data to pretrain a fine
grained dog recognition model and then fine tune the model
on the Stanford Dogs training set. Results on Stanford Dogs
test set are in Table 10. MobileNet can almost achieve the

state of the art results from [18] at greatly reduced computation and size.


**4.4. Large Scale Geolocalizaton**


PlaNet [35] casts the task of determining where on earth
a photo was taken as a classification problem. The approach
divides the earth into a grid of geographic cells that serve as
the target classes and trains a convolutional neural network


on millions of geo-tagged photos. PlaNet has been shown
to successfully localize a large variety of photos and to outperform Im2GPS [6, 7] that addresses the same task.
We re-train PlaNet using the MobileNet architecture on
the same data. While the full PlaNet model based on the In
ception V3 architecture [31] has 52 million parameters and
5.74 billion mult-adds. The MobileNet model has only 13
million parameters with the usual 3 million for the body and
10 million for the final layer and 0.58 Million mult-adds.
As shown in Tab. 11, the MobileNet version delivers only
slightly decreased performance compared to PlaNet despite
being much more compact. Moreover, it still outperforms
Im2GPS by a large margin.


**4.5. Face Attributes**


Another use-case for MobileNet is compressing large
systems with unknown or esoteric training procedures. In
a face attribute classification task, we demonstrate a synergistic relationship between MobileNet and distillation [9],
a knowledge transfer technique for deep networks. We
seek to reduce a large face attribute classifier with 75
million parameters and 1600 million Mult-Adds. The
classifier is trained on a multi-attribute dataset similar to
YFCC100M [32].
We distill a face attribute classifier using the MobileNet
architecture. Distillation [9] works by training the classifier to emulate the outputs of a larger model [2] instead of the
ground-truth labels, hence enabling training from large (and
potentially infinite) unlabeled datasets. Marrying the scalability of distillation training and the parsimonious parameterization of MobileNet, the end system not only requires
no regularization (e.g. weight-decay and early-stopping),
but also demonstrates enhanced performances. It is evident from Tab. 12 that the MobileNet-based classifier is resilient to aggressive model shrinking: it achieves a similar
mean average precision across attributes (mean AP) as the
in-house while consuming only 1% the Multi-Adds.


**4.6. Object Detection**


MobileNet can also be deployed as an effective base network in modern object detection systems. We report results
for MobileNet trained for object detection on COCO data
based on the recent work that won the 2016 COCO chal
lenge [10]. In table 13, MobileNet is compared to VGG
and Inception V2 [13] under both Faster-RCNN [23] and
SSD [21] framework. In our experiments, SSD is evaluated
with 300 input resolution (SSD 300) and Faster-RCNN is
compared with both 300 and 600 input resolution (FasterRCNN 300, Faster-RCNN 600). The Faster-RCNN model
evaluates 300 RPN proposal boxes per image. The models
are trained on COCO train+val excluding 8k minival images


2 The emulation quality is measured by averaging the per-attribute
cross-entropy over all attributes.



Table 12. Face attribute classification using the MobileNet architecture. Each row corresponds to a different hyper-parameter setting (width multiplier _α_ and image resolution).


Width Multiplier / Mean Million Million
Resolution AP Mult-Adds Parameters

1.0 MobileNet-224 88.7% 568 3.2

0.5 MobileNet-224 88.1% 149 0.8

0.25 MobileNet-224 87.2% 45 0.2

1.0 MobileNet-128 88.1% 185 3.2

0.5 MobileNet-128 87.7% 48 0.8

0.25 MobileNet-128 86.4% 15 0.2

Baseline 86.9% 1600 7.5


Table 13. COCO object detection results comparison using different frameworks and network architectures. mAP is reported with
COCO primary challenge metric (AP at IoU=0.50:0.05:0.95)


Framework Model mAP Billion Million

Resolution Mult-Adds Parameters

deeplab-VGG 21.1% 34.9 33.1
SSD 300 Inception V2 22.0% 3.8 13.7
MobileNet 19.3% 1.2 6.8

Faster-RCNN VGG 22.9% 64.3 138.5

300 Inception V2 15.4% 118.2 13.3
MobileNet 16.4% 25.2 6.1

Faster-RCNN VGG 25.7% 149.6 138.5

600 Inception V2 21.9% 129.6 13.3
Mobilenet 19.8% 30.5 6.1

![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/literature/markdown/CONV1D/CONV1D.pdf-6-0.png)


Figure 6. Example objection detection results using MobileNet
SSD.


and evaluated on minival. For both frameworks, MobileNet
achieves comparable results to other networks with only a
fraction of computational complexity and model size.


**4.7. Face Embeddings**


The FaceNet model is a state of the art face recognition
model [25]. It builds face embeddings based on the triplet
loss. To build a mobile FaceNet model we use distillation

to train by minimizing the squared differences of the output


Table 14. MobileNet Distilled from FaceNet

Model 1e-4 Million Million

Accuracy Mult-Adds Parameters
FaceNet [25] 83% 1600 7.5

1.0 MobileNet-160 79.4% 286 4.9

1.0 MobileNet-128 78.3% 185 5.5

0.75 MobileNet-128 75.2% 166 3.4

0.75 MobileNet-128 72.5% 108 3.8


of FaceNet and MobileNet on the training data. Results for
very small MobileNet models can be found in table 14.


**5. Conclusion**


We proposed a new model architecture called MobileNets based on depthwise separable convolutions. We
investigated some of the important design decisions leading
to an efficient model. We then demonstrated how to build
smaller and faster MobileNets using width multiplier and
resolution multiplier by trading off a reasonable amount of
accuracy to reduce size and latency. We then compared different MobileNets to popular models demonstrating superior size, speed and accuracy characteristics. We concluded
by demonstrating MobileNet’s effectiveness when applied
to a wide variety of tasks. As a next step to help adoption
and exploration of MobileNets, we plan on releasing models in Tensor Flow.


**References**


[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen,
C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, et al.
Tensorflow: Large-scale machine learning on heterogeneous
systems, 2015. _Software available from tensorflow. org_, 1,
2015. 4

[2] W. Chen, J. T. Wilson, S. Tyree, K. Q. Weinberger, and
Y. Chen. Compressing neural networks with the hashing
trick. _CoRR, abs/1504.04788_, 2015. 2

[3] F. Chollet. Xception: Deep learning with depthwise separable convolutions. _arXiv preprint arXiv:1610.02357v2_, 2016.

1

[4] M. Courbariaux, J.-P. David, and Y. Bengio. Training deep
neural networks with low precision multiplications. _arXiv_
_preprint arXiv:1412.7024_, 2014. 2

[5] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. _CoRR, abs/1510.00149_, 2, 2015.
2

[6] J. Hays and A. Efros. IM2GPS: estimating geographic information from a single image. In _Proceedings of the IEEE_
_International Conference on Computer Vision and Pattern_
_Recognition_, 2008. 7

[7] J. Hays and A. Efros. Large-Scale Image Geolocalization.
In J. Choi and G. Friedland, editors, _Multimodal Location_
_Estimation of Videos and Images_ . Springer, 2014. 6, 7




[8] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. _arXiv preprint arXiv:1512.03385_,
2015. 1

[9] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge
in a neural network. _arXiv preprint arXiv:1503.02531_, 2015.
2, 7

[10] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara,
A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al.
Speed/accuracy trade-offs for modern convolutional object
detectors. _arXiv preprint arXiv:1611.10012_, 2016. 7

[11] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and
Y. Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. _arXiv_
_preprint arXiv:1609.07061_, 2016. 2

[12] F. N. Iandola, M. W. Moskewicz, K. Ashraf, S. Han, W. J.
Dally, and K. Keutzer. Squeezenet: Alexnet-level accuracy
with 50x fewer parameters and¡ 1mb model size. _arXiv_
_preprint arXiv:1602.07360_, 2016. 1, 6

[13] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
_arXiv preprint arXiv:1502.03167_, 2015. 1, 3, 7

[14] M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up
convolutional neural networks with low rank expansions.
_arXiv preprint arXiv:1405.3866_, 2014. 2

[15] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. _arXiv preprint_
_arXiv:1408.5093_, 2014. 4

[16] J. Jin, A. Dundar, and E. Culurciello. Flattened convolutional
neural networks for feedforward acceleration. _arXiv preprint_
_arXiv:1412.5474_, 2014. 1, 3

[17] A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei.
Novel dataset for fine-grained image categorization. In _First_
_Workshop on Fine-Grained Visual Categorization, IEEE_
_Conference on Computer Vision and Pattern Recognition_,
Colorado Springs, CO, June 2011. 6

[18] J. Krause, B. Sapp, A. Howard, H. Zhou, A. Toshev,
T. Duerig, J. Philbin, and L. Fei-Fei. The unreasonable effectiveness of noisy data for fine-grained recognition. _arXiv_
_preprint arXiv:1511.06789_, 2015. 6

[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet
classification with deep convolutional neural networks. In
_Advances in neural information processing systems_, pages
1097–1105, 2012. 1, 6

[20] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and
V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. _arXiv preprint_
_arXiv:1412.6553_, 2014. 2

[21] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed.
Ssd: Single shot multibox detector. _arXiv preprint_
_arXiv:1512.02325_, 2015. 7

[22] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi. Xnornet: Imagenet classification using binary convolutional neural networks. _arXiv preprint arXiv:1603.05279_, 2016. 1, 2

[23] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards
real-time object detection with region proposal networks. In
_Advances in neural information processing systems_, pages
91–99, 2015. 7


[24] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
et al. Imagenet large scale visual recognition challenge.
_International Journal of Computer Vision_, 115(3):211–252,
2015. 1

[25] F. Schroff, D. Kalenichenko, and J. Philbin. Facenet: A unified embedding for face recognition and clustering. In _Pro-_
_ceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition_, pages 815–823, 2015. 8

[26] L. Sifre. _Rigid-motion scattering for image classification_ .
PhD thesis, Ph. D. thesis, 2014. 1, 3

[27] K. Simonyan and A. Zisserman. Very deep convolutional
networks for large-scale image recognition. _arXiv preprint_
_arXiv:1409.1556_, 2014. 1, 6

[28] V. Sindhwani, T. Sainath, and S. Kumar. Structured transforms for small-footprint deep learning. In _Advances in_
_Neural Information Processing Systems_, pages 3088–3096,
2015. 1

[29] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4,
inception-resnet and the impact of residual connections on
learning. _arXiv preprint arXiv:1602.07261_, 2016. 1

[30] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In _Proceedings of the IEEE_
_Conference on Computer Vision and Pattern Recognition_,
pages 1–9, 2015. 6

[31] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.
Rethinking the inception architecture for computer vision.
_arXiv preprint arXiv:1512.00567_, 2015. 1, 3, 4, 7

[32] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni,
D. Poland, D. Borth, and L.-J. Li. Yfcc100m: The new
data in multimedia research. _Communications of the ACM_,
59(2):64–73, 2016. 7

[33] T. Tieleman and G. Hinton. Lecture 6.5-rmsprop: Divide
the gradient by a running average of its recent magnitude.
_COURSERA: Neural Networks for Machine Learning_, 4(2),
2012. 4

[34] M. Wang, B. Liu, and H. Foroosh. Factorized convolutional
neural networks. _arXiv preprint arXiv:1608.04337_, 2016. 1

[35] T. Weyand, I. Kostrikov, and J. Philbin. PlaNet - Photo Geolocation with Convolutional Neural Networks. In _European_
_Conference on Computer Vision (ECCV)_, 2016. 6, 7

[36] J. Wu, C. Leng, Y. Wang, Q. Hu, and J. Cheng. Quantized
convolutional neural networks for mobile devices. _arXiv_

_preprint arXiv:1512.06473_, 2015. 1

[37] Z. Yang, M. Moczulski, M. Denil, N. de Freitas, A. Smola,
L. Song, and Z. Wang. Deep fried convnets. In _Proceedings_
_of the IEEE International Conference on Computer Vision_,
pages 1476–1483, 2015. 1


