Published as a conference paper at ICLR 2025

## G ATED D ELTA N ETWORKS : I MPROVING M AMBA 2 WITH D ELTA R ULE

Songlin Yang [∗]
MIT CSAIL
yangsl66@mit.edu

Jan Kautz
NVIDIA

jkautz@nvidia.com

Ali Hatamizadeh [⋆]

NVIDIA

ahatamizadeh@nvidia.com

A BSTRACT

Linear Transformers have gained attention as efficient alternatives to standard
Transformers, but their performance in retrieval and long-context tasks has been
limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted
updates. Building on this insight, we introduce the gated delta rule and develop a
parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and
DeltaNet across multiple benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures
that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.
[Code: https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

1 I NTRODUCTION

The Transformer architecture has significantly advanced the capabilities of Large Language Models
(LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism. This mechanism excels in precise sequence modeling and leverages the parallel
processing capabilities of modern GPUs during training. However, the self-attention component
scales quadratically with sequence length, leading to substantial computational demands that pose
challenges for both training and inference.

To mitigate these issues, researchers have explored alternatives such as linear Transformers
(Katharopoulos et al., 2020a), which replace traditional softmax-based attention with kernelized
dot-product-based linear attention, substantially reducing memory requirements during inference
by reframing as a linear RNN with matrix-valued states. While early versions of linear Transformers underperformed in language modeling tasks compared to standard Transformers, recent
enhancements—such as incorporating data-dependent gating mechanisms akin to those in LSTMs,
exemplified by models like GLA (Yang et al., 2024a) and Mamba2 (Dao & Gu, 2024a)—have
shown promising improvements. However, challenges persist in managing information over long
sequences, particularly for in-context retrieval tasks where traditional Transformers maintain their
advantage (Arora et al., 2023a; 2024a; Jelassi et al., 2024; Wen et al., 2024; Akyürek et al., 2024).

This phenomenon is not surprising: linear Transformers can be interpreted as implementing an
outer-product-based key-value association memory, reminiscent of tensor product representation
(Smolensky, 1990). However, the number of orthogonal key-value pairs they can store is bounded by
the model’s dimensionality. When the sequence length exceeds this dimension, “memory collisions“
become inevitable, hindering exact retrieval (Schlag et al., 2021a).

Mamba2 addresses this limitation by introducing a simple gated update rule, S t = α t S t−1 + v t k t [⊺] [,]
which uniformly decays all key-value associations at each time step by a dynamic ratio, α t ∈ (0, 1).

∗ Equation contribution. Work done during SY’s internship at NVIDIA.

1

Published as a conference paper at ICLR 2025

However, this approach does not account for the varying importance of different key-value associations, potentially leading to inefficient memory utilization. If the model needs to forget a specific
key-value association, all key-value associations are equally forgotten, making the process less targeted and efficient.

In contrast, the linear Transformer with the delta rule (Widrow et al., 1960), known as DeltaNet
(Schlag et al., 2021a; Yang et al., 2024b), selectively updates memory by (softly) replacing an old
key-value pair with the incoming one in a sequential manner. This method has demonstrated impressive performance in synthetic benchmarks for in-context retrieval. However, since this process
only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated
or irrelevant information, especially during context switches where previous data needs to be erased.
Consequently, DeltaNet has been found to perform moderately on real-world tasks (Yang et al.,
2024b), likely due to the absence of a robust memory-clearing mechanism.

Recognizing the complementary advantages of the gated update rule and the delta rule in memory
management, we propose the gated delta rule, a simple and intuitive mechanism that combines both
approaches. This unified rule enables flexible memory control: it can promptly clear memory by
setting α t → 0, while selectively updating specific content without affecting other information by
setting α t → 1 (effectively switching to the pure delta rule).

The remaining challenge lies in implementing the gated delta rule in a hardware-efficient manner.
Building upon Yang et al. (2024b)’s efficient algorithm that parallelizes the delta rule computation
using the WY representation (Bischof & Loan, 1985), we carefully extend their approach to incorporate the gating terms. Our extension preserves the benefits of chunkwise parallelism (Hua et al.,
2022b; Sun et al., 2023a; Yang et al., 2024a;b), enabling hardware-efficient training.

Our resulting architecture, Gated DeltaNet, consistently outperforms both Mamba2 and DeltaNet
across a comprehensive suite of benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. Building on these
results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with
sliding window attention or Mamba2 layers, further enhancing both training efficiency and model
performance.

2 P RELIMINARY

2.1 M AMBA 2: L INEAR A TTENTION WITH DECAY

It is known that the linear transformer (Katharopoulos et al., 2020b) can be formulated as the following linear recurrence when excluding normalization and query/key activations:

S t = S t−1 + v t k t [⊺] [∈] [R] [d] [v] [×][d] [k] [,] o t = S t q t ∈ R [d] [v]

where d k and d v represent the (head) dimensions for query/key and value, respectively. By expanding the recurrence, we can express it in both vector form (left) and matrix form (right) as follows:

t
� v i (k i [⊺] [q] [t] [)][ ∈] [R] [d] [v] [,] O = (QK [⊺] ⊙ M)V ∈ R [L][×][d] [v]

i=1

o t =

t
�(v i k i [⊺] [)][q] [t] [ =]

i=1

t
�

where L is the sequence length, and M ∈ R [L][×][L] is the causal mask defined by M ij = 0 when i < j,
and 1 otherwise.

However, this vanilla linear attention underperforms Transformers in language modeling by a large
margin. To address this, it is common to add a decay term to forget historical information. Here we
take Mamba2 (Dao & Gu, 2024a) as an example, which can be represented by the following linear
recurrence (up to specific parameterization):
S t = α t S t−1 + v t k t [⊺] [,] o t = S t q t
where α t ∈ (0, 1) is a data-dependent scalar-valued decay term that varies with t. Define the cumulative decay product γ j = [�] [j] i=1 [α] [i] [, and by expanding the recurrence, we can express the result in]
both a vector form (left) and a matrix parallel form (right):

γ t
� γ i

2

v i k i [⊺] q t =

�

t

v i

�

i=1

, O = ((QK [⊺] ) ⊙ Γ) V
�

o t =

t
�

i=1

γ t
� γ i

k i [⊺] [q] [t]

Published as a conference paper at ICLR 2025

Here, Γ ∈ R [L][×][L] is a decay-aware causal mask where Γ ij = γ [γ] j [i] [if][ i][ ≥] [j][ and][ Γ] [ij] [ = 0][ otherwise.]

The equivalence between these parallel and recurrent forms is also referred to as the state space
duality (SSD) described in Dao & Gu (2024a). This recurrence structure appears in several other
architectures including Gated RFA (Peng et al., 2021), xLSTM (Beck et al., 2024), and Gated RetNet
(Sun et al., 2024b). When γ t is data-independent, the formulation reduces to RetNet (Sun et al.,
2023a) and Lightning-Attention (Qin et al., 2024a). Furthermore, if γ t is extended to be matrixvalued rather than scalar-valued, efficient training algorithms remain possible when parameterized
with an outer-product structure, as demonstrated by Yang et al. (2024a) and used by Yang et al.
(2024a); Peng et al. (2024); Qin et al. (2024b); Zhang et al. (2024); Chou et al. (2024); He et al.
(2025); Lu et al. (2025).

Chunkwise training However, both the recurrent and parallel forms are not ideal for efficient
training (Hua et al., 2022b; Yang et al., 2024a), which motivates the use of the chunkwise parallel
form (Hua et al., 2022b; Sun et al., 2023a) for hardware-efficient, linear-time training, as introduced
below. To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of
size C, and computes outputs for each chunk based on the final state of the previous chunk and the
query/key/value blocks of the current chunk. Following the notation of Sun et al. (2023b); Yang
et al. (2024a;b), we take the query block, q, as an example. We denote Q [t] := q tC+1:(t+1)C+1 as
the query block for chunk t, and q [ [r] t] [:=][ q] [tC][+][r] [ as the][ r][-th query within chunk][ t][. The initial state of]
chunk t is defined as S [t] := S [0] [t] [=][ S] [C] [t−1] [. By partially expanding the recurrence, we have]

r
�

� v [ [i] t] �k [ [i] t [⊺] ] [q] [ [r] t] � ∈ R [d] [v]

i=1

S [r] [t] [=][ S] [[][t][]] [+]

r
�

� v [ [i] t] [k] [ [i] t [⊺] ] [∈] [R] [d] [v] [×][d] [k] [,] o [r] [t] [=][ S] [r] [t] [q] [ [r] t] [=][ S] [[][t][]] [q] [ [r] t] [+]

i=1

Equivalently, in matrix form:
S [t+1] = S [t] + V [t] K [⊺] [t] [∈] [R] [d] [v] [×][d] [k] [,] O [t] = Q [t] S [⊺] [t] [+] �Q [t] K [⊺] [t] [⊙] [M] � V [t] ∈ R [C][×][d] [v]

where M ∈ R [C][×][C] is the causal mask. The above equations are rich in matrix multiplications
(matmuls), allowing for tensor-core-based hardware optimization. This chunkwise algorithm could
be easily extended to linear attention with decay:

S [t+1] = [−→] S [t] + V [ [⊺] t]

−−→
K [t] ∈ R [d] [v] [×][d] [k], O [t] = [←−−] Q [t] S [⊺] [t] [+] �Q [t] K [⊺] [t] [⊙] [Γ] [[][t][]]

� V [t] ∈ R [C][×][d] [v] (1)

where (Γ [t] ) ij = γγ [[ [j][i] tt]] [, γ] [ [j] t] [=][ �] j [tC] = [+] tC [j] +1 [α] [j] [.] [ 1] [ Here we use the left arrow (][←−·][ ) or the right arrow (][−→·][ )]

to denote a variable decaying to the first position and the last position of each chunk, respectively,

←−
q [ [r] t] [=][ γ] [ [r] t] [q] [ [r] t] decaying each vector to the first position of chunk t

−→
k [ [r] t] [=]

γ [ [C] t]
γ [ [r] t]

k [ [r] t] decaying each vector to the last position of chunk t

−→
S [t] = γ [ [C] t] [S] [[][t][]] decaying the state matrix over the entire chunk t (2)

and likewise for other variables (e.g., [−→] v ). The SSD decomposition algorithm introduced in Mamba2
is largely equivalent to this chunkwise algorithm. For a more generalized approach, Yang et al.
(2024a) proposed an extended chunkwise algorithm for linear attention that incorporates finegrained decay mechanisms.

2.2 D ELTA N ETWORKS : L INEAR A TTENTION WITH D ELTA R ULE

The delta update rule (Widrow et al., 1960; Schlag et al., 2021b) dynamically erases the value (v t [old] [)]
associated with the current input key (k t ) and writes a new value (v t [new] ), which is a linear combination of the current input value and the old value based on the “writing strength” β t ∈ (0, 1). [2]

k t [⊺] [=][ S] [t][−][1] [(][I][ −] [β] [t] [k] [t] [k] t [⊺] [) +][ β] [t] [v] [t] [k] t [⊺]

S t = S t−1 − (S t−1 k t )
� �� �
v t [old]

k t [⊺] [+ (][β] [t] [v] [t] [+ (1][ −] [β] [t] [)][S] [t][−][1] [k] [t] [))]
� �� �
v t [new]

1
Here we slightly abuse the notation of γ to denote the cumulative product for each chunk (starting with the
first position of each chunk separately) instead of the entire sequence.
2 It is possible to set β t ∈ (0, 2) to allow negative eigenvalue to unlock the state tracking abilities of DeltaNet
(Grazzi et al., 2024; Siems et al., 2025).

3

Published as a conference paper at ICLR 2025

As shown above, DeltaNet implements a first-order linear recurrence with generalized Householder
transition matrices (I − β t k t k t [⊺] [)][. Despite demonstrating superior associative recall and language]
modeling performance (Schlag et al., 2021a), DeltaNet received limited attention due to computational inefficiency until Yang et al. (2024b) introduced a hardware-efficient chunkwise training
algorithm, as detailed below.

Chunkwise parallel form. By partially expanding the recurrence, we have

�I − β [ [j] t] [k] [ [j] t] [k] [ [j] t [⊺] ] �



�I − β [ [j] t] [k] [ [j] t] [k] [ [j] t [⊺] ] �

�



β [ [i] t] [v] [ [i] t] [k] [ [i] t [⊺] ]

r
�

j=i+1

S [r] [t] [=][ S] [[][t][]]

r
� I − β [ [i] t] [k] [ [i] t] [k] [ [i] t [⊺] ]
� i=1

+

r
�

i=1

(3)

� �� �
:=P [r] [t]

� �� �
:=H [r] [t]

where P [j] [t] [involves cumulative products of generalized Householder matrices, which could be opti-]
mized by the classical WY representation (Bischof & Loan, 1985):

�

r−1
� �w [ [i] t] [(][k] [ [i] t [⊺] ] [k] [ [r] t] [)] � [�]

i=1

∈ R [d] [k] (4)

P [r] [t] [=][ I][ −]

r
� w [ [i] t] [k] [ [i] t [⊺] ] [∈] [R] [d] [k] [×][d] [k] w [ [r] t] [=][ β] [ [r] t]

i=1

k [ [r] t] [−]

Likewise, H [r] [t] [could be represented as:]

�u [i] [t] [(][k] [ [i] t [⊺] ] [k] [ [r] t] [)] � [�]

�

r−1
�

i=1

∈ R [d] [v] (5)

H [r] [t] [=]

r
� u [i] [t] [k] [ [i] t [⊺] ] [∈] [R] [d] [v] [×][d] [k] u [r] [t] [=][ β] [ [r] t]

i=1

v [ [r] t] [−]

and in matrix form: P [t] = I − W [ [⊤] t] [K] [[][t][]] [ ∈] [R] [d] [k] [×][d] [k] [,][ H] [[][t][]] [ =][ U] [ [⊤] t] [K] [[][t][]] [ ∈] [R] [d] [v] [×][d] [k] [. By using the UT]
transform (Joffrain et al., 2006), we can further write W and U in matrix form:

−1
T [t] = �I + strictLower �diag(β [t] )K [t] K [⊺] [t] �� diag �β [t] � ∈ R [C][×][C] (6)

W [t] = T [t] K [t] ∈ R [C][×][d] [k], U [t] = T [t] V [t] ∈ R [C][×][d] [v] (7)

Substituting these back into Eq. 3 yields a hardware-efficient chunkwise algorithm for DeltaNet that
leverages matmuls, enabling tensor core based GPU optimization:

⊺
S [t+1] = S [t] P [t] + H [t] = S [t] + �U [t] − W [t] S [⊺] [t] � K [t] ∈ R [d] [v] [×][d] [k] (8)

O [t] = Q [t] S [⊺] [t] [+ (][Q] [[][t][]] [K] [⊺] [t] [⊙] [M][)] �U [t] − W [t] S [⊺] [t] � ∈ R [C][×][d] [v] (9)

3 G ATED D ELTA N ETWORKS

3.1 F ORMULATION : G ATED D ELTA R ULE

The proposed gated delta rule is simple yet effective:

S t = S t−1 (α t (I − β t k t k t [⊺] [)) +][ β] [t] [v] [t] [k] t [⊺] (10)

where the data-dependent gating term α t ∈ (0, 1) controls state decay. This formulation unifies the
advantages of both gating mechanisms and the delta rule: the gating term enables adaptive memory
management, while the delta update structure facilitates effective key-value association learning.

We present a formal analysis of the gated delta rule through the lens of the online learning framework
introduced by Liu et al. (2024). In this framework, recurrent state updates emerge as closed-form
solutions to an online learning problem, as shown in Table 1. Recent linear RNN architectures typically incorporate a regularization term in their online learning objective to prevent state divergence
from previous values, thereby enabling memory retention. However, this retention mechanism becomes problematic when the state becomes saturated with information. In such cases, each state
would encode a superposition of multiple information pieces, making precise retrieval challenging.
To address this limitation, Mamba2 and Gated DeltaNet introduce an adaptive scaling factor α t that
relaxes the regularization term, allowing controlled deviations between S t and S t−1 . This modification enables dynamic memory management through selective forgetting, which could be useful in
filtering out irrelevant information (see §3.2).

4

Published as a conference paper at ICLR 2025

Table 1: Comparison of different linear RNN models and their corresponding online learning objectives using
the framework from Liu et al. (2024). For convenience, we simplify Longhorn’s vector-valued β to scalar β.

Method Online Learning Objective Online Update

LA ∥S t − S t−1 ∥ [2] F [−] [2][⟨][S] [t] [k] [t] [,][ v] [t] [⟩] S t = S t−1 + v t k [T] t
Mamba2 ∥S t − α t S t−1 ∥ [2] F [−] [2][⟨][S] [t] [k] [t] [,][ v] [t] [⟩] S t = α t S t−1 + v t k [T] t

β t
Longhorn ∥S t − S t−1 ∥ [2] F [−] [β] [t] [∥][S] [t] [k] [t] [ −] [v] [t] [∥] [2] S t = S t−1 (I − ǫk t k [T] t [) +][ ǫ] [t] [v] [t] [k] [T] t [, ǫ] [t] [ =] 1 + β t k [⊤] t [k] [t]
DeltaNet ∥S t − S t−1 ∥ [2] F [−] [2][⟨][S] [t] [k] [t] [, β] [t] [ (][v] [t] [ −] [S] [t][−][1] [k] [t] [)][⟩] S t = S t−1 (I − β t k t k [T] t [) +][ β] [t] [v] [t] [k] [T] t
Gated DeltaNet ∥S t − α t S t−1 ∥ [2] F [−] [2][⟨][S] [t] [k] [t] [, β] [t] [ (][v] [t] [ −] [α] [t] [S] [t][−][1] [k] [t] [)][⟩] S t = S t−1 �α t (I − β t k t k [T] t [)] � + β t v t k [T] t

Table 2: Zero-shot performance comparison on S-NIAH benchmark suite for 1.3B models (see §4 for setups)

S-NIAH-1 S-NIAH-2 S-NIAH-3
(pass-key retrieval) (number in haystack) (uuid in haystack)

Model 1K 2K 4K 8K 1K 2K 4K 8K 1K 2K 4K

DeltaNet 97.4 96.8 99.0 98.8 98.4 45.6 18.6 14.4 85.2 47.0 22.4

Mamba2 99.2 98.8 65.4 30.4 99.4 98.8 56.2 17.0 64.4 47.6 4.6

Gated DeltaNet 98.4 88.4 91.4 91.8 100.0 99.8 92.2 29.6 86.6 84.2 27.6

On the other hand, Linear Attention (LA) and Mamba2 use a simple negative inner-product loss
-⟨S t k t, v t ⟩, while Longhorn (Liu et al., 2024) uses a more expressive online regression objective
∥S t k t − v t ∥ [2] for better modeling of key-value associations. The resulting Longhorn’s update rule
closely resembles the delta update rule, [3] suggesting the superiority of the (gated) delta rule over
Mamba2 in in-context associative recall.

From the perspective of fast weight programming (Irie et al., 2022a) and test-time training (Sun et al.,
2024a) and regression (Wang et al., 2025), the hidden state S can be interpreted as a (fast) weight
matrix, with the delta rule optimizing the online regression objective L(S t ) = [1] 2 [∥][S] [t] [k] [t] [ −] [v] [t] [∥] [2] [ via]

test-time stochastic gradient descent (SGD):

S t+1 = S t − β t ∇L(S t ) = S t − β t (S t k t − v t )k t [⊺] [=][ S] [t] [ (][I][ −] [β] [t] [k] [t] [k] t [⊺] [) +][ β] [t] [v] [t] [k] t [⊺]

where β t represents the (adaptive) learning rate. From this perspective, the gated delta rule can be
viewed as incorporating an adaptive weight decay term α t into the SGD update, a technique widely
used in deep learning (Krogh & Hertz, 1991; Andriushchenko et al., 2023). Concurrently, Titans
(Behrouz et al., 2024) demonstrated the effectiveness of incorporating weight decay mechanisms in
RNN test-time SGD updates.

3.2 C ASE STUDY : S INGLE N EEDLE IN A H AYSTACK (S-NIAH)

To better understand the complementary strength between the delta rule and the gated rule, we
present a case study on the Single Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER
(Hsieh et al., 2024), where a key-value pair acts as a needle in the haystack (context) and the model
must recall the value when given the key. Table 2 presents the results and we draw three main
observations:

Decay hurts memory retention. In the simplest S-NIAH-1 setting with repeated synthetic context, models memorize minimal information, testing long-term retention. DeltaNet achieves nearperfect performance across all sequence lengths. Mamba2 degrades significantly beyond 2K sequences since it decays historical information too quickly, while Gated DeltaNet’s degradation is
less severe thanks to the use of delta rule.

Gating facilitates filtering. In S-NIAH-2/3 with real-world-essay context, models store all potentially relevant information, testing efficient memory management. With fixed state size, lack
of clearance causes memory collision—information becomes superimposed and indistinguishable.
DeltaNet’s performance drops significantly at longer sequences due to poor memory clearance.
Mamba2 and Gated DeltaNet maintain better performance through gating mechanisms that filter
irrelevant information.

3 The theoretical distinction lies in the optimization approach: Longhorn uses implicit online learning (Kulis
& Bartlett, 2010) to derive closed-form globally optimal updates, while DeltaNet optimizes the same objective
through one-step explicit gradient descent, as noted by Liu et al. (2024).

5

Published as a conference paper at ICLR 2025

Delta rule helps memorization. In S-NIAH-3, values change from numbers to UUIDs, testing
complex pattern memorization. Mamba2’s performance drops quickly, while Gated DeltaNet performs better, verifying that the delta rule indeed has better memorization ability.

3.3 A LGORITHM : H ARDWARE - EFFICIENT C HUNKWISE TRAINING

In this subsection, we derive a hardware-efficient chunkwise algorithm for training Gated DeltaNet.
By partially expanding the recurrence in Eq. 10, we have



β [ [i] t] [v] [ [i] t] [k] [ [i] t [⊺] ]

r
�

j=i+1

�I − β [ [j] t] [k] [ [j] t] [k] [ [j] t [⊺] ] �



�I − β [ [i] t] [k] [ [i] t] [k] [ [i] t [⊺] ] � [�]

S [r] [t] [=][ S] [[][t][]]

r
�
� i=1

α [i] [t]

+

r
�

i=1

α [j]

[t]

� �� �
:=F [r] [t]

� �� �
:=G [r] [t]

It is easy to see that F [r] [t] [=][ γ] [ [r] t] [P] [r] [t] [=][ ←−] P [r] [t] [. As for][ G] [r] [t] [, we adapt Eq.][ 5][ as follows,]

�

γ [ [r] t]
γ [ [i] t]

k [ [i] t [⊺] ] [k] [ [r] t] [)]

��

�

γ [ [r] t]
γ [ [i] t]

∈ R [d] [v]

r−1
�

i=1

u˜ [i] [t] [(]

G [r] [t] [=]

r
�

i=1

u˜ [i] [t] [k] [ [i] t [⊺] ] [∈] [R] [d] [v] [×][d] [k] u˜ [r] [t] [=][ β] [ [r] t]

v [ [r] t] [−]

(see §A for a proof). By UT transform, we have the matrix form:

� −1
U [t] = �I + strictLower �diag �β [t] � (Γ [t] ⊙ K [t] K [⊺] [t] [)] �� diag �β [t] � V [t] ∈ R [C][×][d] [v]

Similar to how Mamba2 extends linear attention (Eq. 1), we can adapt DeltaNet’s chunkwise algorithm (Eq. 8-9) for Gated DeltaNet to enable hardware-efficient training as follows:

S [t+1] = [−→] S [t] + �U� [t] − W [←−−] [t] S [⊺] [t] � ⊺ −−→K [t] ∈ R [d] [v] [×][d] [k]

O [t] = [←−−] Q [t] S [⊺] [t] [+ (][Q] [[][t][]] [K] [⊺] [t] [⊙] [M][)] �U� [t] − W [←−−] [t] S [⊺] [t] � ∈ R [C][×][d] [v]

γ [ [C] t]
where [←−] q [ [r] t] [=][ γ] [ [r] t] [q] [ [r] t] [,][ ←−−] w [ [r] t] [=][ γ] [ [r] t] [w] [ [r] t] [,][ −→] k [ [r] t] [=] γ [ [r] t] [k] [ [r] t] [, and][ −→] S [t] = γ [ [C] t] [S] [[][t][]] [ like we defined in Eq.][ 2][.]

3.4 G ATED D ELTA N ETWORKS AND H YBRID M ODELS

Token mixer block. The basic Gated DeltaNet follows Llama’s macro architecture, stacking token mixer layers with SwiGLU MLP layers, but replaces self-attention with gated delta rule token
mixing. Fig. 1 (right) shows its block design. For the gated delta rule (Eq. 10), queries, keys and
values {q, k, v} are generated through linear projection, short convolution and SiLU, with L2 normalization applied to q, k for training stability. α, β use linear projection only. [4] Following Sun et al.
(2023a), the output is processed through normalization and gating before applying output projection.

Hybrid models. Linear transformers have limitations in modeling local shifts and comparisons,
and their fixed state size makes it hard for retrieval tasks (Arora et al., 2024a). Following recent
hybrid architectures like Griffin (De et al., 2024) and Samba (Ren et al., 2024), we combine linear
recurrent layers with sliding window attention (SWA), resulting in GatedDeltaNet-H1. We also stack
Mamba2, GatedDeltaNet and SWA, resulting in GatedDeltaNet-H2.

4 E XPERIMENTS

Setup Our experiments encompass a comprehensive comparison of recent state-of-the-art architectures, including pure Transformer models, RNN-based approaches, and hybrid architectures. We
evaluate against the following baselines: RetNet (Sun et al., 2023a), HGRN2 (Qin et al., 2024b),
Mamba (Gu & Dao, 2023), Mamba2 (Dao & Gu, 2024b), Samba (Ren et al., 2024), and DeltaNet
(Yang et al., 2024b). For fair comparison, all models are trained under identical conditions with
1.3B parameters on 100B tokens sampled from the FineWeb-Edu dataset (Penedo et al., 2024). We
use the AdamW optimizer with a peak learning rate of 4e-4, weight decay of 0.1, and gradient clipping of 1.0. The learning rate follows a cosine annealing schedule with a 1B token warm-up period

4
We use Mamba2’s parameterization for α but omit it for brevity.

6

Published as a conference paper at ICLR 2025

Outputs

Outputs

![](./GATED-DETLA.pdf-6-1.png)

N ×

![](./GATED-DETLA.pdf-6-0.png)

Gated DeltaNet-H1 Gated DeltaNet-H2

Block Design

Figure 1: Visualization of the (hybrid) architecture and block design of Gated DeltaNet models. Gated
DeltaNet-H1 and H2 use Gated DeltaNet + SWA and Mamba2 + Gated DeltaNet + SWA patterns, respectively. In the block design, query/key paths consist of linear proj., shortconv., SiLU and L2 norm; value path
includes linear proj., shortconv. and SiLU; alpha/beta use linear proj.; and output gate applies linear proj. with
SiLU.

Model Wiki. LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c SIQA BoolQ Avg.
ppl ↓ ppl ↓ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑

Recurrent models

RetNet 19.08 17.27 40.52 70.07 49.16 54.14 67.34 33.78 40.78 60.39 52.02

HGRN2 19.10 17.69 39.54 70.45 49.53 52.80 69.40 35.32 40.63 56.66 51.79

Mamba 17.92 15.06 43.98 71.32 52.91 52.95 69.52 35.40 37.76 61.13 53.12

Mamba2 16.56 12.56 45.66 71.87 55.67 55.24 72.47 37.88 40.20 60.13 54.89

DeltaNet 17.71 16.88 42.46 70.72 50.93 53.35 68.47 35.66 40.22 55.29 52.14

Gated DeltaNet 16.42 12.17 46.65 72.25 55.76 57.45 71.21 38.39 40.63 60.24 55.32

Attention or hybrid models

Transformer++ 18.53 18.32 42.60 70.02 50.23 53.51 68.83 35.10 40.66 57.09 52.25

Samba 16.13 13.29 44.94 70.94 53.42 55.56 68.81 36.17 39.96 62.11 54.00

Gated DeltaNet-H1 16.07 12.12 47.73 72.57 56.53 58.40 71.75 40.10 41.40 63.21 56.40

Gated DeltaNet-H2 15.91 12.55 48.76 72.19 56.88 57.77 71.33 39.07 41.91 61.55 56.18

Table 3: Performance comparison on language modeling and zero-shot common-sense reasoning.

and batch size of 0.5M tokens. All models employ the Llama2 tokenizer with a vocabulary size of
32,000. For sequence modeling, we set the training length to 4K tokens, with Samba and our hybrid
models using a sliding window size of 2K. See § B.1 for evaluation settings and § B.2 for ablation
studies.

Common-sense reasoning In Table 3,
we present the language modeling perplexity and zero-shot accuracy on commonsense reasoning benchmarks for models
with 400M and 1.3B parameters. Gated
DeltaNet consistently outperforms other
linear models, including RetNet, HGRN2,
Mamba, Mamba2, and DeltaNet, at both
scales. As expected, the hybrid variant further enhances performance.

In-context retrieval on real-world data
Table 4 presents results on real-world
recall-intensive tasks used by Arora et al.
(2024b). As expected, linear recurrent
models show a significant performance

Models SWDE SQD FDA TQA NQ Drop Avg

Recurrent models

RetNet 14.0 28.5 7.0 54.4 16.2 17.3 22.9

HGRN2 8.3 25.3 4.8 51.2 14.2 16.9 20.1

Mamba 9.8 25.8 3.7 54.3 14.9 17.4 21.0

Mamba2 19.1 33.6 25.3 61.0 20.8 19.2 29.8

DeltaNet 17.9 30.9 18.4 53.9 17.3 18.6 26.2

Gated DeltaNet 25.4 34.8 23.7 60.0 20.0 19.8 30.6

Attention or hybrid models

Transformer++ 29.5 38.0 52.2 58.3 22.5 21.6 37.0

Samba 33.0 39.2 50.5 57.7 23.5 20.2 37.3

Gated DeltaNet-H1 35.6 39.7 52.0 60.1 24.6 22.2 39.0

Gated DeltaNet-H2 38.2 40.4 50.7 63.3 24.8 23.3 40.1

Table 4: Accuracy on recall-world retrieval tasks with input
truncated to 2K tokens. SQD: SQUADE. TQA: Trivial QA.

7

Published as a conference paper at ICLR 2025

gap compared to Transformers, while hybrid models combining linear recurrence and attention outperform pure attention models in retrieval tasks.

For pure recurrent models, despite DeltaNet’s superior performance on synthetic in-context retrieval
tasks (Yang et al., 2024b), its real-world retrieval performance lags behind Mamba2, consistent
with our observations in S-NIAH-2 and S-NIAH-3 (Table 2). Gated DeltaNet outperforms both
DeltaNet and Mamba2 thanks to its gated delta rule, though the improvement margin is smaller
than in Table 2. We attribute this reduced performance gap to instruction-unaligned small language
models being prone to repetition errors, which are the primary source of errors in these tasks (cf.
Arora et al. (2024b, Appendix E)). Since this issue is largely independent of the update rule choice,
the performance differences between models are less pronounced compared to Table 2.

GovReport

9

8

7

6

![](./GATED-DETLA.pdf-7-0.png)
4k 8k 14k 20k

Qasper

20

18

16

14

![](./GATED-DETLA.pdf-7-3.png)

4k 8k 14k 20k

Length

QMSum

18

16

14

![](./GATED-DETLA.pdf-7-1.png)

4k 8k 14k 20k

CodeParrot

20

15

10

5

![](./GATED-DETLA.pdf-7-5.png)
4k 8k 14k 20k

Length

NarrativeQA

20

18

16

![](./GATED-DETLA.pdf-7-2.png)

4k 8k 20k

PG19

15

14

13

![](./GATED-DETLA.pdf-7-4.png)

4k 8k 14k 20k

Length

Mamba1 DeltaNet Mamba2 Samba

GatedDeltaNet GatedDeltaNet-H1 GatedDeltaNet-H2

Figure 2: Length extrapolation on six long benchmarks.

Length extrapolation on long sequences. As shown in Fig.2, we evaluate the models’ capacity to
extrapolate to sequences of up to 20K tokens across six long-context benchmarks. Gated DeltaNet
achieves the lowest overall perplexity across tasks among RNN models. While we observe mixed
results in length extrapolation, Gated DeltaNet exhibits relatively more robust performance, suggesting better memory management. The hybrid models further improve upon this by leveraging
attention for local context modeling, which reduces the memory management burden on their recurrent components. Future work will explore these models’ capabilities on even longer sequences.

Long context understanding As demonstrated in Table 5, we evaluated the models’ performance
on LongBench (Bai et al., 2023). In recurrent models, Gated DeltaNet shows consistent advantages,
especially in single-doc QA, few-shot in-context learning, and Code tasks, demonstrating its superior
capabilities in retrieval, in-context learning, and state tracking, respectively.

Throughput Comparison. The training throughput comparison across different models is presented in Fig. 3. As our analysis shows, the proposed gated delta rule introduces only marginal
overhead compared to the original delta rule, with Gated DeltaNet achieving essentially the same
throughput as DeltaNet. Both are slightly slower than Mamba2 (2-3K tokens/sec) due to their more
expressive transition matrices.

The Transformer++ achieves the best performance in the 2K context window domain, thanks to the
highly optimized Flash-Attention-2 kernel (Dao, 2023). Consequently, hybrid approaches combining 2K window-size SWA attention with other token mixers demonstrate higher throughput than

8

Published as a conference paper at ICLR 2025

Single-Doc QA Multi-Doc QA Summarization Few-shot Code Avg
Model NQA QQA MFQ HQA 2WM Mus GvR QMS MNs TRC TQA SSM LCC RBP

Recurrent models

RetNet 12.1 10.7 19.1 10.7 18.0 5.8 4.8 15.8 7.9 19.0 18.0 12.8 14.1 17.9 13.2

HGRN2 10.7 12.1 19.1 11.3 15.7 6.0 5.2 15.1 9.2 16.0 15.8 10.3 18.6 20.8 13.5

Mamba 13.0 10.1 20.4 10.1 16.7 6.0 7.2 15.9 8.4 23.1 21.9 11.2 17.9 19.0 14.6

DeltaNet 12.9 10.8 21.5 10.9 13.2 5.1 6.5 13.5 7.2 15.5 23.3 11.6 17.6 20.3 13.6

Mamba2 11.1 11.3 18.6 11.8 15.1 6.7 6.7 14.5 7.4 13.0 23.6 8.4 17.9 20.6 13.5

Gated DeltaNet 14.1 14.0 23.3 13.7 14.4 5.8 7.5 16.4 7.9 30.0 22.4 23.0 18.7 22.1 16.6

Attention or hyrbid models
Transformer++ 11.8 9.3 10.0 10.9 4.2 6.1 7.4 15.8 6.6 16.9 13.5 3.9 17.2 18.7 11.0

Samba 12.5 12.9 25.4 11.2 19.7 6.8 9.1 15.7 11.0 20.0 22.7 22.8 18.1 21.1 15.9

Gated DeltaNet-H1 14.5 12.3 26.6 12.6 23.6 6.1 9.1 16.1 12.8 33.5 23.9 26.8 15.5 19.2 17.8

Gated DeltaNet-H2 12.7 13.0 27.1 12.7 20.6 7.5 10.4 16.2 13.0 40.5 22.7 27.9 19.9 22.1 18.4

Table 5: Accuracy on 14 tasks from LongBench (Bai et al., 2023): Narrative QA, QasperQA, MultiField QA,
HotpotQA, 2WikiMulti QA, Musique, GovReport, QMSum, MultiNews, TRec, Trivia QA, SamSum, LCC,
and RepoBench-P by order.

60

55

50

45

40

35

30

25

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
|||Transformer++<br>Gated DeltaNet|DeltaNet<br>Mamba1||||
||||||||
|||Mamba2<br>|Gated DeltaNet-H1<br>||||
||||||||
|||Samba|Gated DeltaNet-H2|Gated DeltaNet-H2|||
||||||||

2K×16 4K×8 8K×4 16K×2

Sequence Length × Batch Size

Figure 3: Training throughput comparison of 1.3B models on a single H100 GPU.

standalone mixers: Samba outperforms Mamba, while Gated DeltaNet-H1 and -H2 outperform
Gated DeltaNet. Notably, Gated DeltaNet-H1 maintains compelling training throughput across all
sequence lengths, even on short sequences.

5 R ELATED W ORK

Gated linear RNN. Large linear recurrent language models have attracted significant attention
due to their training and inference efficiency. The field of linear RNNs has rapidly evolved from
using data-independent decay mechanisms, as exemplified by models like S4 (Gu et al., 2022), S5
(Smith et al., 2023), LRU (Orvieto et al., 2023), RWKV4/5 (Peng et al., 2023), and RetNet (Sun
et al., 2023a), to incorporating data-dependent decay mechanisms in more recent architectures such
as HGRN1/2 (Qin et al., 2024b; 2023b), Mamba1/2 (Gu & Dao, 2023; Dao & Gu, 2024a), RWKV6
(Peng et al., 2024), GSA (Zhang et al., 2024). This transition stems from the proven advantages of
gating/forgetting mechanisms (termed selective mechanisms in Mamba)—a classical concept originating in the gated RNN literature (Gers et al., 2000) whose significance has been consistently
reaffirmed (Greff et al., 2015; van der Westhuizen & Lasenby, 2018; Qin et al., 2024b; 2023b; Gu &
Dao, 2023).

Modern forget gates differ from traditional designs like those in LSTM by removing the dependency on the previous hidden state, relying solely on input data. This modification enables efficient
parallelism across sequence lengths (Martin & Cundy, 2018; Qin et al., 2023b). The absence of a
forget gate has been a notable limitation in DeltaNet, and our gated extension addresses this gap in

9

Published as a conference paper at ICLR 2025

a natural, effective, and hardware-efficient way. We also note a recent concurrent work RWKV-7 [5]
using a similar idea, but with a more relaxable formalism using diagonal-plus-low-rank transitions:
S t = S t−1 (diag(d t ) − a t b [⊤] t [) +][ v] [t] [k] t [⊤] [where][ d] [t] [,][ a] [t] [,][ b] [t] [∈] [R] [d] [k] [. The chunkwise algorithm could be]
similarly adapted to this case, as implemented in Flash Linear Attention (Yang & Zhang, 2024). [6]

Delta rule. The delta learning rule demonstrates superior memory capacity compared to Hebbian learning (Gardner, 1988; Prados & Kak, 1989), an advantage DeltaNet leverages while linear
transformers rely on Hebbian-like rules. This memory capacity advantage is evident in synthetic
in-context learning tasks and extends to language modeling (Irie et al., 2021; Yang et al., 2024b),
reinforcement learning (Irie et al., 2022b), and image generation (Irie & Schmidhuber, 2023).
Yang et al. (2024b) parallelized delta rule computation and demonstrated how DeltaNet’s datadependent identity-plus-low-rank structure (I − β t k t k t [⊺] [) offers greater flexibility than Mamba2’s]
data-dependent diagonal matrices (α t I). This structural advantage could enable complex reasoning, including regular language recognition (Fan et al., 2024; Grazzi et al., 2024) and state-tracking
beyond TC [0] complexity (Merrill et al., 2024)—crucial for coding and reasoning applications.

Despite these significant advantages, the delta rule faces theoretical limitations (Irie et al., 2023) and
shows only moderate performance on real-world datasets (Yang et al., 2024b), suggesting room for
improvement. Previous attempts to enhance expressiveness through nonlinear recurrence (Irie et al.,
2021; 2022b) addressed some limitations but sacrificed training parallelism, creating a performanceefficiency tradeoff. Recent work proposes some enhancements without compromising parallelism
for better state tracking performance, including using negative eigenvalues (Grazzi et al., 2024) and
multiple products of householder transition matrices (Siems et al., 2025) which enable high-rank
transformations. These methods could be applied to Gated DeltaNet seamlessly.

From a (online) learning objective perspective, alternative formulations could further extend expressiveness: nonlinear regression (L(S t ) = 12 [||][f] [S] [t] [(][k] [t] [)][ −] [v] [t] [||] [2] [) as in TTT (][Sun et al.][,][ 2024a][) and]
Titans (Behrouz et al., 2024), where f S is a nonlinear function parameterized by S; or regression
considering the entire history (L(S t ) = 21 � ti=1 [||][S] [t] [k] [i] [ −] [v] [i] [||] [2] [) as in Mesa layer (][von Oswald]
et al., 2024)—analogous to the difference between Least Mean Square and Recursive Least Square
algorithms. However, these more expressive variants introduce nonlinear recurrence and require
workarounds, such as performing nonlinear updates only after processing entire chunks (as in TTT
and Titans); or approximating nonlinear recurrence methods like Lim et al. (2024); Gonzalez et al.
(2024); Schöne et al. (2025).

Hybrid models. In this work, we explore interleaving hybrid attention layers across layers, which
is commonly used such as in MiniMax-01 (MiniMax et al., 2025) and Hybrid Mamba2-Attention
(Waleffe et al., 2024). It is also interesting to investigate hybrid linear/softmax attention within a
single layer (Hua et al., 2022a; Zancato et al., 2024; Munkhdalai et al., 2024; Nunez et al., 2024;
Dong et al., 2025; Zhang et al., 2025).

6 C ONCLUSION

In this work, we introduced Gated DeltaNet, which enables better key-value association learning
compared to Mamba2 and more adaptive memory clearance than DeltaNet, leading to consistently
better empirical results across various tasks. We extended the parallel algorithm from Yang et al.
(2024b) to enable hardware-efficient training of Gated DeltaNet. Our hybrid Gated DeltaNet model
achieves even higher training throughputand overall performance, making it well-suited for practical
deployment.

A CKNOWLEDGMENT

We thank Yu Zhang for assistance with figure creation and model evaluation; Kazuki Irie for providing valuable feedback on the draft; Simeng Sun and Zhixuan Lin for insightful discussions on

5 [https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)
6 [https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule)
[generalized_delta_rule.](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule)

10

Published as a conference paper at ICLR 2025

long-sequence task evaluation settings; and Eric Alcaide and Volodymyr Kyrylov for their helpful
discussions on the online learning perspective of DeltaNet.

R EFERENCES

Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. In-Context Language Learning: Archi[tectures and Algorithms, 2024. URL https://arxiv.org/abs/2401.12973.](https://arxiv.org/abs/2401.12973)

Maksym Andriushchenko, Francesco D’Angelo, Aditya Vardhan Varre, and Nicolas Flammarion.
Why do we need weight decay in modern deep learning? ArXiv, abs/2310.04415, 2023. URL
[https://api.semanticscholar.org/CorpusID:263829417.](https://api.semanticscholar.org/CorpusID:263829417)

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra,
and Christopher Ré. Zoology: Measuring and improving recall in efficient language models.
[ArXiv preprint, abs/2312.04927, 2023a. URL https://arxiv.org/abs/2312.04927.](https://arxiv.org/abs/2312.04927)

Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, and Christopher Ré. Language Models Enable Simple Systems for Generating Structured
[Views of Heterogeneous Data Lakes, 2023b. URL https://arxiv.org/abs/2304.09433.](https://arxiv.org/abs/2304.09433)

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley,
James Zou, Atri Rudra, and Christopher Ré. Simple linear attention language models balance
[the recall-throughput tradeoff. ArXiv preprint, abs/2402.18668, 2024a. URL https://arxiv.](https://arxiv.org/abs/2402.18668)
[org/abs/2402.18668.](https://arxiv.org/abs/2402.18668)

Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao,
Ashish Rao, Atri Rudra, and Christopher Ré. Just read twice: closing the recall gap for recurrent
[language models, 2024b. URL https://arxiv.org/abs/2407.05483.](https://arxiv.org/abs/2407.05483)

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du,
Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long
[context understanding. ArXiv preprint, abs/2308.14508, 2023. URL https://arxiv.org/abs/](https://arxiv.org/abs/2308.14508)
[2308.14508.](https://arxiv.org/abs/2308.14508)

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova,
Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended
[long short-term memory. ArXiv preprint, abs/2405.04517, 2024. URL https://arxiv.org/](https://arxiv.org/abs/2405.04517)
[abs/2405.04517.](https://arxiv.org/abs/2405.04517)

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time, 2024.
[URL https://arxiv.org/abs/2501.00663.](https://arxiv.org/abs/2501.00663)

Christian H. Bischof and Charles Van Loan. The WY representation for products of householder
matrices. In SIAM Conference on Parallel Processing for Scientific Computing, 1985. URL
[https://api.semanticscholar.org/CorpusID:36094006.](https://api.semanticscholar.org/CorpusID:36094006)

Yonatan Bisk, Rowan Zellers, Ronan LeBras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about
physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence
Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 7432–7439. AAAI Press, 2020.
[URL https://aaai.org/ojs/index.php/AAAI/article/view/6239.](https://aaai.org/ojs/index.php/AAAI/article/view/6239)

Yuhong Chou, Man Yao, Kexin Wang, Yuqi Pan, Rui-Jie Zhu, Jibin Wu, Yiran Zhong, Yu Qiao,
Bo XU, and Guoqi Li. MetaLA: Unified optimal linear approximation to softmax attention map.
In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL
[https://openreview.net/forum?id=Y8YVCOMEpz.](https://openreview.net/forum?id=Y8YVCOMEpz)

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein,
Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,

11

Published as a conference paper at ICLR 2025

Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, 2019. Association
[for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL https://aclanthology.org/](https://aclanthology.org/N19-1300)
[N19-1300.](https://aclanthology.org/N19-1300)

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
[ArXiv preprint, abs/1803.05457, 2018. URL https://arxiv.org/abs/1803.05457.](https://arxiv.org/abs/1803.05457)

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. ArXiv
[preprint, abs/2307.08691, 2023. URL https://arxiv.org/abs/2307.08691.](https://arxiv.org/abs/2307.08691)

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through
structured state space duality. arXiv preprint arXiv: 2405.21060, 2024a.

Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms
through structured state space duality. In Proceedings of the 41st International Conference on
Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 10041–10071.
[PMLR, 2024b. URL https://proceedings.mlr.press/v235/dao24a.html.](https://proceedings.mlr.press/v235/dao24a.html)

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. A dataset of
information-seeking questions and answers anchored in research papers. In Kristina Toutanova,
Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, pp. 4599–4610, Online, 2021. Association for Computational Linguistics. doi:
[10.18653/v1/2021.naacl-main.365. URL https://aclanthology.org/2021.naacl-main.365.](https://aclanthology.org/2021.naacl-main.365)

Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, and
Caglar Gulcehre. Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient
[Language Models, 2024. URL https://arxiv.org/abs/2402.19427.](https://arxiv.org/abs/2402.19427)

Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, ZIJIA CHEN, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van keirsbilck, Min-Hung Chen, Yoshi Suhara, Yingyan Celine
Lin, Jan Kautz, and Pavlo Molchanov. Hymba: A hybrid-head architecture for small language
models. In The Thirteenth International Conference on Learning Representations, 2025. URL
[https://openreview.net/forum?id=A1ztozypga.](https://openreview.net/forum?id=A1ztozypga)

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.
In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference
of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2368–2378, Minneapolis, Minnesota,
[2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https://](https://aclanthology.org/N19-1246)
[aclanthology.org/N19-1246.](https://aclanthology.org/N19-1246)

Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale
multi-document summarization dataset and abstractive hierarchical model. In Anna Korhonen,
David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1074–1084, Florence, Italy, 2019. Association for
[Computational Linguistics. doi: 10.18653/v1/P19-1102. URL https://aclanthology.org/](https://aclanthology.org/P19-1102)
[P19-1102.](https://aclanthology.org/P19-1102)

Ting-Han Fan, Ta-Chung Chi, and Alexander Rudnicky. Advancing regular language reasoning in
linear recurrent neural networks. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 45–53, Mexico
[City, Mexico, 2024. Association for Computational Linguistics. URL https://aclanthology.](https://aclanthology.org/2024.naacl-short.4)
[org/2024.naacl-short.4.](https://aclanthology.org/2024.naacl-short.4)

12

Published as a conference paper at ICLR 2025

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric
Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language
model evaluation, 2021.

E. Gardner. The space of interactions in neural network models. Journal of Physics A, 21:257–270,
[1988. URL https://api.semanticscholar.org/CorpusID:15378089.](https://api.semanticscholar.org/CorpusID:15378089)

Felix A. Gers, Jürgen Schmidhuber, and Fred A. Cummins. Learning to forget: Continual prediction
with LSTM. Neural Comput., 12(10):2451–2471, 2000.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A humanannotated dialogue dataset for abstractive summarization. In Lu Wang, Jackie Chi Kit Cheung,
Giuseppe Carenini, and Fei Liu (eds.), Proceedings of the 2nd Workshop on New Frontiers in
Summarization, pp. 70–79, Hong Kong, China, 2019. Association for Computational Linguistics.
[doi: 10.18653/v1/D19-5409. URL https://aclanthology.org/D19-5409.](https://aclanthology.org/D19-5409)

Xavier Gonzalez, Andrew Warrington, Jimmy T.H. Smith, and Scott Linderman. Towards scalable and stable parallelization of nonlinear RNNs. In The Thirty-eighth Annual Conference
on Neural Information Processing Systems, 2024. [URL https://openreview.net/forum?](https://openreview.net/forum?id=hBCxxVQDBw)
[id=hBCxxVQDBw.](https://openreview.net/forum?id=hBCxxVQDBw)

Riccardo Grazzi, Julien N. Siems, Jorg K. H. Franke, Arber Zela, Frank Hutter, and Massimiliano Pontil. Unlocking state-tracking in linear rnns through negative eigenvalues. 2024. URL
[https://api.semanticscholar.org/CorpusID:274141450.](https://api.semanticscholar.org/CorpusID:274141450)

Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber.
Lstm: A search space odyssey. IEEE Transactions on Neural Networks and Learning Systems,
[28:2222–2232, 2015. URL https://api.semanticscholar.org/CorpusID:3356463.](https://api.semanticscholar.org/CorpusID:3356463)

Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2023.

Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured
state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Vir[tual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?](https://openreview.net/forum?id=uYLFoz1vlAC)
[id=uYLFoz1vlAC.](https://openreview.net/forum?id=uYLFoz1vlAC)

Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Julian J. McAuley. Longcoder: A long-range pretrained language model for code completion. In Andreas Krause, Emma Brunskill, Kyunghyun
Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference
on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of
[Proceedings of Machine Learning Research, pp. 12098–12107. PMLR, 2023. URL https://](https://proceedings.mlr.press/v202/guo23j.html)
[proceedings.mlr.press/v202/guo23j.html.](https://proceedings.mlr.press/v202/guo23j.html)

Zhihao He, Hang Yu, Zi Gong, Shizhan Liu, Jianguo Li, and Weiyao Lin. Rodimus*: Breaking the
accuracy-efficiency trade-off with efficient attentions. In The Thirteenth International Conference
[on Learning Representations, 2025. URL https://openreview.net/forum?id=IIVYiJ1ggK.](https://openreview.net/forum?id=IIVYiJ1ggK)

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel,
and Chengqing Zong (eds.), Proceedings of the 28th International Conference on Computational
Linguistics, pp. 6609–6625, Barcelona, Spain (Online), 2020. International Committee on Com[putational Linguistics. doi: 10.18653/v1/2020.coling-main.580. URL https://aclanthology.](https://aclanthology.org/2020.coling-main.580)
[org/2020.coling-main.580.](https://aclanthology.org/2020.coling-main.580)

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang
Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language
[models? ArXiv preprint, abs/2404.06654, 2024. URL https://arxiv.org/abs/2404.06654.](https://arxiv.org/abs/2404.06654)

Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V. Le. Transformer quality in linear time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.),

13

Published as a conference paper at ICLR 2025

International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 9099–9117. PMLR,
[2022a. URL https://proceedings.mlr.press/v162/hua22a.html.](https://proceedings.mlr.press/v162/hua22a.html)

Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V. Le. Transformer quality in linear time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.),
International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 9099–9117. PMLR,
[2022b. URL https://proceedings.mlr.press/v162/hua22a.html.](https://proceedings.mlr.press/v162/hua22a.html)

Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for
long document summarization. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek
Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou
(eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pp. 1419–1436, Online, 2021. Asso[ciation for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.112. URL https://](https://aclanthology.org/2021.naacl-main.112)
[aclanthology.org/2021.naacl-main.112.](https://aclanthology.org/2021.naacl-main.112)

Kazuki Irie and Jürgen Schmidhuber. Images as weight matrices: Sequential image generation
through synaptic learning rules. In The Eleventh International Conference on Learning Repre[sentations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://](https://openreview.net/pdf?id=ddad0PNUvV)
[openreview.net/pdf?id=ddad0PNUvV.](https://openreview.net/pdf?id=ddad0PNUvV)

Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. Going beyond
linear transformers with recurrent fast weight programmers. In Marc’Aurelio Ranzato,
Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan
(eds.), Advances in Neural Information Processing Systems 34: Annual Conference on
Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 7703–7717, 2021. [URL https://proceedings.neurips.cc/paper/2021/hash/](https://proceedings.neurips.cc/paper/2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html)
[3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html.](https://proceedings.neurips.cc/paper/2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html)

Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In Kamalika
Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland,
USA, volume 162 of Proceedings of Machine Learning Research, pp. 9639–9659. PMLR, 2022a.
[URL https://proceedings.mlr.press/v162/irie22a.html.](https://proceedings.mlr.press/v162/irie22a.html)

Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. A modern self-referential
weight matrix that learns to modify itself. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song,
Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine
Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings
[of Machine Learning Research, pp. 9660–9677. PMLR, 2022b. URL https://proceedings.](https://proceedings.mlr.press/v162/irie22b.html)
[mlr.press/v162/irie22b.html.](https://proceedings.mlr.press/v162/irie22b.html)

Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. Practical computational power of linear
transformers and their recurrent and self-referential extensions. In Houda Bouamor, Juan Pino,
and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pp. 9455–9465, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.588. [URL https://aclanthology.org/2023.](https://aclanthology.org/2023.emnlp-main.588)
[emnlp-main.588.](https://aclanthology.org/2023.emnlp-main.588)

Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat After Me: Transformers are Better than State Space Models at Copying. ArXiv preprint, abs/2402.01032, 2024.
[URL https://arxiv.org/abs/2402.01032.](https://arxiv.org/abs/2402.01032)

Thierry Joffrain, Tze Meng Low, Enrique S. Quintana-Ortí, Robert A. van de Geijn, and Field
G. Van Zee. Accumulating householder transformations, revisited. ACM Trans. Math. Softw., 32:
[169–179, 2006. URL https://api.semanticscholar.org/CorpusID:15723171.](https://api.semanticscholar.org/CorpusID:15723171)

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan

14

Published as a conference paper at ICLR 2025

(eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, 2017a. Association for Computa[tional Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147.](https://aclanthology.org/P17-1147)

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan
(eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, 2017b. Association for Computa[tional Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147.](https://aclanthology.org/P17-1147)

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are
rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume
[119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020a. URL http://](http://proceedings.mlr.press/v119/katharopoulos20a.html)
[proceedings.mlr.press/v119/katharopoulos20a.html.](http://proceedings.mlr.press/v119/katharopoulos20a.html)

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are
rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume
[119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020b. URL http://](http://proceedings.mlr.press/v119/katharopoulos20a.html)
[proceedings.mlr.press/v119/katharopoulos20a.html.](http://proceedings.mlr.press/v119/katharopoulos20a.html)

Tomáš Koˇciský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis,
and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of
the Association for Computational Linguistics, 6:317–328, 2018. doi: 10.1162/tacl_a_00023.
[URL https://aclanthology.org/Q18-1023.](https://aclanthology.org/Q18-1023)

Anders Krogh and John A. Hertz. A simple weight decay can improve generalization. In Neural
[Information Processing Systems, 1991. URL https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:10137788)
[10137788.](https://api.semanticscholar.org/CorpusID:10137788)

Brian Kulis and Peter L. Bartlett. Implicit online learning. In Johannes Fürnkranz and Thorsten
Joachims (eds.), Proceedings of the 27th International Conference on Machine Learning (ICML[10), June 21-24, 2010, Haifa, Israel, pp. 575–582. Omnipress, 2010. URL https://icml.cc/](https://icml.cc/Conferences/2010/papers/429.pdf)
[Conferences/2010/papers/429.pdf.](https://icml.cc/Conferences/2010/papers/429.pdf)

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research. Transactions of the
Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL

[https://aclanthology.org/Q19-1026.](https://aclanthology.org/Q19-1026)

Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th International
[Conference on Computational Linguistics, 2002. URL https://aclanthology.org/C02-1150.](https://aclanthology.org/C02-1150)

Yi Heng Lim, Qi Zhu, Joshua Selfridge, and Muhammad Firmansyah Kasim. Parallelizing non[linear sequential models over the sequence length, 2024. URL https://arxiv.org/abs/2309.](https://arxiv.org/abs/2309.12252)
[12252.](https://arxiv.org/abs/2309.12252)

Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, and Qiang Liu.
@articleDBLP:journals/corr/abs-2407-14207, author = Bo Liu and Rui Wang and Lemeng
Wu and Yihao Feng and Peter Stone and Qiang Liu, title = Longhorn: State Space Models
are Amortized Online Learners, journal = CoRR, volume = abs/2407.14207, year = 2024,
url = https://doi.org/10.48550/arXiv.2407.14207, doi = 10.48550/ARXIV.2407.14207, eprinttype = arXiv, eprint = 2407.14207, timestamp = Fri, 23 Aug 2024 08:12:16 +0200, biburl
= https://dblp.org/rec/journals/corr/abs-2407-14207.bib, bibsource = dblp computer science
bibliography, https://dblp.org : State space models are amortized online learners. ArXiv preprint,
[abs/2407.14207, 2024. URL https://arxiv.org/abs/2407.14207.](https://arxiv.org/abs/2407.14207)

Tianyang Liu, Canwen Xu, and Julian McAuley. RepoBench: Benchmarking repository-level code
[auto-completion systems. ArXiv preprint, abs/2306.03091, 2023. URL https://arxiv.org/](https://arxiv.org/abs/2306.03091)
[abs/2306.03091.](https://arxiv.org/abs/2306.03091)

15

Published as a conference paper at ICLR 2025

Colin Lockard, Prashant Shiralkar, and Xin Luna Dong. OpenCeres: When open information extraction meets the semi-structured web. In Jill Burstein, Christy Doran, and Thamar Solorio
(eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers),
pp. 3047–3056, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi:
[10.18653/v1/N19-1309. URL https://aclanthology.org/N19-1309.](https://aclanthology.org/N19-1309)

Peng Lu, Ivan Kobyzev, Mehdi Rezagholizadeh, Boxing Chen, and Philippe Langlais. Regla: Refin[ing gated linear attention, 2025. URL https://arxiv.org/abs/2502.01578.](https://arxiv.org/abs/2502.01578)

Eric Martin and Chris Cundy. Parallelizing linear recurrent neural nets over sequence length. In
6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada,
[April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://](https://openreview.net/forum?id=HyUNwulC-)
[openreview.net/forum?id=HyUNwulC-.](https://openreview.net/forum?id=HyUNwulC-)

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France,
April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. [URL https://](https://openreview.net/forum?id=Byj72udxe)
[openreview.net/forum?id=Byj72udxe.](https://openreview.net/forum?id=Byj72udxe)

William Merrill, Jackson Petty, and Ashish Sabharwal. The Illusion of State in State-Space Models,
[2024. URL https://arxiv.org/abs/2404.08819.](https://arxiv.org/abs/2404.08819)

MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang,
Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze
Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie,
Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang,
Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin
Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang,
Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su,
Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie
Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li,
Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei
Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, and Zijia
[Wu. Minimax-01: Scaling foundation models with lightning attention, 2025. URL https://](https://arxiv.org/abs/2501.08313)
[arxiv.org/abs/2501.08313.](https://arxiv.org/abs/2501.08313)

Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Efficient
infinite context transformers with infini-attention. ArXiv preprint, abs/2404.07143, 2024. URL
[https://arxiv.org/abs/2404.07143.](https://arxiv.org/abs/2404.07143)

Elvis Nunez, Luca Zancato, Benjamin Bowman, Aditya Golatkar, Wei Xia, and Stefano Soatto.
Expansion span: Combining fading memory and retrieval in hybrid state space models, 2024.
[URL https://arxiv.org/abs/2412.13328.](https://arxiv.org/abs/2412.13328)

Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Çaglar Gülçehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In Andreas
Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan
Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023,
Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 26670–
[26698. PMLR, 2023. URL https://proceedings.mlr.press/v202/orvieto23a.html.](https://proceedings.mlr.press/v202/orvieto23a.html)

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi,
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset:
Word prediction requiring a broad discourse context. In Katrin Erk and Noah A. Smith (eds.), Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 1525–1534, Berlin, Germany, 2016. Association for Computational Linguistics.
[doi: 10.18653/v1/P16-1144. URL https://aclanthology.org/P16-1144.](https://aclanthology.org/P16-1144)

Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro
Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data
[at scale. ArXiv preprint, abs/2406.17557, 2024. URL https://arxiv.org/abs/2406.17557.](https://arxiv.org/abs/2406.17557)

16

Published as a conference paper at ICLR 2025

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman,
Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi
Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej
Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito,
Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Wo´zniak, Zhenyuan Zhang, Qinghua
Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer era. In Houda
Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14048–14077, Singapore, 2023. Association for Computational Lin[guistics. doi: 10.18653/v1/2023.findings-emnlp.936. URL https://aclanthology.org/2023.](https://aclanthology.org/2023.findings-emnlp.936)
[findings-emnlp.936.](https://aclanthology.org/2023.findings-emnlp.936)

Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Xingjian Du, Teddy Ferdinan, Haowen Hou, Przemysław Kazienko, Kranthi Kiran
GV, Jan Koco´n, Bartłomiej Koptyra, Satyapriya Krishna, Ronald McClelland Jr., Niklas Muennighoff, Fares Obeid, Atsushi Saito, Guangyu Song, Haoqin Tu, Stanisław Wo´zniak, Ruichong
Zhang, Bingchen Zhao, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui-Jie Zhu. Eagle and Finch:
[RWKV with Matrix-Valued States and Dynamic Recurrence, 2024. URL https://arxiv.org/](https://arxiv.org/abs/2404.05892)
[abs/2404.05892.](https://arxiv.org/abs/2404.05892)

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, and Lingpeng Kong.
Random feature attention. In 9th International Conference on Learning Representations, ICLR
[2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.](https://openreview.net/forum?id=QtTKTdVrFBB)
[net/forum?id=QtTKTdVrFBB.](https://openreview.net/forum?id=QtTKTdVrFBB)

DL Prados and SC Kak. Neural network capacity using delta rule. Electronics Letters, 3(25):197–
199, 1989.

Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Fei Yuan, Xiao Luo, Y. Qiao, and Yiran Zhong. Transnormerllm: A faster and better
[large language model with improved transnormer. 2023a. URL https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:260203124)
[org/CorpusID:260203124.](https://api.semanticscholar.org/CorpusID:260203124)

Zhen Qin, Songlin Yang, and Yiran Zhong. Hierarchically gated recurrent neural network for sequence modeling. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt,
and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA,
[December 10 - 16, 2023, 2023b. URL http://papers.nips.cc/paper_files/paper/2023/](http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html)
[hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html)

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention2: A free lunch for handling unlimited sequence lengths in large language models. 2024a.

Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong.
Hgrn2: Gated linear rnns with state expansion. ArXiv preprint, abs/2404.07904, 2024b. URL
[https://arxiv.org/abs/2404.07904.](https://arxiv.org/abs/2404.07904)

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions
for SQuAD. In Iryna Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 784–789, Melbourne, Australia, 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124.
[URL https://aclanthology.org/P18-2124.](https://aclanthology.org/P18-2124)

Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple
hybrid state space models for efficient unlimited context language modeling. ArXiv preprint,
[abs/2406.07522, 2024. URL https://arxiv.org/abs/2406.07522.](https://arxiv.org/abs/2406.07522)

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence
Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 8732–8740. AAAI Press, 2020.
[URL https://aaai.org/ojs/index.php/AAAI/article/view/6399.](https://aaai.org/ojs/index.php/AAAI/article/view/6399)

17

Published as a conference paper at ICLR 2025

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun
Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pp. 4463–4473, Hong Kong, China, 2019. Association for Computational Linguistics.
[doi: 10.18653/v1/D19-1454. URL https://aclanthology.org/D19-1454.](https://aclanthology.org/D19-1454)

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight
programmers. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International
Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139
[of Proceedings of Machine Learning Research, pp. 9355–9366. PMLR, 2021a. URL http://](http://proceedings.mlr.press/v139/schlag21a.html)
[proceedings.mlr.press/v139/schlag21a.html.](http://proceedings.mlr.press/v139/schlag21a.html)

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight
programmers. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International
Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139
[of Proceedings of Machine Learning Research, pp. 9355–9366. PMLR, 2021b. URL http://](http://proceedings.mlr.press/v139/schlag21a.html)
[proceedings.mlr.press/v139/schlag21a.html.](http://proceedings.mlr.press/v139/schlag21a.html)

Mark Schöne, Babak Rahmani, Heiner Kremer, Fabian Falck, Hitesh Ballani, and Jannes Gladrow.
Implicit language models are rnns: Balancing parallelization and expressivity, 2025. URL
[https://arxiv.org/abs/2502.07827.](https://arxiv.org/abs/2502.07827)

Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo
Grazzi. Deltaproduct: Increasing the expressivity of deltanet through products of householders,
[2025. URL https://arxiv.org/abs/2502.10297.](https://arxiv.org/abs/2502.10297)

Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers
for sequence modeling. In The Eleventh International Conference on Learning Representations,
[ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.](https://openreview.net/pdf?id=Ai8Hw3AXqks)
[net/pdf?id=Ai8Hw3AXqks.](https://openreview.net/pdf?id=Ai8Hw3AXqks)

Paul Smolensky. Tensor product variable binding and the representation of symbolic structures
in connectionist systems. Artif. Intell., 46(1-2):159–216, 1990. doi: 10.1016/0004-3702(90)
[90007-M. URL https://doi.org/10.1016/0004-3702(90)90007-M.](https://doi.org/10.1016/0004-3702(90)90007-M)

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei
Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to
(learn at test time): Rnns with expressive hidden states. ArXiv preprint, abs/2407.04620, 2024a.
[URL https://arxiv.org/abs/2407.04620.](https://arxiv.org/abs/2407.04620)

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
Furu Wei. Retentive network: A successor to transformer for large language models. ArXiv
[preprint, abs/2307.08621, 2023a. URL https://arxiv.org/abs/2307.08621.](https://arxiv.org/abs/2307.08621)

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
Furu Wei. Retentive network: A successor to transformer for large language models. ArXiv
[preprint, abs/2307.08621, 2023b. URL https://arxiv.org/abs/2307.08621.](https://arxiv.org/abs/2307.08621)

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong
Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models.
[ArXiv preprint, abs/2405.05254, 2024b. URL https://arxiv.org/abs/2405.05254.](https://arxiv.org/abs/2405.05254)

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computa[tional Linguistics, 10:539–554, 2022. doi: 10.1162/tacl_a_00475. URL https://aclanthology.](https://aclanthology.org/2022.tacl-1.31)
[org/2022.tacl-1.31.](https://aclanthology.org/2022.tacl-1.31)

Jos van der Westhuizen and Joan Lasenby. The unreasonable effectiveness of the forget gate. ArXiv
[preprint, abs/1804.04849, 2018. URL https://arxiv.org/abs/1804.04849.](https://arxiv.org/abs/1804.04849)

18

Published as a conference paper at ICLR 2025

Johannes von Oswald, Maximilian Schlegel, Alexander Meulemans, Seijin Kobayashi, Eyvind
Niklasson, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Blaise Agüera y Arcas, Max Vladymyrov, Razvan Pascanu, and João Sacramento. Uncovering mesa-optimization
[algorithms in transformers, 2024. URL https://arxiv.org/abs/2309.05858.](https://arxiv.org/abs/2309.05858)

Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert
Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh,
Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. An empirical study of
[mamba-based language models, 2024. URL https://arxiv.org/abs/2406.07887.](https://arxiv.org/abs/2406.07887)

Ke Alexander Wang, Jiaxin Shi, and Emily B. Fox. Test-time regression: a unifying framework
[for designing sequence models with associative memory, 2025. URL https://arxiv.org/abs/](https://arxiv.org/abs/2501.12352)
[2501.12352.](https://arxiv.org/abs/2501.12352)

Kaiyue Wen, Xingyu Dang, and Kaifeng Lyu. RNNs are not Transformers (Yet): The Key Bottle[neck on In-context Retrieval. ArXiv preprint, abs/2402.18510, 2024. URL https://arxiv.org/](https://arxiv.org/abs/2402.18510)
[abs/2402.18510.](https://arxiv.org/abs/2402.18510)

Bernard Widrow, Marcian E Hoff, et al. Adaptive switching circuits. In IRE WESCON convention
record, volume 4, pp. 96–104. New York, 1960.

Songlin Yang and Yu Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism, 2024. [URL https://github.com/sustcsonglin/](https://github.com/sustcsonglin/flash-linear-attention)
[flash-linear-attention. original-date: 2023-12-20T06:50:18Z.](https://github.com/sustcsonglin/flash-linear-attention)

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention
transformers with hardware-efficient training. In Proceedings of the 41st International Conference
on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 56501–
[56523. PMLR, 2024a. URL https://proceedings.mlr.press/v235/yang24ab.html.](https://proceedings.mlr.press/v235/yang24ab.html)

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. NeurIPS, 2024b.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.),
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing,
pp. 2369–2380, Brussels, Belgium, 2018. Association for Computational Linguistics. doi:
[10.18653/v1/D18-1259. URL https://aclanthology.org/D18-1259.](https://aclanthology.org/D18-1259)

Luca Zancato, Arjun Seshadri, Yonatan Dukler, Aditya Golatkar, Yantao Shen, Benjamin Bowman,
Matthew Trager, Alessandro Achille, and Stefano Soatto. B’MOJO: Hybrid state space realizations of foundation models with eidetic and fading memory. In The Thirty-eighth Annual Confer[ence on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?](https://openreview.net/forum?id=RnQdRY1h5v)
[id=RnQdRY1h5v.](https://openreview.net/forum?id=RnQdRY1h5v)

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a
machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.),
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp.
4791–4800, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/
[P19-1472. URL https://aclanthology.org/P19-1472.](https://aclanthology.org/P19-1472)

Michael Zhang, Simran Arora, Rahul Chalamala, Benjamin Frederick Spector, Alan Wu, Krithik
Ramesh, Aaryan Singhal, and Christopher Re. LoLCATs: On low-rank linearizing of large language models. In The Thirteenth International Conference on Learning Representations, 2025.
[URL https://openreview.net/forum?id=8VtGeyJyx9.](https://openreview.net/forum?id=8VtGeyJyx9)

Yu Zhang, Songlin Yang, Ruijie Zhu, Yue Zhang, Leyang Cui, Yiqiao Wang, Bolun Wang, Freda Shi,
Bailin Wang, Wei Bi, Peng Zhou, and Guohong Fu. Gated slot attention for efficient linear-time
[sequence modeling. 2024. URL https://api.semanticscholar.org/CorpusID:272593079.](https://api.semanticscholar.org/CorpusID:272593079)

19

Published as a conference paper at ICLR 2025

Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan
Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir Radev. QMSum: A new
benchmark for query-based multi-domain meeting summarization. In Kristina Toutanova, Anna
Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell,
Tanmoy Chakraborty, and Yichao Zhou (eds.), Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, pp. 5905–5921, Online, 2021. Association for Computational Linguistics. doi:
[10.18653/v1/2021.naacl-main.472. URL https://aclanthology.org/2021.naacl-main.472.](https://aclanthology.org/2021.naacl-main.472)

20

Published as a conference paper at ICLR 2025

A E XTENDED WY REPRESENTATION FOR GATED DELTA RULE

To reduce notation clutter, we only consider the first chunk here.

For S t, the extended WY representation is

�

�

t−1
�

i=1

S t =

t
�

i=1

γ t

γ i

u i k i [⊺] [,] u t = β t

v t −

γ t

γ i

u i k i [T] [k] [t]

We proof this by mathmetical induction.

Proof.
S t+1 = S t �

α t+1 (I − β t+1 k t+1 k t [⊺] +1 [)] � + β t+1 v t+1 k t [⊺] +1

t

= α t+1 (�

i=1

γ t

γ i

γ t

γ i

t

u i k i [⊺] [)][ −] [α] [t][+1] [β] [t][+1] [(] �

u i k i [⊺] [k] [i] [k] t [⊺] +1 [) +][ β] [t][+1] [v] [t][+1] [k] t [⊺] +1

�

�

t
�

i=1

t
�

i=1

γ t+1

u i k i [T] [k] [t][+1]

γ i

k t [⊺] +1

=

=

=

t
�

i=1

t
�

i=1

t+1
�

i=1

γ t+1

γ i

γ t+1

γ i

γ t+1

γ i

u i k i [⊺] [+][ β] [t][+1]

u i k i [⊺] [+][ γ] [t][+1]

γ t+1
����
1

u i k i [⊺]

v t+1 −

� �� �

u t+1

u t+1 k t [⊺] +1

B E XPERIMENT C ONTUNUED

B.1 E VALUATION

Commonsense reasoning Following Gu & Dao (2023), we evaluate our model on multiple commonsense reasoning benchmarks: PIQA (Bisk et al., 2020), HellaSwag (Hella.; Zellers et al., 2019),
WinoGrande (Wino.; Sakaguchi et al., 2020), ARC-easy (ARC-e) and ARC-challenge (ARC-c)
(Clark et al., 2018), SIQA (Sap et al., 2019), BoolQ (Clark et al., 2019), Wikitext (Wiki.; Merity
et al., 2017), and LAMBADA (LMB.; Paperno et al., 2016).

In-context retrieval Our evaluation comprises both synthetic and real-world tasks. For synthetic
tasks, we utilize the Needle-In-A-Haystack Single (NIAH-S) benchmark suite from RULER (Hsieh
et al., 2024), which includes three increasingly complex tasks: S-NIAH-1 (passkey retrieval), SNIAH-2 (numerical needle in haystack), and S-NIAH-3 (word-based needle in haystack). For realworld tasks, following Arora et al. (2024b), we evaluate on diverse datasets: SWDE (Lockard et al.,
2019) for structured HTML relation extraction, FDA (Arora et al., 2023b) for PDF key-value retrieval, and several question-answering datasets including SQuAD (Rajpurkar et al., 2018), TriviaQA (Joshi et al., 2017a), Drop (Dua et al., 2019), and NQ (Kwiatkowski et al., 2019). Since our
pretrained models lack instruction tuning, we employ the Cloze Completion Formatting prompts
provided by Arora et al. (2024b), which better align with our models’ next-word-prediction training
objective.

Long context understanding We evaluate on 14 tasks from Longbench (Bai et al., 2023), encompassing: narrative comprehension (Narrative QA (Koˇciský et al., 2018)), scientific understanding
(QasperQA (Dasigi et al., 2021)), multi-hop reasoning (MultiField QA, HotpotQA (Yang et al.,
2018), 2WikiMulti QA (Ho et al., 2020), Musique (Trivedi et al., 2022)), document summarization
(GovReport (Huang et al., 2021), QMSum (Zhong et al., 2021), MultiNews (Fabbri et al., 2019)),
and various specialized tasks (TRec (Li & Roth, 2002), Trivia QA (Joshi et al., 2017b), SamSum
(Gliwa et al., 2019), LCC (Guo et al., 2023), and RepoBench-P (Liu et al., 2023)).

21

Published as a conference paper at ICLR 2025

Table S.1: Ablation study on the Gated DeltaNet block. Avg-PPL and Avg-Acc denote average perplexity and
zero-shot commonsense reasoning accuracy (as in Table 3), respectively. All models have 400M parameters
and are trained for 15B tokens on the same subset of FineWeb-Edu dataset (Penedo et al., 2024).

Gated DeltaNet Ablations (400M) Avg-PPL (↓) Avg-Acc (↑)

Gated DeltaNet w Head Dim 128, 27.35 47.26

Macro Design
w. naive Delta Rule 30.87 45.12

w/o. Short Conv 28.95 46.16

w/o. Output Gate 29.12 45.46
w/o. Output Norm 27.55 47.07

Normalization & Feature Map
w. L 1 -norm & ReLU 30.79 45.92

w. L 1 -norm & 1+ELU 30.34 46.05

w. L 1 -norm & SiLU 30.18 46.09

w. L 2 -norm & ReLU 27.67 46.94

w. L 2 -norm & 1+ELU 27.58 47.17

Model Dimensions

w. Head Dim 64 28.31 46.35

w. Head Dim 256 27.13 47.38

B.2 A BLATION S TUDY

Model Wiki. LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c SIQA BoolQ Avg.
ppl ↓ ppl ↓ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑

Hybrid Ablations (500M/15B)

Gated DeltaNet + SWA + Mamba2 24.02 28.20 34.77 67.08 40.84 50.74 60.35 28.83 38.94 61.49 47.88

Gated DeltaNet + Mamba2 + SWA 23.69 26.83 36.17 67.51 41.51 51.85 61.19 29.77 38.58 53.73 47.54

Mamba2 + SWA + Gated DeltaNet 24.14 25.21 36.79 64.96 41.18 52.01 60.90 30.03 38.07 59.44 47.92

Mamba2 + Gated DeltaNet + SWA 23.54 24.11 36.92 66.48 41.70 52.72 61.06 30.54 39.91 60.51 48.73

Table S.2: Ablation studies of Gated DeltaNet models. All evaluations are performed by using
lm-evaluation-harness (Gao et al., 2021). All models use the Llama tokenizer and are trained on the same
subset of the FineWeb-Edu dataset (Penedo et al., 2024).

Table S.1 presents ablation studies on the Gated DeltaNet block’s components. Our experiments
demonstrate that both the short convolution and output gate are crucial for model performance, while
output normalization yields marginal improvements. Consistent with Yang et al. (2024b), we found
L2 normalization to be essential for optimal performance, though the choice of feature map was
less influential. Nevertheless, SiLU consistently outperformed other activation functions, aligning
with observations from Qin et al. (2023a). Through empirical analysis, we determined that a head
dimension of 128 provides an optimal trade-off between performance and computational efficiency.
Additionally, Table S.2 demonstrates that among various hybrid architectures, the combination of
Mamba2, Gated DeltaNet, and SWA in this specific order produces superior results.

22

