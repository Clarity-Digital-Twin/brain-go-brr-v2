## PySeizure: A single machine learning classifier framework to detect seizures in diverse datasets

Bart�lomiej Chybowski [1,2,3*], Shima Abdullateef [2], Hollan Haule [3],
Alfredo Gonzalez-Sulser [1,4], Javier Escudero [1,3]


1 Muir Maxwell Epilepsy Centre, University of Edinburgh, Edinburgh,
Scotland.
2 School of Medicine, Deanery of Clinical Sciences, University of
Edinburgh, 50 Little France Crescent, Edinburgh, EH16 4TJ, Scotland.
3 School of Engineering, Institute for Imaging, Data and
Communications, University of Edinburgh, Alexander Graham Bell
Building, Thomas Bayes Road, Edinburgh, EH9 3FG, Scotland.
4 School of Medicine, Centre for Discovery Brain Sciences, University of
Edinburgh, 1 George Square, Edinburgh, EH8 9JZ,Scotland.


*Corresponding author(s). E-mail(s): b.s.chybowski@sms.ed.ac.uk;
Contributing authors: sabdull2@exseed.ed.ac.uk; s1607398@ed.ac.uk;

agonzal2@exseed.ed.ac.uk; javier.escudero@ed.ac.uk;


**Abstract**


Reliable seizure detection is critical for diagnosing and managing epilepsy, yet
clinical workflows remain dependent on time-consuming manual EEG interpretation. While machine learning has shown promise, existing approaches often
rely on dataset-specific optimisations, limiting their real-world applicability and
reproducibility. Here, we introduce an innovative, open-source machine-learning
framework that enables robust and generalisable seizure detection across various
clinical datasets. We evaluate our approach on two publicly available independent
EEG datasets that differ in patient populations and electrode configurations. To
enhance robustness, the framework incorporates an automated pre-processing
pipeline to standardise data and a majority voting mechanism, in which multiple
models independently assess each second of EEG before reaching a final decision. We train, tune, and evaluate models within each dataset, assessing their
cross-dataset transferability.
Our models achieve high within-dataset performance (AUC 0.904 _**±**_ 0.059 for
CHB-MIT, 0.864 _**±**_ 0.060 for TUSZ) and strong cross-dataset generalisation


1


despite differing EEG setups (AUC 0.615 _**±**_ 0.039 and 0.762 _**±**_ 0.175). Mild postprocessing further improved both within- and cross-dataset results. These
findings highlight the framework’s potential for deployment in diverse clinical
environments. By ensuring complete reproducibility, our framework provides
a foundation for robust, dataset-agnostic seizure detection that complements
clinical expertise.


**Keywords:** Electroencephalography, EEG, Epilepsy, Seizure, Machine Learning, Deep
Learning

### **1 Main**


Epilepsy is a chronic neurological disorder characterised by recurrent, unprovoked
seizures, affecting approximately 51.7 million individuals globally in 2021 [1]. It is one
of the most prevalent neurological disorders worldwide, significantly impacting quality
of life by affecting physical health, cognitive abilities, and social engagement [2]. This
highlights the urgent need for effective management and treatment strategies [3].
Accurate seizure identification is crucial for managing and diagnosing epilepsy.
Currently, the clinical gold standard relies on the visual inspection of electroencephalography (EEG) recordings by neurophysiologists [4]. Although this method is
highly accurate, it is labour-intensive, time-consuming, and prone to human variability [5–7]. Moreover, limited access to trained specialists, particularly in low-resource
settings, exacerbates diagnostic delays and care inequalities [8–10].
Several studies have recently attempted patient-independent and cross-subject
seizure detection. Ali _et al._ [11] employed the Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) dataset to
address key challenges in seizure detection, including class imbalance and intersubject
variability. Their approach used a random forest (RF) classifier with 5-second windows, combined with event-level post-processing. The training data were balanced,
while the test data remained unaltered. However, they evaluated the seizure detection performance exclusively within the dataset. Antonoudiou _et al._ [12] proposed the
SeizyML framework, initially tested on rodent EEG data and later on CHB-MIT.
Nonetheless, a large number of recordings and seizures were manually excluded based
on duration and amplitude thresholds. Zhao _et al._ [13] evaluated their method separately on CHB-MIT and Temple University Hospital EEG Seizure Corpus (TUSZ)
using models trained within each dataset. Despite the strong reported performance
(sensitivity of 0.774 and specificity of 0.763), this study was subjected to several limitations, including the use of a small, manually selected subset of patients with long
seizures and differences in input dimensions and configuration settings between the
datasets. Abou-Abbas _et al._ [14] evaluated a method on TUSZ only, using a specific montage configuration. While the results were robust (sensitivity of 0.767 and
specificity of 0.955), no external dataset was used for validation. Peh _et al._ [15] evaluated models across six datasets using different window sizes. For a 3-second window,
they reported strong within-dataset performance (sensitivity of 0.847 and specificity


2


of 0.751). They also presented cross-dataset results, where models trained on TUSZ
are evaluated among others on CHB-MIT. These findings highlight a trade-off: longer
windows improve accuracy but reduce sensitivity to fine-grained temporal patterns.
These diverse approaches reflect ongoing innovation in seizure detection, focusing on enhancing real-life applicability. The integration of patient-specific models
and event-based cross-validation continues to enhance predictive accuracy, advancing
the potential of these technologies in managing neurological disorders and improving
patient care quality [16].
Nevertheless, despite significant progress in machine learning (ML)-driven medical seizure identification, clinical adoption remains limited due to several key issues.
Many ML models for seizure detection (SeiD) were developed and evaluated on
single, often small datasets, which are further reduced by selecting only the most
favourable examples [11]. This process made them susceptible to overfitting to specific
recording conditions, feature distributions, or patient populations, ultimately limiting
their real-world performance [17, 18] and rendering them unable to generalise effectively. Furthermore, a lack of reproducibility and interpretability – due to proprietary
datasets, non-transparent preprocessing steps, or reliance on dataset-specific manual
optimisations – prevents effective clinical integration. The absence of standardised
evaluation across datasets further hinders adoption, contributing to the persistent gap
between research and real-world implementation [19, 20].
To address current limitations in seizure detection, we introduce **PySeizure**, a
modular and clinically oriented ML framework designed to support epileptic seizure
detection, offering broad applicability across EEG datasets. In contrast to previous
methods that rely on dataset-specific tuning, curated subsets, or extensive manual
preprocessing, PySeizure operates with minimal human intervention and prioritises
generalisation, reproducibility, and ease of integration into diverse workflows. It
standardises preprocessing, automates feature extraction, and evaluates performance
across structurally diverse datasets. The modular architecture supports seamless integration of state-of-the-art ML models while maintaining interoperability. Rather than
proposing a single model, this study presents the complete detection pipeline as a flexible, scalable solution adaptable to various EEG systems. We demonstrate a high level
of accuracy across datasets using a generalisable pipeline without any dataset-specific
manual data curation and tuning. PySeizure comprises four main modules. _(1) Stan-_
_dardised preprocessing_ includes common filtering steps, optional bipolar re-referencing,
and resampling to a uniform frequency, ensuring compatibility across heterogeneous
EEG datasets without requiring dataset-specific modifications. Artefact-affected segments are automatically marked rather than excluded, enabling transparent and
configurable quality control. This module also supports data augmentation strategies
to improve model robustness without requiring additional data collection. ( _2) Epoch_
_segmentation and feature extraction_ support both raw signal- and feature-based models. The feature extraction module computes nearly 40 features per channel, covering
temporal, frequency-based, connectivity, and graph-theory-derived domains. _(3) Model_
_selection and optimisation_ are fully configurable, allowing users to choose from simple classifiers to complex architectures depending on their specific research or clinical
needs. _(4) Seizure detection on an epoch-wise basis_ uses an ensemble of seven models,


3


combined through a majority voting mechanism to improve classification robustness.
This modular design enhances adaptability across datasets and supports the development of clinically viable ML models. By ensuring transparency in preprocessing,
feature selection, and evaluation, our work aligns with best practices in explainable artificial intelligence (XAI) [21], essential for regulatory approval and real-world
deployment.

### **2 Results**


We evaluate the performance of PySeizure on two key tasks: within-dataset and crossdataset seizure detection.

We employed three-fold cross-validation with subject-dependent data splits, ensuring that all recordings from a given patient were assigned to a single set, thereby
enhancing the reliability of our results. To improve temporal resolution, we segmented
the data into 1-second epochs, allowing for more fine-grained predictions. We trained
and evaluated seven models of varying complexity and capability, namely Logistic
regression (LR), XGBoost (XGB), Multilayer perceptron (MLP), Convolutional neural network (CNN), Compact convolutional network for EEG-based brain-computer
interfaces (EEGNet), Convolutional neural network with long short-term memory
(ConvLSTM), and Convolutional neural network with a self-attention-based transformer classification head (ConvTransformer). Additionally, we implemented a voting
mechanism that utilises the predictions of these models to further enhance predictive

accuracy.
In the following subsections, we first analyse the models’ ability to accurately
detect seizures within single datasets, assessing their overall performance using a
range of metrics. We then evaluate the models’ generalisation ability across different datasets, exploring their robustness in handling configurational variations across
EEG recordings. To preserve clarity, we provide results without any post-processing.
Detailed performance metrics, including area under the receiver operating characteristic curve (ROC AUC), sensitivity, specificity, and other relevant measures, are
provided in the supplementary materials (Supplementary Section B), with accompanying visualisations in the main text to highlight key findings, as well as in the
supplementary materials (Supplementary Section A). Finally, we illustrate the impact
of straightforward post-processing on the results.


**2.1 Within-dataset results**


We assessed PySeizure on single datasets to evaluate its seizure detection performance.
The implemented models demonstrated high accuracy in identifying seizure events
within each dataset. These outcomes underscore the model’s capacity to differentiate
seizures from non-seizure activity across various recording conditions reliably. Figure
1 presents the comparison of ROC AUC scores across all models for both datasets.
The remaining metrics are detailed in Supplementary Section A. Average performances of selected metrics in individual datasets are provided in Table 1 and Table 2,
alongside comprehensive results in Supplementary Section B


4


![](markdown/PYSEIZURE/PYSEIZURE.pdf-4-0.png)

**Fig. 1** : Comparison of area under the receiver operating characteristic curve (ROC
AUC) scores for Children’s Hospital Boston - Massachusetts Institute of Technology
Scalp EEG Database (CHB-MIT) and Temple University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models, including results for binary and mean
voting approaches. Dots indicate the scores for individual folds.


**Table 1** : Average performance of proposed models on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset with standard deviation (STD),
including results for binary and mean voting approaches.


**ROC** **Accuracy** **Sensitivity** **Specificity**


**LR** 0.7132 _±_ 0.0502 0.6517 _±_ 0.0386 0.6517 _±_ 0.0386 0.6574 _±_ 0.0404

**XGB** 0.7602 _±_ 0.0415 0.8180 _±_ 0.0330 0.8180 _±_ 0.0330 0.7250 _±_ 0.0084

**MLP** 0.7532 _±_ 0.1048 0.7282 _±_ 0.0727 0.7282 _±_ 0.0727 0.6866 _±_ 0.0911

**CNN** 0.7952 _±_ 0.0668 0.7908 _±_ 0.0454 0.7908 _±_ 0.0454 0.6923 _±_ 0.0832

**EEGNet** 0.7067 _±_ 0.0675 0.6775 _±_ 0.1760 0.6775 _±_ 0.1760 0.5683 _±_ 0.0478

**ConvLSTM** 0.8594 _±_ 0.0445 0.8549 _±_ 0.0342 0.8549 _±_ 0.0342 0.7365 _±_ 0.0486

**ConvTransformer** 0.8265 _±_ 0.0594 0.8287 _±_ 0.0352 0.8287 _±_ 0.0352 0.6749 _±_ 0.0344

**Binary voting** 0.7313 _±_ 0.0420 0.8418 _±_ 0.0464 0.8418 _±_ 0.0464 0.7313 _±_ 0.0420
**Mean voting** 0.8638 _±_ 0.0603 0.8531 _±_ 0.0425 0.8531 _±_ 0.0425 0.7413 _±_ 0.0378


**2.2 Cross-dataset results**


To assess the generalisation capability of PySeizure, we tested the framework with all
models across multiple datasets with diverse configurational characteristics. Despite
variations in recording conditions and patient populations, the models exhibited strong
performance, consistently achieving comparable seizure detection accuracy across


5


**Table 2** : Average performance of proposed models on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHBMIT) dataset with standard deviation (STD), including results for binary and
mean voting approaches.


**ROC** **Accuracy** **Sensitivity** **Specificity**


**LR** 0.8675 _±_ 0.0913 0.8163 _±_ 0.0907 0.8163 _±_ 0.0907 0.7990 _±_ 0.0471

**XGB** 0.8769 _±_ 0.0395 0.9695 _±_ 0.0114 0.9695 _±_ 0.0114 0.8605 _±_ 0.0627

**MLP** 0.9266 _±_ 0.0536 0.8906 _±_ 0.0260 0.8906 _±_ 0.0260 0.8574 _±_ 0.0640

**CNN** 0.8861 _±_ 0.0574 0.8025 _±_ 0.2322 0.8025 _±_ 0.2322 0.7655 _±_ 0.0789

**EEGNet** 0.8059 _±_ 0.0654 0.9023 _±_ 0.1018 0.9023 _±_ 0.1018 0.7245 _±_ 0.0660

**ConvLSTM** 0.9067 _±_ 0.0414 0.9286 _±_ 0.0694 0.9286 _±_ 0.0694 0.7570 _±_ 0.0984

**ConvTransformer** 0.7848 _±_ 0.0557 0.8306 _±_ 0.1691 0.8306 _±_ 0.1691 0.6823 _±_ 0.0149

**Binary voting** 0.5829 _±_ 0.0530 0.9684 _±_ 0.0235 0.9684 _±_ 0.0235 0.5829 _±_ 0.0530
**Mean voting** 0.9044 _±_ 0.0586 0.9752 _±_ 0.0142 0.9752 _±_ 0.0142 0.6309 _±_ 0.0998


datasets. Specifically, results for models trained on CHB-MIT and evaluated on TUSZ
are presented in Table 3, while models trained on TUSZ and evaluated on CHBMIT are shown in Table 4. Figure 2 presents the comparison of ROC AUC scores
across all models. The remaining metrics are detailed in Supplementary Section A and
Supplementary Section B

This evaluation underscores the framework’s ability to adapt to various data
sources and manage variations without extensive retraining or post-processing.


**Table 3** : The average performance of proposed models trained on Children’s
Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database
(CHB-MIT) and tested on Temple University Hospital EEG Seizure Corpus
(TUSZ) dataset, including results for binary and mean voting approaches.


**ROC** **Accuracy** **Sensitivity** **Specificity**


**LR** 0.5895 _±_ 0.0074 0.6590 _±_ 0.0791 0.6590 _±_ 0.0791 0.5511 _±_ 0.0129

**XGB** 0.5510 _±_ 0.0417 0.7660 _±_ 0.0267 0.7660 _±_ 0.0267 0.5231 _±_ 0.0080

**MLP** 0.6171 _±_ 0.0088 0.6930 _±_ 0.0457 0.6930 _±_ 0.0457 0.5648 _±_ 0.0162

**CNN** 0.6059 _±_ 0.0837 0.6704 _±_ 0.0835 0.6704 _±_ 0.0835 0.5383 _±_ 0.0287

**EEGNet** 0.5125 _±_ 0.0426 0.7609 _±_ 0.0307 0.7609 _±_ 0.0307 0.5015 _±_ 0.0057

**ConvLSTM** 0.6157 _±_ 0.0681 0.7118 _±_ 0.0690 0.7118 _±_ 0.0690 0.5372 _±_ 0.0475

**ConvTransformer** 0.5660 _±_ 0.0729 0.6233 _±_ 0.1209 0.6233 _±_ 0.1209 0.5021 _±_ 0.0026

**Binary voting** 0.5257 _±_ 0.0200 0.7552 _±_ 0.0256 0.7552 _±_ 0.0256 0.5257 _±_ 0.0200
**Mean voting** 0.6148 _±_ 0.0398 0.7562 _±_ 0.0274 0.7562 _±_ 0.0274 0.5244 _±_ 0.0200


**2.3 Feature importance**


Using the Boruta feature eliminator [22], one of the two methods available in our
framework, we selected the best-performing features and conducted Shapley additive
explanations (SHAP) analysis to identify the most important features for models using


6


![](markdown/PYSEIZURE/PYSEIZURE.pdf-6-0.png)

**Fig. 2** : Comparison of area under the receiver operating characteristic curve (ROC
AUC) scores for all the models trained on the Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and evaluated
on the Temple University Hospital EEG Seizure Corpus (TUSZ) datasets, including
results for binary and mean voting approaches. In the legend, the arrow symbol ( _→_ )
denotes that models were trained on the dataset indicated before the arrow and eval
uated on the dataset indicated after the arrow.


**Table 4** : The average performance of proposed models trained on Temple
University Hospital EEG Seizure Corpus (TUSZ) and tested on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset, including results for binary and mean voting approaches.


**ROC** **Accuracy** **Sensitivity** **Specificity**


**LR** 0.6610 _±_ 0.1712 0.6002 _±_ 0.2003 0.6002 _±_ 0.2003 0.6102 _±_ 0.1527

**XGB** 0.5358 _±_ 0.1573 0.5778 _±_ 0.1870 0.5778 _±_ 0.1870 0.5501 _±_ 0.1414

**MLP** 0.6734 _±_ 0.1541 0.6503 _±_ 0.1292 0.6503 _±_ 0.1292 0.6226 _±_ 0.1393

**CNN** 0.5502 _±_ 0.1515 0.5704 _±_ 0.1176 0.5704 _±_ 0.1176 0.5427 _±_ 0.0608

**EEGNet** 0.4630 _±_ 0.1839 0.3480 _±_ 0.2644 0.3480 _±_ 0.2644 0.4208 _±_ 0.0978

**ConvLSTM** 0.5656 _±_ 0.1188 0.6402 _±_ 0.1017 0.6402 _±_ 0.1017 0.5531 _±_ 0.0788

**ConvTransformer** 0.6186 _±_ 0.0870 0.7310 _±_ 0.0259 0.7310 _±_ 0.0259 0.5932 _±_ 0.0717

**Binary voting** 0.5478 _±_ 0.0903 0.5964 _±_ 0.1088 0.5964 _±_ 0.1088 0.5519 _±_ 0.0868
**Mean voting** 0.6438 _±_ 0.1673 0.5614 _±_ 0.1551 0.5614 _±_ 0.1551 0.5800 _±_ 0.1276


these features for seizure detection, as shown in Figure 3. For models using raw data,


7


we analysed the channels with the greatest impact (Figure 4), where impact refers to
the ability to predict whether the analysed segment indicates a seizure.
The SHAP analysis provided insight into the relative importance of individual features by quantifying their contribution to the model’s prediction. The most influential
variables included a range of features related to power and energy across different
frequency bands, such as _Power Spectral Density_, _Power Spectral Centroid_, or _Total_
_Signal Energy_ . Temporal features such as _Coastline_, _Zero Crossing_, or _Signal To Noise_
defining the shape of the signal and the amount of noise in it, as well as connectivity properties such as the _Phase Slope Index_, _Coherence_, _Imaginary Coherence_, and
_cross-correlation maximum coefficient_, also contributed substantially.
For models trained on unprocessed EEG data, the channel-level SHAP analysis (Figure 4) revealed that frontal-central and temporal channels had the greatest
impact on predictions. Detailed SHAP analyses of individual models across datasets
are provided in Supplementary Section A.


**2.4 Post-processing**


Thus far, we have reported results without any post-processing to provide a clear
view of PySeizure’s baseline performance – that is, all results were reported considering an epoch assessment. Here, we apply a mild post-processing (Section 4.5), which
combines epoch-based sampling (EPOCH) and any-overlap method (OVLP) methods [23] to evaluation cross-validation folds. Table 5 reports the average improvement
for each metric, along with p-values from the Wilcoxon signed-rank test [24], corrected for multiple comparisons using the false discovery rate (FDR) method via the
Benjamini-Hochberg procedure [25]. The results are also visualised in Figure 5, which
additionally shows the effect size calculated using Cliff’s Delta [26]. These results
indicate statistically significant improvements across datasets and experimental variations, including models trained and evaluated on the TUSZ and CHB-MIT datasets,
as well as cross-dataset evaluations (i.e., trained on CHB-MIT and tested on TUSZ,
and vice versa). Detailed results for individual datasets are provided in Supplementary Table B2. Overall, the analysis shows that most post-processing improvements
are statistically significant, underscoring the potential of post-processing to enhance
model performance and reliability.

### **3 Discussion**


**3.1 Clinical relevance and real-world impact**


Reliable seizure detection is critical for both diagnosis and long-term patient management, yet current clinical workflows rely heavily on manual EEG classification, a
time-intensive and highly variable process. The proposed framework addresses these
challenges by providing an automated and generalisable solution designed for realworld clinical deployment. Unlike traditional approaches that require dataset-specific
tuning [27], this framework is inherently cross-compatible, standardising EEG processing across diverse datasets by automatically handling variations in sampling rates,


8


(a) (b)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-8-0.png)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-8-1.png)


(c) (d)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-8-2.png)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-8-3.png)


**Fig. 3** : Global feature importance derived from SHAP values, showing the top twenty
most influential features across all models using engineered features within a) the
Temple University Hospital EEG Seizure Corpus (TUSZ), b) the Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset, and cross-dataset evaluation: c) trained on TUSZ and evaluated on CHBMIT, and d) trained on CHB-MIT and evaluated on TUSZ. Feature importance was
aggregated over all evaluation folds to reflect consistent patterns across diverse clinical
settings.


9


(a) (b)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-9-0.png)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-9-1.png)


(c) (d)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-9-2.png)

![](markdown/PYSEIZURE/PYSEIZURE.pdf-9-3.png)


**Fig. 4** : Channel importance using SHAP values for models using unprocessed data
within a) the Temple University Hospital EEG Seizure Corpus (TUSZ), b) the Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database
(CHB-MIT) dataset, and cross-dataset evaluation: c) trained on TUSZ and evaluated
on CHB-MIT, and d) trained on CHB-MIT and evaluated on TUSZ.


referencing schemes, and signal artefacts. This adaptability will facilitate future seamless integration into different clinical environments, research studies, and ambulatory


10


**Table 5** : The average improvement in the metric after postprocessing, with the p-value indicating whether the difference is
statistically significant. Asterisks denote significance level: * _p <_
0 _._ 05; ** _p <_ 0 _._ 01; *** _p <_ 0 _._ 001. Values are aggregated for each
metric across all folds and all models. The arrow symbol ( _→_ )
denotes that models were trained on the dataset indicated before

the arrow and evaluated on the dataset indicated after the arrow.


**Dataset** **Mean improvement** **p-value**



**ROC**


**Accuracy**


**Sensitivity**


**Specificity**



CHB-MIT _→_ CHB-MIT 7 _._ 30 _×_ 10 _[−]_ [4] 8.70e-02
CHB-MIT _→_ TUSZ _−_ 1 _._ 27 _×_ 10 _[−]_ [3] 3.92e-01
TUSZ _→_ CHB-MIT _−_ 3 _._ 63 _×_ 10 _[−]_ [3] 8.56e-01
TUSZ _→_ TUSZ _−_ 7 _._ 63 _×_ 10 _[−]_ [3] 4.08e-02


CHB-MIT _→_ CHB-MIT 2 _._ 30 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ 1 _._ 59 _×_ 10 _[−]_ [2] 6.47e-05 _[∗∗∗]_

TUSZ _→_ CHB-MIT 4 _._ 22 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ 1 _._ 89 _×_ 10 _[−]_ [2] 3.86e-05 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 2 _._ 30 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ 1 _._ 59 _×_ 10 _[−]_ [2] 6.47e-05 _[∗∗∗]_

TUSZ _→_ CHB-MIT 4 _._ 22 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ 1 _._ 89 _×_ 10 _[−]_ [2] 3.86e-05 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 9 _._ 66 _×_ 10 _[−]_ [3] 7.20e-02
CHB-MIT _→_ TUSZ 2 _._ 65 _×_ 10 _[−]_ [3] 2.62e-01
TUSZ _→_ CHB-MIT 1 _._ 30 _×_ 10 _[−]_ [2] 2.39e-03 _[∗∗]_

TUSZ _→_ TUSZ 4 _._ 67 _×_ 10 _[−]_ [3] 1.22e-01



monitoring systems, supporting broader clinical adoption of artificial intelligence
(AI)-driven seizure detection.
A key advantage of this framework is its flexibility. Users can configure preprocessing parameters, artefact handling, feature extraction, and model selection to
suit specific clinical or research objectives. For instance, instead of excluding noisy
epochs outright, the system allows them to be marked, enabling users to make informed
decisions based on data quality rather than applying rigid exclusion criteria. Another
example is the framework’s support for both raw signal-based and feature-based models, allowing integration into a wide range of analytical workflows. In this work, we
evaluated models spanning the spectrum from simple classifiers, such as logistic regression – which is unlikely to fully capture the complexity of EEG signals – to deep
learning architectures designed to learn directly from raw data. This range was chosen
deliberately to demonstrate the versatility of the framework across different modelling
paradigms. This adaptability aligns with precision medicine initiatives by supporting individualised analyses and enabling rapid, data-driven clinical decision-making.
While our experiments report results for a single configuration, many of the framework’s parameters – such as the epoch length or downsampling – are user-adjustable.
We established default parameters empirically, for example, a 1-second epoch length
to maximise temporal resolution and improve the granularity of predictions. The sampling frequency of 256 Hz was chosen to ensure consistency across datasets, as it was


11


![](markdown/PYSEIZURE/PYSEIZURE.pdf-11-0.png)

**Fig. 5** : The average improvement on the metric after post-processing. The top plot
shows the average improvement and indicator of statistical significance after multiple
tests correction using the False discovery rate (FDR) test (Benjamini/Hochberg). The
bottom plot shows the effect size calculated using Cliff’s Delta. In the legend, the
arrow symbol ( _→_ ) denotes that models were trained on the dataset indicated before
the arrow and evaluated on the dataset indicated after the arrow.


the lowest common frequency available. These choices are not fixed, and users may
tailor them to better suit their specific data or application needs.
In addition to evaluating the current version on publicly available datasets, we
also tested early versions of the framework [28] using clinical data [29]. These findings further underscore the potential impact and real-world applicability of our
approach in clinical environments. Our results already demonstrate strong performance across diverse datasets, even when using general-purpose models. This suggests
that PySeizure provides a robust foundation for seizure detection. We hypothesise
that incorporating state-of-the-art architectures specifically tailored to EEG or seizure
detection may further enhance performance, particularly in clinical applications. Automated seizure monitoring could enable earlier intervention, reduce misdiagnoses, and
support continuous assessment of treatment efficacy [30]. Furthermore, by structuring
EEG data in a standardised format, the framework facilitates large-scale, multi-centre
validation studies – a critical step towards regulatory approval and clinical integration. By bridging the gap between AI research and real-world neurology practice, this
framework paves the way for more scalable, efficient, and accessible seizure detection
in clinical care. Moreover, the framework aligns with the broader trend of AI systems
designed to support, rather than replace, clinical decision-making.


12


**3.2 Alignment with trends in AI and healthcare**


The proposed framework aligns with the growing trend of leveraging AI to improve
healthcare outcomes, particularly in the domain of neurology. As healthcare increasingly adopts AI-driven tools for diagnostics and decision-making, the need for reliable,
generalisable, and scalable solutions becomes paramount. Our framework directly
addresses this need by offering an adaptable, cross-dataset solution for seizure detection that can seamlessly integrate into diverse clinical environments, ensuring that AI
models can be reliably applied across multiple hospital systems, patient populations,
and EEG recording configurations.
A key trend in AI healthcare applications is the emphasis on generalisability, allowing AI models to be effective not only on the data they were trained on but also across
different datasets [31]. The framework’s ability to handle varying signal quality, electrode configurations, and artefacts across datasets ensures its robustness in real-world
clinical scenarios, positioning it as a strong candidate for widespread adoption. This
emphasis on cross-dataset generalisation is crucial for AI solutions that need to function in a variety of clinical settings, as datasets are often heterogeneous, especially in
multi-centre studies and global healthcare systems.
Our results demonstrate strong generalisation across heterogeneous datasets using
a fixed, short 1-second window and minimal pre-processing. In contrast, several prior
studies rely on dataset-specific manual adjustments, hand-picked examples or larger
window sizes. Here, we compare our work with studies mentioned in the Main Section
(Section 1), emphasising the differences between solutions. Ali _et al._ [11] report sensitivities of 0.726 (subject-wise 5-fold) and 0.753 (Leave One Out) using a RF classifier
on 5-second segments from CHB-MIT, with event-level post-processing. The training
data are balanced while the test data remain unmodified. Their results are limited to
CHB-MIT only. Antonoudiou _et al._ [12] evaluate four classical classifiers – Gaussian
Na¨ıve Bayes (GNB), Decision tree (DT), Stochastic gradient descent (SGD) classifier, and Passive Aggressive Classifier (PAC) – on CHB-MIT after excluding seizures
shorter than 30 seconds or with low amplitude. This filtering removes 66 recordings and
99 seizures, leaving 86 seizures across 24 subjects. Their best F1 score for CHB-MIT is
approximately 0.2, and they do not evaluate on other EEG datasets. The code is publicly available. Zhao _et al._ [13] train and test on balanced data from manually selected
patients – 9 from CHB-MIT and 12 from TUSZ – choosing cases with longer seizures.
Their within-dataset performance is high, reporting accuracy: 0.767, specificity: 0.763,
sensitivity: 0.774, AUC: 0.826, and F1: 0.761. However, they do not report any crossdataset evaluations, and the work is not publicly available. Abou-Abbas _et al._ [14]
focus exclusively on TUSZ with an Averaged Reference (AR) montage. They report
accuracy: 0.917, recall: 0.767, precision: 0.808, and specificity: 0.955. Their evaluation
is limited to a single dataset and montage. The code is not available publicly. Peh _et_
_al._ [15] present a study across six datasets using their CNN-transformer with belief
matching loss (CNN-TRF-BM) model. They demonstrate that performance improves
with larger windows, peaking at a window size of 20 seconds. However, this comes at
the cost of temporal granularity. For a 3-second window – their shortest evaluated and
closest to PySeizure’s 1-second setup – the CNN-TRF-BM model achieves accuracy:
0.823, sensitivity: 0.885, specificity: 0.616, and F1: 0.824 on TUSZ; and accuracy: 0.833,


13


sensitivity: 08080, specificity: 0.886, and F1: 0.837 on CHB-MIT. When trained on
TUSZ and tested on CHB-MIT, the performance drops to accuracy: 0.584, sensitivity:
0.365, specificity: 0.959, and F1: 0.547. The implementation is not publicly available.
These findings highlight the trade-off between performance and temporal resolution:
while longer windows yield higher accuracy, they may reduce the model’s ability to
capture fine-grained temporal patterns, which PySeizure targets more directly using
1-second windows, although with slightly lower overall metrics.


**3.3 Limitations and challenges**


We acknowledge several limitations and challenges in our work. A key limitation is
the reliance on publicly available datasets, which may not fully capture the diversity of clinical data. To address this, future efforts should include validation with
hospital-acquired EEG data to ensure robustness in diverse clinical settings. Incorporating advanced artefact suppression techniques could further mitigate noise impacts,
enhancing model reliability.
Another challenge is the handling of artefacts and noisy epochs. The framework
automatically marks artefact-affected segments rather than discarding them, allowing
users to make informed decisions about data inclusion. However, this approach requires
careful consideration, as excessive noise may still impact model performance. Future
improvements could incorporate advanced artefact suppression techniques or adaptive
weighting strategies to mitigate the influence of low-quality recordings.
Finally, the computational demands of feature extraction and the model ensemble
present an additional challenge. While the system is optimised for scalability, training
and deploying multiple models require substantial computational resources. Running
the complete PySeizure pipeline on large datasets such as TUSZ or CHB-MIT requires
several days on a high-performance computing cluster [32], primarily due to timeconsuming steps such as data preprocessing, feature extraction, and model training.
While inference is fast, especially for short recordings, comprehensive evaluations with
SHAP analysis can still take hours for entire datasets. This limits deployment in
real-time or resource-constrained settings. To address this, techniques such as model
pruning or knowledge distillation could improve efficiency with minimal performance
loss. Future work may also focus on reducing ensemble size by selecting either the
highest-performing models or those meeting predefined performance thresholds.

### **4 Methods**


In this section, we detail the methodologies employed in our study, beginning with
a presentation of the datasets (Section 4.1). We then provide an overview of the
architecture (Section 4.2), followed by a detailed description of the pre-processing
steps (Section 4.3), models (Section 4.4) and post-processing (Section 4.5. For complete transparency and to enable replication, the code is available in the GitHub
repository [33].


14


**4.1 Datasets**


In this section, we will discuss the datasets chosen for this study: the TUSZ and
CHB-MIT. These datasets have been selected due to their large size and variability,
offering both intra- and inter-dataset diversity. The rich variability found within each
dataset, alongside the differences between them, provides a comprehensive framework
for evaluating the robustness and generalisability of algorithms. Datasets used in this
study are publicly available and were accessed in accordance with their respective data
use agreements. Ethical approval for this work was obtained from the University of
Edinburgh School of Engineering.


**4.1.1 Temple University Hospital EEG Seizure Corpus**


The Temple University Hospital EEG Seizure Corpus (TUSZ) is a comprehensive
dataset widely utilised in seizure detection research, comprised of a large collection
of annotated EEG recordings [34]. The dataset includes 1,493 EEG sessions from 613
patients, with a total of 2,981 seizure events. It covers eight distinct seizure types,
with expert-verified annotations detailing the precise onset and offset of each event.
EEG recordings are sampled at a minimum rate of 250 Hz per second. Annotations
are available in two formats: per channel, which provides event details for individual EEG channels, and for all channels, offering an aggregated view of the events.
Additionally, the dataset includes recordings with one of three reference point types:
linked ears, average reference, or an alternative version of the average reference. TUSZ
is continually updated, with ongoing improvements to annotations that enhance its
clinical relevance. Reflecting real-world clinical conditions, the dataset incorporates
inter-patient variability and variations in seizure manifestations. For this study, we
conducted experiments using version 2.0.0 of the dataset.


**4.1.2 Children’s Hospital Boston - Massachusetts Institute of**
**Technology Scalp EEG Database**


The Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset, developed by the Massachusetts Institute of Technology (MIT) and Children’s Hospital Boston (CHB), is a widely recognised resource for
seizure detection and prediction research [35]. It contains 915 hours of EEG recordings from 23 paediatric subjects, with a total of 198 episodes, including 84 seizures
across 5 seizure types. The dataset adopts the International 10-20 System’s bipolar
montage method, capturing EEG signals from 22 electrodes at a 256 Hz sampling rate.
In some cases, recordings are made with 18 channels. Each EEG recording file typically lasts for one hour, with most subjects having between 9 and 42 consecutive EEG
files. The dataset includes annotations specifying the precise onset and offset times for
each seizure event. Additionally, data from _CHB01_ and _CHB21_ were collected from
the same patient, 1.5 years apart, providing an opportunity for studying the evolution of seizures over time. CHB-MIT’s expert-verified annotations, detailing seizure
events across all channels. The dataset reflects real-world clinical conditions, with
inter-patient variability and challenges in detecting seizures in paediatric populations.


15


**4.2 Architecture overview**

![](markdown/PYSEIZURE/PYSEIZURE.pdf-15-0.png)


**Fig. 6** : Diagram of the modules and components of the framework. a) Data reader
module responsible for reading the data from European Data Format (EDF) files and
extracting the annotation either from external files or annotations embedded in data
files; b) Pre-processing module liable for filtering and marking common artefacts as
well as for resampling, re-referencing, and augmenting data; c) Feature engineering
module responsible for calculating features and preparing final shape of the data; d)
Model optimisation and training module managing the optimisation and training of
the models along with their evaluation.


Figure 6 provides an overview of the architecture, divided into modules that will
be described in more detail in the following subsections.
The proposed framework processes EEG data from EDF files through a structured pipeline designed for robust seizure detection. The preprocessing stage includes
automatic frequency filtering to reduce artefacts, re-referencing signals to a bipolar
montage, resampling to a predefined (in configuration, by user, default value is 256 Hz)
frequency, and automatically marking epochs that are considered noisy or artefactcontaminated, allowing the user to decide whether to include them in training (by
default excluded from training). The data is then segmented into epochs, serving as
the basis for further analysis.
To improve the generalisability of the model, data augmentation is applied using
sign flipping, time reversal [36], and their combination, effectively quadrupling available data. For models employing feature-based learning, a feature extraction step
derives nearly 40 unique features per channel, encompassing temporal and frequencydomain characteristics, inter-channel connectivity, and graph-theoretic properties. The
full list of available features is presented in the Supplementary Table B3.
A feature selection step follows, ensuring that only the most relevant features are
retained for classification. The framework offers two configurable options: the Boruta


16


feature elimination algorithm [22], which identifies all relevant features by comparing
them to randomised shadow features using a random forest classifier, and CrossValidated Support Vector Machine Recursive Feature Elimination (SVM-RFECV),
which recursively removes the least informative features based on their weights in a
linear support vector machines (SVM) model [37]. The default framework’s feature
selector is Boruta, as it exhibits a higher stability in feature selection compared to
recursive feature elimination (RFE) [38]. Furthermore, Boruta, with the number of
iterations set at 20, has been faster in comparison to SVM-RFECV with 3 crossvalidation rounds for both tested datasets. The results presented in the paper were
computed with Boruta for the reasons mentioned earlier.
Hyperparameter tuning is individually performed for each of the seven models to
optimise their configurations, using the Optuna [39] framework. This approach ensures
that each model is precisely tailored for optimal performance.
During the evaluation, each model independently classifies every second of the EEG
data, determining whether the segment indicates a seizure. We also implemented a
majority voting mechanism, in which each model’s prediction contributes to the final
decision.

The framework addresses class imbalance in two ways. Users can optionally provide custom class weights to rebalance the loss function during training. However,
by default, the framework selects all seizure epochs and randomly samples an equal
number of background epochs for training, feature selection, and hyperparameter tuning. This balanced subset ensures robust model development. For final evaluation,
all available epochs (typically highly imbalanced) are used without shuffling, thereby
preserving the original temporal order of the recordings.
The framework is designed to be highly flexible and scalable, allowing researchers
to customise input data, parameters, feature sets, and processing steps to suit different
datasets and clinical requirements, enhancing adaptability across various applications.


**4.3 Preprocessing**


To ensure consistency and quality in EEG signal processing, which is crucial for
the analysis of such signals [40], the framework implements a structured, automated
pre-processing pipeline. Initially, EEG data are imported from EDF files with accompanying annotations, which are standardised into a structured table based on the
predefined configuration. This supports both generalised and per-channel annotations, while allowing for the optional inclusion of artefact markers, facilitating flexible
downstream analysis.
To improve signal integrity and enhance cross-dataset compatibility, the framework applies automatic filtering to remove common artefacts, including power line
interference at 50 and 60 Hz and the Hanning window finite impulse response (FIR)
dual high-band filter of 0.6 Hz. We also propose a smoothing mechanism that detects
unnaturally high amplitudes – defined as those exceeding _±_ 5 times the standard deviation (STD) of the median amplitude – thereby addressing the problem of changes
in signal gain (Figure 7). We also mark sections of the recording where individual
signals are extremely similar (cosine similarity above 0.95) as artefacts. The signals


17


are then resampled to a uniform frequency (default: 256 Hz), ensuring comparability across recordings acquired at different sampling rates. If the data are recorded in
a unipolar montage, an optional re-referencing step converts them to a bipolar configuration, reducing inter-electrode variability and aligning the signal representation
across datasets. These standardisation steps are critical for enabling robust model
performance across datasets.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-17-0.png)


**Fig. 7** : Threshold-based smoothing of the signal. The top plot shows the raw, unprocessed signal. The middle plot shows the amplitude envelope with empirically derived
thresholds. The bottom plot shows the processed signal where out-of-threshold amplitudes are normalised to the median value, preserving in-range dynamics.


The data are then segmented into non-overlapping epochs (default: 1 second),
which serve as the fundamental units for analysis. The default duration of 1 second
reflects a practical compromise – short enough to limit background activity in brief
seizures, yet sufficiently long to capture ictal features – while remaining adaptable
to other epoch lengths depending on clinical or computational requirements. Each
epoch is assessed for artefacts using a combined slope-based approach, which identifies segments with exceptionally steep slopes [41] and includes our implementation of
flatlining detection to capture signal loss or amplifier saturation (i.e., when the signal plateaus due to disconnection or hardware limits). Rather than excluding noisy
epochs, the framework flags them, allowing researchers to determine their inclusion
based on study requirements.


18


The framework enables two analytical workflows. In the first case, raw EEG epochs
are provided for models that leverage direct signal analysis. In the second case, a comprehensive feature extraction process derives nearly 40 unique features per channel.
These features capture temporal and frequency characteristics, inter-channel connectivity, and properties derived from graph theory [42–46]. We present the full list of
features in the Supplementary Section B. This approach ensures the retention of clinically and computationally relevant information, enhancing the interpretability of the
models.

All processed epochs, whether in raw or feature-extracted form, are stored as structured records in an SQLite database, facilitating efficient retrieval, reproducibility, and
integration into large-scale clinical studies.


**4.4 Models**


Here, we present the models implemented within our framework. All deep learning
models incorporating normalisation layers utilise Batch Normalisation, which was
selected as the default during development. Additionally, Leaky ReLU activation
functions and Kaiming normal initialisation were employed where appropriate to
enhance model convergence and stability [47]. Hyperparameter optimisation is performed independently for each cross-validation fold and each dataset as part of the
processing pipeline. As a result, we do not use a fixed architecture. Instead, we define
an optimisation search space for each model and report the range of selected values
across folds to guide future uses of our pipeline in similar problems. The complete
search space is presented in the Supplementary Table B1


**Logistic regression** is a fundamental statistical model used for binary classification tasks [48]. It estimates the probability of an outcome by applying the logistic
function to a linear combination of input features. An architecture diagram of this
model is presented in Supplementary Figure A1a). The most common parameters
across all folds were: learning rate of 0.01, Adam optimiser, weight decay of 0.01 and
0.00001, and batch size of 64.


**XGBoost** is a powerful ensemble learning algorithm based on gradient boosting
decision trees [49]. It employs advanced techniques such as regularisation, tree pruning, and parallel computation to enhance performance and mitigate over-fitting. An
architecture diagram of this model is presented in the Supplementary Figure A1 b).
The most common parameters across all folds were: learning rate of 0.03, number of
estimators of 300 and 400, number of parallel trees of 900, max delta step of 0.1668,
gamma of 0.0 or 0.0002, lambda of 0.0215, minimum child weight of 278 and 1668,
subsample of 0.55, and colsample by tree of 0.4481 and 0.5123, and batch size of 160
and 32. Other parameters were non-conclusive.


**Multilayer perceptron** is an artificial neural network composed of multiple
layers of interconnected neurons [50]. It utilises nonlinear activation functions and
back-propagation for training, enabling it to capture complex patterns in data. An
architecture diagram of this model is presented in the Supplementary Figure A1 c).


19


The most common parameters across all folds were: learning rate of 0.0001, Adam
optimiser, weight decay of 0.001 and 0.0000001, batch size of 64 and 128, and hidden
dimensions 1, 2, and 3 of 512, 256, 128 or 64, respectively.

**Convolutional neural network** are deep learning models designed to process
spatially structured data, particularly images [51]. They employ convolutional layers
to extract hierarchical features, followed by pooling layers to reduce dimensionality
and fully connected layers for classification. An architecture diagram of this model
is presented in the Supplementary Figure A2. The most common parameters across
all folds were: learning rate of 0.00001, Adam optimiser, weight decay of 0.0000001,
batch size of 128, the first convolution layer of 512 and kernel of 4, max pool of 2, the
second convolution layer of 256 with kernel of 3, the third convolution later of 128
and 64 with kernel of 3, and the fully connected layer of 128.


**EEGNet** is a compact and efficient deep learning architecture tailored for EEG
signal analysis [52]. It incorporates depthwise and separable convolutions to capture
both spatial and temporal features while maintaining low computational complexity.
EEGNet has demonstrated strong performance in brain-computer interfaces (BCI)
applications and EEG-based classification tasks, offering a balance between accuracy
and model efficiency. An architecture diagram of this model is presented in the Supplementary Figure A3. This model’s hyperparameters are defined by the authors and
therefore are not tunable.


**ConvLSTM** model integrates CNN with long short-term memory (LSTM) networks to leverage both spatial and temporal dependencies in sequential data [53].
CNNs extract high-level features, which are subsequently processed by LSTMs to
capture long-term dependencies. This hybrid approach is particularly effective for
time series analysis, including medical signal classification and speech recognition [54].
An architecture diagram of this model is presented in the Supplementary Figure A4.
The most common parameters across all folds were: Adam optimiser, weight decay of
0.0001, batch size of 64, the first convolution layer of 512 (kernel size inconclusive),
max pool of 2, the second convolution layer of 256 with kernel of 5, the third convolution later of 128 and 64 with kernel of 5, the number of LSTM layers of 2, the hidden
dimension of LSTM of 128, the first fully connected layer of 128, and the second fully
connected layer of 64 and 32. The value of the learning rate was inconclusive.


**ConvTransformer** model combines the feature extraction capabilities of CNNs
with the self-attention mechanism of transformers [55]. CNNs encode local spatial patterns, while the transformer component captures long-range dependencies,
enhancing the model’s ability to process complex sequential data [56]. An architecture diagram of this model is presented in the Supplementary Figure A5. The most
common parameters across all folds were: Adam optimiser, weight decay of 0.000001,
batch size of 32, the learning rate of 0.0001 and 0.00001, the first convolution layer
of 1024 with kernel size of 3, max pool of 2, the second convolution layer of 256 and
128 with kernel of 3 and 4, the third convolution later of 64 with kernel of 5, the
vocabulary size of 5500, the feed forward layer of 1024, number of heads of 6, number


20


of encoder layers of 4 and 5, and model dimension of 300.


**4.5 Post-processing**


Our framework incorporates a post-processing module designed to make minimal
adjustments, targeting single-epoch artefacts such as drift, isolated epochs, or gaps
in consecutive series. This method emulates human correction by addressing minor
misalignments and clear errors without significantly altering the predictions, which
comprise a mixture of EPOCH and OVLP [23]. We continue to analyse the recording
at high granularity, treating each epoch as a standalone signature rather than considering the entire event as a single unit. However, we permit minimal misalignment or
gaps where prediction overlap is high. For example, if a single-epoch disruption has a
predicted value of 0.23 but the true label is 1 and the neighbouring epochs are confidently positive, the post-processing module may raise it to 0.51. This results in a
correct binary classification without overstating the model’s certainty, thus preserving metric integrity. The goal is to improve temporal consistency while ensuring that
adjusted values remain close to the decision threshold, reflecting a nuanced correction rather than artificial performance inflation. Figure 8 illustrates these subtle but
meaningful refinements. Importantly, the module remains optional and is disabled by
default to maintain analytical transparency.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-20-0.png)


**Fig. 8** : Diagram illustrating the effect of post-processing on a signal. The figure shows
three signals: the true signal (y true), the predicted signal (y predicted), and the postprocessed result (y post-processed). In the post-processed signal, highlighted regions
show the effect of post-processing on an isolated epoch, a drift, and a gap. Labels “1”
and “0” denote an event and background, respectively.


These refinements, illustrated in Figure 8, more closely reflect human judgment and
support a nuanced enhancement of the overall analysis. The post-processing module
operates on an epoch-wise basis, enabling highly granular classification. While this
granularity can lead to an increased number of false alarms, it supports more precise
estimates of sensitivity and specificity than event-based post-processing methods [23].


21


### **5 Conclusion**

In this work, we introduced **PySeizure**, a machine learning-based framework for
automated epileptic seizure detection. Our models achieved high within-dataset
performance (area under the curve (AUC) 0.904 _±_ 0.059 and 0.864 _±_ 0.060) and demonstrated strong generalisation across datasets, despite differences in experimental set-up
and patient populations, with AUC values of 0.762 _±_ 0.175 and 0.615 _±_ 0.039. These
results were obtained without any post-processing, highlighting the robustness and
adaptability of PySeizure in varying clinical settings. The framework’s design prioritises generalisability, reproducibility, and ease of integration into existing workflows,
addressing key challenges faced by current automated systems and offering a scalable solution for epilepsy management. To further enhance performance, we applied
mild post-processing based on a combination of EPOCH and OVLP strategies [23],
resulting in improved AUC scores of 0.913 _±_ 0.064 and 0.867 _±_ 0.058 within-dataset and
0.768 _±_ 0.172 and 0.619 _±_ 0.036 cross-datasets. Notably, PySeizure employs a votingbased model ensemble, justified by the observed occasional lack of agreement between
individual models. In complex classification settings such as seizure detection – where
subtle patterns in EEG can lead models to diverge in their predictions – this disagreement is not a weakness, but rather an opportunity: aggregating predictions through
a voting mechanism allows the system to exploit complementary strengths of individual models, improving overall robustness and reducing the risk of overfitting to
dataset-specific noise.
Future directions include extending PySeizure’s evaluation with hospital-acquired
data to enhance its clinical applicability. We also plan to expand testing across more
diverse datasets. This will provide further insights into the system’s performance in
real-world clinical environments. Future work will also explore real-time performance
improvements and the potential integration with wearable devices for continuous
monitoring. As PySeizure progresses, it holds promise to bridge the gap between stateof-the-art AI research and tangible clinical applications, driving advancements in the
management of epilepsy and potentially other neurological disorders.


22


### **Declarations**


- Funding
_Not applicable_

- Conflict of interest/Competing interests
_Not applicable_

- Ethics approval and consent to participate
_Datasets used in this study are publicly available and were accessed in accor-_
_dance with their respective data use agreements. Ethical approval for this work was_
_obtained from the University of Edinburgh School of Engineering._

- Consent for publication
_Not applicable_

- Data availability
_Datasets used in this study are publicly available and were accessed in accordance_
_with their respective data use agreements._


–
[TUH EEG Seizure Detection Corpus](https://isip.piconepress.com/projects/tuh_eeg/)

–
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)


- Materials availability
_Not applicable_

- Code availability
_The_ _code_ _for_ _PySeizure_ _framework_ _is_ _publicly_ _available_ _at_
_[https://github.com/bartlomiej-chybowski/PySeizure](https://github.com/bartlomiej-chybowski/PySeizure)_

- Author contribution

_B.C.: Conceptualisation, Methodology, Software, Validation, Formal analysis,_
_Investigation, Writing – original draft. S.A., H.H.: Methodology, Validation, Writ-_
_ing – review & editing. A.G., J.E.: Supervision, Validation, Methodology, Project_
_administration, Writing – review & editing. All authors reviewed and approved the_
_final manuscript._

### **Acknowledgements**


For the purpose of open access, the author has applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising from this
submission.


23


### **References**


[1] Feigin, V. L. _et al._ Global, regional, and national burden of epilepsy, 1990–2021: a
systematic analysis for the Global Burden of Disease Study 2021. _The Lancet Pub-_
_lic Health_ **0** [(2025). URL https://www.thelancet.com/journals/lanpub/article/](https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(24)00302-5/fulltext)
[PIIS2468-2667(24)00302-5/fulltext. Publisher: Elsevier.](https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(24)00302-5/fulltext)


[2] Reilly, C. _et al._ Factors associated with quality of life in active childhood epilepsy:
A population-based study. _European Journal of Paediatric Neurology_ **19**, 308–
[313 (2015). URL https://www.ejpn-journal.com/article/S1090-3798(15)00006-9/](https://www.ejpn-journal.com/article/S1090-3798(15)00006-9/fulltext)
[fulltext. Publisher: Elsevier.](https://www.ejpn-journal.com/article/S1090-3798(15)00006-9/fulltext)


[3] Meisel, C. _et al._ Machine learning from wristband sensor data for wearable, noninvasive seizure forecasting. _Epilepsia_ **61**, 2653–2666 (2020).
URL [https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.16719.](https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.16719) ~~e~~ print:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/epi.16719.


[4] Anuragi, A., Singh Sisodia, D. & Pachori, R. B. Epileptic-seizure classification
using phase-space representation of FBSE-EWT based EEG sub-band signals and
ensemble learners. _Biomedical Signal Processing and Control_ **71**, 103138 (2022).
[URL https://www.sciencedirect.com/science/article/pii/S1746809421007357.](https://www.sciencedirect.com/science/article/pii/S1746809421007357)


[5] van Donselaar, C. A., Stroink, H., Arts, W.-F. & Dutch Study Group of Epilepsy
in Childhood. How confident are we of the diagnosis of epilepsy? _Epilepsia_ **47**
**Suppl 1**, 9–13 (2006).


[6] Zaidi, A., Clough, P., Cooper, P., Scheepers, B. & Fitzpatrick, A. P. Misdiagnosis
of epilepsy: many seizure-like attacks have a cardiovascular cause. _J Am Coll_
_Cardiol_ **36**, 181–184 (2000).


[7] Buettner, R., Frick, J. & Rieg, T. High-performance detection of epilepsy
in seizure-free EEG recordings: A novel machine learning approach using very
specific epileptic EEG sub-bands .


[8] Maloney, E. M., Corcoran, P., Costello, D. J. & O’Reilly, E. J. Association between social deprivation and incidence of first seizures and
epilepsy: A prospective population-based cohort. _Epilepsia_ **63**, 2108–2119
[(2022). URL https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.17313.](https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.17313) ~~e~~ print:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/epi.17313.


[9] Steer, S., Pickrell, W. O., Kerr, M. P. & Thomas, R. H. Epilepsy prevalence and socioeconomic deprivation in England. _Epilepsia_ **55**, 1634–1641
[(2014). URL https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.12763.](https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.12763) ~~e~~ print:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/epi.12763.


[10] Thomas, R. H. _et al._ 056 Variability in adult epilepsy prevalence in the UK. _J Neu-_
_rol Neurosurg Psychiatry_ **83** [, e1–e1 (2012). URL https://jnnp.bmj.com/content/](https://jnnp.bmj.com/content/83/3/e1.222)


[24](https://jnnp.bmj.com/content/83/3/e1.222)


[83/3/e1.222. Publisher: BMJ Publishing Group Ltd Section: ABN abstracts.](https://jnnp.bmj.com/content/83/3/e1.222)


[11] Ali, E., Angelova, M. & Karmakar, C. Epileptic seizure detection using CHBMIT dataset: The overlooked perspectives. _R Soc Open Sci_ **11**, 230601. URL
[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11286169/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11286169/)


[12] Antonoudiou, P., Basu, T. & Maguire, J. SeizyML: An Application for
Semi-Automated Seizure Detection Using Interpretable Machine Learning Models. _Neuroinform_ **23**, 23 (2025). [URL https://link.springer.com/10.1007/](https://link.springer.com/10.1007/s12021-025-09719-4)
[s12021-025-09719-4.](https://link.springer.com/10.1007/s12021-025-09719-4)


[13] Zhao, Y. _et al._ Multi-view cross-subject seizure detection with information bottleneck attribution. _J. Neural Eng._ **19** [, 046011 (2022). URL https://dx.doi.org/](https://dx.doi.org/10.1088/1741-2552/ac7d0d)
[10.1088/1741-2552/ac7d0d. Publisher: IOP Publishing.](https://dx.doi.org/10.1088/1741-2552/ac7d0d)


[14] Abou-Abbas, L., Henni, K., Jemal, I. & Mezghani, N. Generative
AI with WGAN-GP for boosting seizure detection accuracy. _Front._
_Artif. Intell._ **7**, 1437315 (2024). [URL https://www.frontiersin.org/journals/](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1437315/full)
[artificial-intelligence/articles/10.3389/frai.2024.1437315/full.](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1437315/full) Publisher: Frontiers.


[15] Peh, W. Y. _et al._ Six-Center Assessment of CNN-Transformer with Belief Matching Loss for Patient-Independent Seizure Detection in EEG. _Int J Neural Syst_
**33**, 2350012 (2023).


[16] Ren, Z., Han, X. & Wang, B. The performance evaluation of the state-of-the-art
EEG-based seizure prediction models. _Front Neurol_ **13**, 1016224 (2022). URL
[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9732735/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9732735/)


[17] Bradshaw, T. J., Huemann, Z., Hu, J. & Rahmim, A. A Guide to Cross-Validation
for Artificial Intelligence in Medical Imaging. _Radiol Artif Intell_ **5**, e220232
[(2023). URL https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10388213/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10388213/)


[18] Steyerberg, E. W., Bleeker, S. E., Moll, H. A., Grobbee, D. E. & Moons, K. G. M.
Internal and external validation of predictive models: A simulation study of bias
and precision in small samples. _Journal of Clinical Epidemiology_ **56**, 441–447
[(2003). URL https://www.jclinepi.com/article/S0895-4356(03)00047-7/fulltext.](https://www.jclinepi.com/article/S0895-4356(03)00047-7/fulltext)
Publisher: Elsevier.


[19] Markowetz, F. All models are wrong and yours are useless: making clinical prediction models impactful for patients. _npj Precis. Onc._ **8**, 1–3 (2024).
[URL https://www.nature.com/articles/s41698-024-00553-6.](https://www.nature.com/articles/s41698-024-00553-6) Publisher: Nature
Publishing Group.


[20] Aggarwal, R. _et al._ Diagnostic accuracy of deep learning in medical imaging: a
systematic review and meta-analysis. _npj Digit. Med._ **4** [, 1–23 (2021). URL https:](https://www.nature.com/articles/s41746-021-00438-z)
[//www.nature.com/articles/s41746-021-00438-z.](https://www.nature.com/articles/s41746-021-00438-z) Publisher: Nature Publishing


25


Group.


[21] Chaddad, A., Peng, J., Xu, J. & Bouridane, A. Survey of Explainable AI Techniques in Healthcare. _Sensors (Basel)_ **23** [, 634 (2023). URL https://www.ncbi.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9862413/)
[nlm.nih.gov/pmc/articles/PMC9862413/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9862413/)


[22] Kursa, M. B. & Rudnicki, W. R. Feature Selection with the **Boruta** Package. _J._
_Stat. Soft._ **36** [(2010). URL http://www.jstatsoft.org/v36/i11/.](http://www.jstatsoft.org/v36/i11/)


[23] Shah, V., Golmohammadi, M., Obeid, I. & Picone, J. in _Objective Evaluation_
_Metrics for Automatic Classification of EEG Events_ (eds Obeid, I., Selesnick, I. &
Picone, J.) _Biomedical Signal Processing_ 223–255 (Springer International Publish[ing, Cham, 2021). URL https://link.springer.com/10.1007/978-3-030-67494-6](https://link.springer.com/10.1007/978-3-030-67494-6_8) ~~8~~ .


[24] Wilcoxon, F. Individual Comparisons by Ranking Methods. _Biometrics Bul-_
_letin_ **1** [, 80–83 (1945). URL https://www.jstor.org/stable/3001968. Publisher:](https://www.jstor.org/stable/3001968)

[International Biometric Society, Wiley].


[25] Benjamini, Y. & Hochberg, Y. Controlling the False Discovery Rate: A Practical
and Powerful Approach to Multiple Testing. _Journal of the Royal Statistical Soci-_
_ety Series B: Statistical Methodology_ **57** [, 289–300 (1995). URL https://academic.](https://academic.oup.com/jrsssb/article/57/1/289/7035855)
[oup.com/jrsssb/article/57/1/289/7035855.](https://academic.oup.com/jrsssb/article/57/1/289/7035855) Publisher: Oxford University Press
(OUP).


[26] Macbeth, G., Razumiejczyk, E. & Ledesma, R. D. Cliff ´ s Delta Calculator: A
non-parametric effect size program for two groups of observations. _Universitas_
_Psychologica_ **10** [, 545–555 (2011). URL https://revistas.javeriana.edu.co/index.](https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643)
[php/revPsycho/article/view/643.](https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643)


[27] Probst, P., Boulesteix, A.-L. & Bischl, B. Tunability: Importance of Hyperparameters of Machine Learning Algorithms .


[28] Chybowski, B., Gonzalez-Sulser, A. & Escudero, J. Can a single machine learning
classifier pipeline detect seizures in two different patient datasets? _Epilepsia_
**64** [, 7–570 (2023). URL https://onlinelibrary.wiley.com/doi/10.1111/epi.17787.](https://onlinelibrary.wiley.com/doi/10.1111/epi.17787)
Abstract 1604.


[29] Pentimalli Biscaretti Di Ruffia, F. _et al._ Abstracts. _Epilepsia_ **65**, 1–508 (2024).
[URL https://onlinelibrary.wiley.com/doi/10.1111/epi.18151. Abstract 591.](https://onlinelibrary.wiley.com/doi/10.1111/epi.18151)


[30] Rai, P. _et al._ Automated analysis and detection of epileptic seizures
in video recordings using artificial intelligence. _Front. Neuroinform._ **18**
[(2024). URL https://www.frontiersin.org/journals/neuroinformatics/articles/10.](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2024.1324981/full)
[3389/fninf.2024.1324981/full. Publisher: Frontiers.](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2024.1324981/full)


[31] Yang, J. _et al._ Generalizability assessment of AI models across hospitals in a
low-middle and high income country. _Nat Commun_ **15** [, 8270 (2024). URL https:](https://www.nature.com/articles/s41467-024-52618-6)


[26](https://www.nature.com/articles/s41467-024-52618-6)


[//www.nature.com/articles/s41467-024-52618-6.](https://www.nature.com/articles/s41467-024-52618-6) Publisher: Nature Publishing
Group.


[[32] Edinburgh Compute and Data Facility (2024). URL https://information-services.](https://information-services.ed.ac.uk/research-support/research-computing/ecdf)
[ed.ac.uk/research-support/research-computing/ecdf.](https://information-services.ed.ac.uk/research-support/research-computing/ecdf)


[33] Chybowski, B. Pyseizure (2025). URL [https://github.com/](https://github.com/bartlomiej-chybowski/PySeizure)
[bartlomiej-chybowski/PySeizure.](https://github.com/bartlomiej-chybowski/PySeizure)


[34] Obeid, I. & Picone, J. The Temple University Hospital EEG Data Corpus. _Front_
_Neurosci_ **10**, 196 (2016). [URL https://www.ncbi.nlm.nih.gov/pmc/articles/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4865520/)
[PMC4865520/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4865520/)


[35] Shoeb, A. H. _Application of machine learning to epileptic seizure onset detection_
_and treatment_ [. Thesis, Massachusetts Institute of Technology (2009). URL https:](https://dspace.mit.edu/handle/1721.1/54669)
[//dspace.mit.edu/handle/1721.1/54669. Accepted: 2010-04-28T17:17:43Z.](https://dspace.mit.edu/handle/1721.1/54669)


[36] Abdallah, T., Jrad, N., Abdallah, F., Humeau-Heurtier, A. & Van Bogaert, P.
Cross-Site Generalization for Imbalanced Epileptic Classification 1–5 (2023).


[37] Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. Gene Selection for Cancer
Classification using Support Vector Machines. _Machine Learning_ **46**, 389–422
[(2002). URL https://doi.org/10.1023/A:1012487302797.](https://doi.org/10.1023/A:1012487302797)


[38] Degenhardt, F., Seifert, S. & Szymczak, S. Evaluation of variable selection methods for random forests and omics data sets. _Brief Bioinform_ **20**, 492–503 (2017).
[URL https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6433899/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6433899/)


[39] Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. Optuna: A nextgeneration hyperparameter optimization framework 2623–2631 (2019).


[40] Del Pup, F., Zanola, A., Fabrice Tshimanga, L., Bertoldo, A. & Atzori, M.
The More, the Better? Evaluating the Role of EEG Preprocessing for Deep
Learning Applications. _IEEE Transactions on Neural Systems and Rehabil-_
_itation Engineering_ **33**, 1061–1070 (2025). [URL https://ieeexplore.ieee.org/](https://ieeexplore.ieee.org/document/10909332)
[document/10909332. Conference Name: IEEE Transactions on Neural Systems](https://ieeexplore.ieee.org/document/10909332)
and Rehabilitation Engineering.


[41] Fasol, M. C. M., Escudero, J. & Gonzalez-Sulser, A. Single-Channel EEG Artifact
[Identification with the Spectral Slope 2482–2487 (2023). URL https://ieeexplore.](https://ieeexplore.ieee.org/document/10385840/?arnumber=10385840)
[ieee.org/document/10385840/?arnumber=10385840. ISSN: 2156-1133.](https://ieeexplore.ieee.org/document/10385840/?arnumber=10385840)


[42] Mozafari, M. & Sardouie, S. H. Automatic epileptic seizure detection in a mixed
generalized and focal seizure dataset 172–176 (2019).


[43] Zabihi, M., Kiranyaz, S., Ince, T. & Gabbouj, M. Patient-specific epileptic seizure
detection in long-term EEG recording in paediatric patients with intractable


27


seizures 1–7 (2013).


[44] Guo, Y. _et al._ Epileptic Seizure Detection by Cascading Isolation Forest-Based
Anomaly Screening and EasyEnsemble. _IEEE Transactions on Neural Systems_
_and Rehabilitation Engineering_ **30**, 915–924 (2022). Conference Name: IEEE
Transactions on Neural Systems and Rehabilitation Engineering.


[45] Boonyakitanont, P., Lek-uthai, A., Chomtho, K. & Songsiri, J. A review of feature
extraction and performance evaluation in epileptic seizure detection using EEG.
_Biomedical Signal Processing and Control_ **57** [, 101702 (2020). URL https://www.](https://www.sciencedirect.com/science/article/pii/S1746809419302836)
[sciencedirect.com/science/article/pii/S1746809419302836.](https://www.sciencedirect.com/science/article/pii/S1746809419302836)


[[46] NetworkX — NetworkX documentation. URL https://networkx.org/.](https://networkx.org/)


[47] He, K., Zhang, X., Ren, S. & Sun, J. Delving Deep into Rectifiers: Surpassing
Human-Level Performance on ImageNet Classification 1026–1034 (2015). URL
[https://ieeexplore.ieee.org/document/7410480/. ISSN: 2380-7504.](https://ieeexplore.ieee.org/document/7410480/)


[48] Harris, J. K. Primer on binary logistic regression. _Fam Med Community_
_Health_ **9**, e001290 (2021). [URL https://www.ncbi.nlm.nih.gov/pmc/articles/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8710907/)
[PMC8710907/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8710907/)


[49] Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System 785–794
[(2016). URL https://dl.acm.org/doi/10.1145/2939672.2939785.](https://dl.acm.org/doi/10.1145/2939672.2939785)


[50] 4. Fully Connected Deep Networks - TensorFlow for Deep Learning [Book]. URL

[https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/ch04.html)
[ch04.html.](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/ch04.html)


[51] Lecun, Y., Bottou, L., Bengio, Y. & Haffner, P. Gradient-based learning applied to
document recognition. _Proc. IEEE_ **86** [, 2278–2324 (1998). URL http://ieeexplore.](http://ieeexplore.ieee.org/document/726791/)
[ieee.org/document/726791/.](http://ieeexplore.ieee.org/document/726791/)


[52] Lawhern, V. J. _et al._ EEGNet: a compact convolutional neural network for EEGbased brain–computer interfaces. _J. Neural Eng._ **15** [, 056013 (2018). URL https:](https://dx.doi.org/10.1088/1741-2552/aace8c)
[//dx.doi.org/10.1088/1741-2552/aace8c. Publisher: IOP Publishing.](https://dx.doi.org/10.1088/1741-2552/aace8c)


[53] Liu, X., Jia, J. & Zhang, R. Automatic Detection of Epilepsy EEG based on
[CNN-LSTM Network Combination Model 225–232 (2020). URL https://dl.acm.](https://dl.acm.org/doi/10.1145/3445815.3445852)
[org/doi/10.1145/3445815.3445852.](https://dl.acm.org/doi/10.1145/3445815.3445852)


[54] Zhao, B., Lu, H., Chen, S., Liu, J. & Wu, D. Convolutional neural networks for
time series classification. _Journal of Systems Engineering and Electronics_ **28**,
[162–169 (2017). URL https://ieeexplore.ieee.org/document/7870510/.](https://ieeexplore.ieee.org/document/7870510/)


[55] Li, C. _et al._ EEG-based seizure prediction via Transformer guided CNN. _Measure-_
_ment_ **203** [, 111948 (2022). URL https://www.sciencedirect.com/science/article/](https://www.sciencedirect.com/science/article/pii/S0263224122011447)


[28](https://www.sciencedirect.com/science/article/pii/S0263224122011447)


[pii/S0263224122011447.](https://www.sciencedirect.com/science/article/pii/S0263224122011447)


[56] Bougourzi, F., Dornaika, F., Distante, C. & Taleb-Ahmed, A. D-TrAttUnet:
Toward hybrid CNN-transformer architecture for generic and subtle segmentation
in medical images. _Computers in Biology and Medicine_ **176**, 108590 (2024). URL
[https://www.sciencedirect.com/science/article/pii/S0010482524006759.](https://www.sciencedirect.com/science/article/pii/S0010482524006759)


29


### **Appendix A Figures**

![](markdown/PYSEIZURE/PYSEIZURE.pdf-29-0.png)

**Fig. A1** : Diagram of the a) Logistic regression (LR), b) XGBoost (XGB), c) Multilayer perceptron (MLP) models. The size of the Fully Connected (FC 1-4) layers and
XGBoost parameters are optimised by the hyperparameter optimisation algorithm.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-29-1.png)


**Fig. A2** : Diagram of the Convolutional neural network (CNN) model architecture.
The size of the convolution (Conv 1-3), Max Pool (MP), and fully connected (FC)
layers is optimised by the hyperparameter optimisation algorithm.


30


![](markdown/PYSEIZURE/PYSEIZURE.pdf-30-0.png)

**Fig. A3** : Diagram of the EEGNet model architecture. All parameters are predefined
according to the original implementation.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-30-1.png)


**Fig. A4** : Diagram of the ConvLSTM model architecture. The size of the convolution
(Conv 1-3), Max Pool (MP), LSTM and fully connected (FC 1 and 2) layers is optimised by the hyperparameter optimisation algorithm.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-30-2.png)


**Fig. A5** : Diagram of the ConvTransformer model architecture. The hyperparameter
optimisation algorithm optimises the size of the convolution (Conv 1-3), Max Pool
(MP) layers, and encoder parameters.


31


![](markdown/PYSEIZURE/PYSEIZURE.pdf-31-0.png)

**Fig. A6** : Comparison of Mean square error (MSE) scores for Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
and Temple University Hospital EEG Seizure Corpus (TUSZ) datasets across all the
models.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-31-1.png)


**Fig. A7** : Comparison of Accuracy scores for Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and Temple
University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models.


32


![](markdown/PYSEIZURE/PYSEIZURE.pdf-32-0.png)

**Fig. A8** : Comparison of Precision scores for Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and Temple
University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-32-1.png)


**Fig. A9** : Comparison of F1 scores for Children’s Hospital Boston - Massachusetts
Institute of Technology Scalp EEG Database (CHB-MIT) and Temple University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models.


33


![](markdown/PYSEIZURE/PYSEIZURE.pdf-33-0.png)

**Fig. A10** : Comparison of Area under the precision recall curve (PRAUC) scores
for Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database and Temple University Hospital EEG Seizure Corpus datasets across all the
models.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-33-1.png)


**Fig. A11** : Comparison of Sensitivity scores for Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and Temple
University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models.


34


![](markdown/PYSEIZURE/PYSEIZURE.pdf-34-0.png)

**Fig. A12** : Comparison of Specificity scores for Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and Temple
University Hospital EEG Seizure Corpus (TUSZ) datasets across all the models.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-34-1.png)


**Fig. A13** : Comparison of Mean square error (MSE) scores for all the models trained
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) and evaluated on Temple University Hospital EEG Seizure
Corpus (TUSZ) datasets and vice versa.


35


![](markdown/PYSEIZURE/PYSEIZURE.pdf-35-0.png)

**Fig. A14** : Comparison of Accuracy scores for all the models trained on Children’s
Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHBMIT) and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
datasets and vice versa.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-35-1.png)


**Fig. A15** : Comparison of Precision scores for all the models trained on Children’s
Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHBMIT) and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
datasets and vice versa.


36


![](markdown/PYSEIZURE/PYSEIZURE.pdf-36-0.png)

**Fig. A16** : Comparison of F1 scores for all the models trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) and
evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ) datasets and
vice versa.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-36-1.png)


**Fig. A17** : Comparison of Area under the precision recall curve (PRAUC) scores for
all the models trained on Children’s Hospital Boston - Massachusetts Institute of
Technology Scalp EEG Database (CHB-MIT) and evaluated on Temple University
Hospital EEG Seizure Corpus (TUSZ) datasets and vice versa.


37


![](markdown/PYSEIZURE/PYSEIZURE.pdf-37-0.png)

**Fig. A18** : Comparison of Sensitivity scores for all the models trained on Children’s
Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHBMIT) and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
datasets and vice versa.

![](markdown/PYSEIZURE/PYSEIZURE.pdf-37-1.png)


**Fig. A19** : Comparison of Specificity scores for all the models trained on Children’s
Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database (CHBMIT) and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
datasets and vice versa.


38


![](markdown/PYSEIZURE/PYSEIZURE.pdf-38-0.png)

**Fig. A20** : Threshold-based smoothing of an artificial signal with the threshold level
reduced to 1 standard deviation for illustrative purposes. The top plot shows the raw,
unprocessed signal. The middle plot shows the amplitude envelope with an empirically
derived threshold. The bottom plot shows the processed signal, where out-of-threshold
amplitudes are normalised to the median value, preserving in-range dynamics.


39


![](markdown/PYSEIZURE/PYSEIZURE.pdf-39-0.png)

**Fig. A21** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the LR model trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on TUSZ using engineered
features.


40


![](markdown/PYSEIZURE/PYSEIZURE.pdf-40-0.png)

**Fig. A22** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the LR model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on CHB-MIT using engineered features.


41


![](markdown/PYSEIZURE/PYSEIZURE.pdf-41-0.png)

**Fig. A23** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the LR model trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on Children’s Hospital Boston

- Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT) using engineered features.


42


![](markdown/PYSEIZURE/PYSEIZURE.pdf-42-0.png)

**Fig. A24** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the LR model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
using engineered features.


43


![](markdown/PYSEIZURE/PYSEIZURE.pdf-43-0.png)

**Fig. A25** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the XGB model trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on TUSZ using engineered
features.


44


![](markdown/PYSEIZURE/PYSEIZURE.pdf-44-0.png)

**Fig. A26** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the XGB model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on CHB-MIT using engineered features.


45


![](markdown/PYSEIZURE/PYSEIZURE.pdf-45-0.png)

**Fig. A27** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the XGB model trained on Temple University
Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
using engineered features.


46


![](markdown/PYSEIZURE/PYSEIZURE.pdf-46-0.png)

**Fig. A28** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the XGB model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
using engineered features.


47


![](markdown/PYSEIZURE/PYSEIZURE.pdf-47-0.png)

**Fig. A29** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the MLP model trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on TUSZ using engineered
features.


48


![](markdown/PYSEIZURE/PYSEIZURE.pdf-48-0.png)

**Fig. A30** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the MLP model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on CHB-MIT using engineered features.


49


![](markdown/PYSEIZURE/PYSEIZURE.pdf-49-0.png)

**Fig. A31** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the MLP model trained on Temple University
Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
using engineered features.


50


![](markdown/PYSEIZURE/PYSEIZURE.pdf-50-0.png)

**Fig. A32** : Global feature importance derived from SHAP values, showing the top
twenty most influential features for the MLP model trained on Children’s Hospital
Boston - Massachusetts Institute of Technology Scalp EEG Database (CHB-MIT)
dataset and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ)
using engineered features.


51


![](markdown/PYSEIZURE/PYSEIZURE.pdf-51-0.png)

**Fig. A33** : Channel importance using SHAP values for the CNN model trained on
Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on
TUSZ using unprocessed data.


52


![](markdown/PYSEIZURE/PYSEIZURE.pdf-52-0.png)

**Fig. A34** : Channel importance using SHAP values for the CNN model trained on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database
(CHB-MIT) dataset and evaluated on CHB-MIT using unprocessed data.


53


![](markdown/PYSEIZURE/PYSEIZURE.pdf-53-0.png)

**Fig. A35** : Channel importance using SHAP values for the CNN model trained
on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) using unprocessed data.


54


![](markdown/PYSEIZURE/PYSEIZURE.pdf-54-0.png)

**Fig. A36** : Channel importance using SHAP values for the CNN model trained on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG Database
(CHB-MIT) dataset and evaluated on Temple University Hospital EEG Seizure Corpus (TUSZ) using unprocessed data.


55


![](markdown/PYSEIZURE/PYSEIZURE.pdf-55-0.png)

**Fig. A37** : Channel importance using SHAP values for the EEGNet model trained on
Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on
TUSZ using unprocessed data.


56


![](markdown/PYSEIZURE/PYSEIZURE.pdf-56-0.png)

**Fig. A38** : Channel importance using SHAP values for the EEGNet model trained
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset and evaluated on CHB-MIT using unprocessed data.


57


![](markdown/PYSEIZURE/PYSEIZURE.pdf-57-0.png)

**Fig. A39** : Channel importance using SHAP values for the EEGNet model trained
on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) using unprocessed data.


58


![](markdown/PYSEIZURE/PYSEIZURE.pdf-58-0.png)

**Fig. A40** : Channel importance using SHAP values for the EEGNet model trained
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset and evaluated on Temple University Hospital EEG
Seizure Corpus (TUSZ) using unprocessed data.


59


![](markdown/PYSEIZURE/PYSEIZURE.pdf-59-0.png)

**Fig. A41** : Channel importance using SHAP values for the ConvLSTM model trained
on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated
on TUSZ using unprocessed data.


60


![](markdown/PYSEIZURE/PYSEIZURE.pdf-60-0.png)

**Fig. A42** : Channel importance using SHAP values for the ConvLSTM model trained
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset and evaluated on CHB-MIT using unprocessed data.


61


![](markdown/PYSEIZURE/PYSEIZURE.pdf-61-0.png)

**Fig. A43** : Channel importance using SHAP values for the ConvLSTM model trained
on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) using unprocessed data.


62


![](markdown/PYSEIZURE/PYSEIZURE.pdf-62-0.png)

**Fig. A44** : Channel importance using SHAP values for the ConvLSTM model trained
on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp EEG
Database (CHB-MIT) dataset and evaluated on Temple University Hospital EEG
Seizure Corpus (TUSZ) using unprocessed data.


63


![](markdown/PYSEIZURE/PYSEIZURE.pdf-63-0.png)

**Fig. A45** : Channel importance using SHAP values for the ConvTransformer model
trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on TUSZ using unprocessed data.


64


![](markdown/PYSEIZURE/PYSEIZURE.pdf-64-0.png)

**Fig. A46** : Channel importance using SHAP values for the ConvTransformer model
trained on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp
EEG Database (CHB-MIT) dataset and evaluated on CHB-MIT using unprocessed
data.


65


![](markdown/PYSEIZURE/PYSEIZURE.pdf-65-0.png)

**Fig. A47** : Channel importance using SHAP values for the ConvTransformer model
trained on Temple University Hospital EEG Seizure Corpus (TUSZ) dataset and evaluated on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp
EEG Database (CHB-MIT) using unprocessed data.


66


![](markdown/PYSEIZURE/PYSEIZURE.pdf-66-0.png)

**Fig. A48** : Channel importance using SHAP values for the ConvTransformer model
trained on Children’s Hospital Boston - Massachusetts Institute of Technology Scalp
EEG Database (CHB-MIT) dataset and evaluated on Temple University Hospital
EEG Seizure Corpus (TUSZ) using unprocessed data.


67


### **Appendix B Tables**

**Table B1** : Overview of the hyperparameter search space used during model tuning
for all evaluated classifiers.


**Model** **Parameter** **Search space**



XGB


LR,
MLP,
CNN,
ConvLSTM,
ConvTransformer


CNN,
ConvLSTM,
ConvTransformer


MLP



learning rate 0.001, 0.01, 0.05, 0.1, 0.2, 0.3
batch size 32 - 256 (linear spacing, step: 32)
max depth 0.7 - 1.6 (log-scale, 10 steps)
min child weight 0.5 - 4 (log-scale, 10 steps)
reg alpha 0.001 - 10 (geometric spacing, 10 steps)
reg lambda 0.001 - 10 (geometric spacing, 10 steps)
reg gamma 0.000001 - 0.2 (geometric spacing, 10 steps)
max delta step 0.1 - 10 (geometric spacing, 10 steps)
colsample bytree 0.3 - 10 (geometric spacing, 10 steps)
num parallel tree 100 - 900 (linear spacing, step: 100)
n estimators ranges 200 - 1000 (linear spacing, step: 100)
subsample 0.55, 0.6, 0.70, 0.85


learning rate 10 _[−]_ [6] _,_ 10 _[−]_ [5] _,_ 10 _[−]_ [4] _,_ 10 _[−]_ [3] _,_ 10 _[−]_ [2]

weight decay 10 _[−]_ [7] _,_ 10 _[−]_ [6] _,_ 10 _[−]_ [5] _,_ 10 _[−]_ [4] _,_ 10 _[−]_ [3] _,_ 10 _[−]_ [2] _,_ 10 _[−]_ [1]

batch size 32 - 128 (linear spacing, step: 32)
optimiser “Adam”, “AdamW”, “SGD”


convolution layer 1 2 [8] _,_ 2 [9] _,_ 2 [10]

convolution layer 2 2 [7] _,_ 2 [8]

convolution layer 3 2 [6] _,_ 2 [7]

convolution kernel 2 3, 4, 5
max pool 2


fully-connected layer 1 2 [8] _,_ 2 [8] _,_ 2 [10]

fully-connected layer 2 2 [7] _,_ 2 [8]

fully-connected layer 3 2 [6] _,_ 2 [7]



CNN fully-connected layer 2 [5] _,_ 2 [6] _,_ 2 [7] _,_ 2 [8]


LSTM layer 2 [6] _,_ 2 [7]



ConvLSTM


ConvTransformer



number of LSTM layers 2, 3, 4, 5
fully-connected layer 1 2 [6] _,_ 2 [7]

fully-connected layer 2 2 [5] _,_ 2 [6]


vocabulary size 1300, 2100, 3400, 5500, 8900
model dimension 100, 200, 300, 500, 800
number of heads 2, 3, 4, 5, 6
number of encoder layers 1, 2, 3, 4, 5, 6
fully-connected layer 2 [7] _,_ 2 [8] _,_ 2 [9] _,_ 2 [10]


68


**Table B2** : The average improvement in the metric after postprocessing, with the p-value indicating whether the difference is
statistically significant. Asterisks denote significance level: * _p <_
0 _._ 05; ** _p <_ 0 _._ 01; *** _p <_ 0 _._ 001. Values are aggregated for each
metric across all folds and all models. The arrow symbol ( _→_ )
denotes that models were trained on the dataset indicated before

the arrow and evaluated on the dataset indicated after the arrow.


**Dataset** **Mean improvement** **p-value**


CHB-MIT _→_ CHB-MIT 2 _._ 30 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_



Accuracy


F1


MSE


PRAUC


Precision


ROC


Sensitivity


Specificity



CHB-MIT _→_ TUSZ 1 _._ 59 _×_ 10 _[−]_ [2] 6.47e-05 _[∗∗∗]_

TUSZ _→_ CHB-MIT 4 _._ 22 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ 1 _._ 89 _×_ 10 _[−]_ [2] 3.86e-05 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 3 _._ 33 _×_ 10 _[−]_ [2] 7.83e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ 1 _._ 54 _×_ 10 _[−]_ [3] 9.29e-01
TUSZ _→_ CHB-MIT 2 _._ 25 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ 1 _._ 27 _×_ 10 _[−]_ [2] 2.71e-03 _[∗∗]_


CHB-MIT _→_ CHB-MIT _−_ 1 _._ 21 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ _−_ 5 _._ 91 _×_ 10 _[−]_ [3] 9.15e-05 _[∗∗∗]_

TUSZ _→_ CHB-MIT _−_ 2 _._ 35 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ _−_ 6 _._ 47 _×_ 10 _[−]_ [3] 6.66e-04 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 3 _._ 19 _×_ 10 _[−]_ [2] 2.24e-04 _[∗∗∗]_

CHB-MIT _→_ TUSZ 7 _._ 49 _×_ 10 _[−]_ [3] 7.81e-03 _[∗∗]_

TUSZ _→_ CHB-MIT 2 _._ 07 _×_ 10 _[−]_ [3] 5.06e-03 _[∗∗]_

TUSZ _→_ TUSZ 1 _._ 04 _×_ 10 _[−]_ [2] 2.39e-03 _[∗∗]_


CHB-MIT _→_ CHB-MIT 2 _._ 93 _×_ 10 _[−]_ [2] 6.72e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ 1 _._ 89 _×_ 10 _[−]_ [2] 2.60e-04 _[∗∗∗]_

TUSZ _→_ CHB-MIT 4 _._ 54 _×_ 10 _[−]_ [3] 3.32e-05 _[∗∗∗]_

TUSZ _→_ TUSZ 3 _._ 05 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 7 _._ 30 _×_ 10 _[−]_ [4] 8.70e-02
CHB-MIT _→_ TUSZ _−_ 1 _._ 27 _×_ 10 _[−]_ [3] 3.92e-01
TUSZ _→_ CHB-MIT _−_ 3 _._ 63 _×_ 10 _[−]_ [3] 8.56e-01
TUSZ _→_ TUSZ _−_ 7 _._ 63 _×_ 10 _[−]_ [3] 4.08e-02


CHB-MIT _→_ CHB-MIT 2 _._ 30 _×_ 10 _[−]_ [2] 3.32e-05 _[∗∗∗]_

CHB-MIT _→_ TUSZ 1 _._ 59 _×_ 10 _[−]_ [2] 6.47e-05 _[∗∗∗]_

TUSZ _→_ CHB-MIT 4 _._ 22 _×_ 10 _[−]_ [2] 1.19e-07 _[∗∗∗]_

TUSZ _→_ TUSZ 1 _._ 89 _×_ 10 _[−]_ [2] 3.86e-05 _[∗∗∗]_


CHB-MIT _→_ CHB-MIT 9 _._ 66 _×_ 10 _[−]_ [3] 7.20e-02
CHB-MIT _→_ TUSZ 2 _._ 65 _×_ 10 _[−]_ [3] 2.62e-01
TUSZ _→_ CHB-MIT 1 _._ 30 _×_ 10 _[−]_ [2] 2.39e-03 _[∗∗]_

TUSZ _→_ TUSZ 4 _._ 67 _×_ 10 _[−]_ [3] 1.22e-01


69


**Table B3** : A complete list of available features, broken down by type and channel scope. Legend: _T_
= Temporal, _ρ_ = Connectivity, _G_ = Graph Theory Derived, _δ, θ, α, β, γ_ = Frequency bands ( _delta,_
_theta, alpha, beta, gamma_ ); _•_ = Single channel, _••_
= Channel pair, [�] = All channels.


**Name** **Type** **Channels**


**Mean** _T_ _•_

**Variance** _T_ _•_

**Skewness** _T_ _•_

**Kurtosis** _T_ _•_
**Interquartile range** _T_ _•_
**Min** _T_ _•_

**Max** _T_ _•_
**Hjorth complexity** _T_ _•_
**Hjorth mobility** _T_ _•_
**Petrosian fractal dimension** _T_ _•_
**Intermittency** _T_ _•_
**Voltage auc** _T_ _•_
**Spikiness** _T_ _•_
**Standard deviation** _T_ _•_
**Zero crossing** _T_ _•_
**Peak to peak** _T_ _•_
**Absolute area under signal** _T_ _•_
**Total signal energy** _T_ _•_
**Spike count** _T_ _•_
**Coastline** _T_, _δ, θ, α, β, γ_ _•_
**Power spectral density** _δ, θ, α, β, γ_ _•_
**Power spectral centroid** _δ, θ, α, β, γ_ _•_
**Signal monotony** _δ, θ, α, β, γ_ _•_
**Signal to noise** _δ, θ, α, β, γ_ _•_
**Energy percentage** _δ, θ, α, β, γ_ _•_
**Discrete wavelet transform** _δ, θ, α, β, γ_ _•_
**Cross correlation max coef** _ρ_ _••_
**Coherence** _ρ_ _••_
**Imaginary coherence** _ρ_ _••_
**Phase slope index** _ρ_ _••_
**Eccentricity** _G_ �

**Clustering coefficient** _G_ �

**Betweenness centrality** _G_ �

**Local efficiency** _G_ �

**Global efficiency** _G_ �

**Diameter** _G_ �

**Radius** _G_ �

**Characteristic path** _G_ �


70


|MSE ROC Accuracy Precision F1 PRAUC Sensitivity Speci i fcity|0.1345±0.0610 0.8675±0.0913 0.8163±0.0907 0.5391 ± 0.0245 0.5167 ± 0.0642 0.3522 ± 0.0407 0.8163 ± 0.0907 0.7990 ± 0.0471 0.0297 ± 0.0102 0.8769 ± 0.0395 0.9695 ± 0.0114 0.6490 ± 0.0291 0.7070 ± 0.0392 0.5462 ± 0.0549 0.9695 ± 0.0114 0.8605 ± 0.0627 0.0898 ± 0.0204 0.9266 ± 0.0536 0.8906 ± 0.0260 0.5505 ± 0.0103 0.5624 ± 0.0242 0.4534 ± 0.0382 0.8906 ± 0.0260 0.8574 ± 0.0640 0.1317 ± 0.1498 0.8861 ± 0.0574 0.8025 ± 0.2322 0.5246 ± 0.0280 0.4784 ± 0.1154 0.1078 ± 0.0883 0.8025 ± 0.2322 0.7655 ± 0.0789 0.0796 ± 0.0799 0.8059 ± 0.0654 0.9023 ± 0.1018 0.5257 ± 0.0306 0.5155 ± 0.0691 0.1882 ± 0.1554 0.9023 ± 0.1018 0.7245 ± 0.0660 0.0585 ± 0.0531 0.9067 ± 0.0414 0.9286 ± 0.0694 0.5401 ± 0.0503 0.5432 ± 0.0873 0.2447 ± 0.1736 0.9286 ± 0.0694 0.7570 ± 0.0984 0.1069 ± 0.0926 0.7848 ± 0.0557 0.8306 ± 0.1691 0.5117 ± 0.0132 0.4708 ± 0.0712 0.0513 ± 0.0398 0.8306 ± 0.1691 0.6823 ± 0.0149 0.0316 ± 0.0235 0.5829 ± 0.0530 0.9684 ± 0.0235 0.6914 ± 0.1414 0.5763 ± 0.0775 0.2966 ± 0.1480 0.9684 ± 0.0235 0.5829 ± 0.0530 0.0382 ± 0.0258 0.9044 ± 0.0586 0.9752 ± 0.0142 0.7139 ± 0.1243 0.6034 ± 0.0739 0.3802 ± 0.1874 0.9752 ± 0.0142 0.6309 ± 0.0998|
|---|---|
||**LR**<br>**XGB**<br>**MLP** <br>**CNN**<br>**EEGNet** <br>**ConvLSTM**<br>**ConvTransformer**<br>**Binary voting** <br>**Mean voting**|


|MSE ROC Accuracy Precision F1 PRAUC Sensitivity Speci i fcity|0.2314 ± 0.0274 0.7132 ± 0.0502 0.6517 ± 0.0386 0.6138 ± 0.0286 0.6038 ± 0.0350 0.4153 ± 0.0739 0.6517 ± 0.0386 0.6574 ± 0.0404 0.1621 ± 0.0357 0.7602 ± 0.0415 0.8180 ± 0.0330 0.7543 ± 0.0561 0.7338 ± 0.0240 0.6354 ± 0.0636 0.8180 ± 0.0330 0.7250 ± 0.0084 0.1802 ± 0.0434 0.7532 ± 0.1048 0.7282 ± 0.0727 0.6521 ± 0.0735 0.6581 ± 0.0826 0.5202 ± 0.1748 0.7282 ± 0.0727 0.6866 ± 0.0911 0.1600 ± 0.0287 0.7952 ± 0.0668 0.7908 ± 0.0454 0.7041 ± 0.0555 0.6885 ± 0.0753 0.5555 ± 0.1152 0.7908 ± 0.0454 0.6923 ± 0.0832 0.2325 ± 0.0953 0.7067 ± 0.0675 0.6775 ± 0.1760 0.6346 ± 0.0649 0.5149 ± 0.0952 0.4060 ± 0.0631 0.6775 ± 0.1760 0.5683 ± 0.0478 0.1137 ± 0.0274 0.8594 ± 0.0445 0.8549 ± 0.0342 0.8345 ± 0.0228 0.7624 ± 0.0498 0.7430 ± 0.0434 0.8549 ± 0.0342 0.7365 ± 0.0486 0.1308 ± 0.0323 0.8265 ± 0.0594 0.8287 ± 0.0352 0.7993 ± 0.0369 0.7021 ± 0.0413 0.6519 ± 0.0460 0.8287 ± 0.0352 0.6749 ± 0.0344 0.1582 ± 0.0464 0.7313 ± 0.0420 0.8418 ± 0.0464 0.7953 ± 0.0615 0.7547 ± 0.0496 0.6799 ± 0.0629 0.8418 ± 0.0464 0.7313 ± 0.0420 0.1217 ± 0.0228 0.8638 ± 0.0603 0.8531 ± 0.0425 0.8200 ± 0.0569 0.7688 ± 0.0443 0.7341 ± 0.0738 0.8531 ± 0.0425 0.7413 ± 0.0378|
|---|---|
||**LR**<br>**XGB**<br>**MLP** <br>**CNN**<br>**EEGNet** <br>**ConvLSTM**<br>**ConvTransformer**<br>**Binary voting** <br>**Mean voting**|



71


|MSE ROC Accuracy Precision F1 PRAUC Sensitivity Speci i fcity|0.2368 ± 0.0288 0.5895 ± 0.0074 0.6590 ± 0.0791 0.5577 ± 0.0142 0.5285 ± 0.0158 0.2830 ± 0.0467 0.6590 ± 0.0791 0.5511 ± 0.0129 0.2212 ± 0.0412 0.5510 ± 0.0417 0.7660 ± 0.0267 0.6196 ± 0.0100 0.4928 ± 0.0268 0.3557 ± 0.0477 0.7660 ± 0.0267 0.5231 ± 0.0080 0.2282 ± 0.0250 0.6171 ± 0.0088 0.6930 ± 0.0457 0.5825 ± 0.0279 0.5502 ± 0.0109 0.3132 ± 0.0489 0.6930 ± 0.0457 0.5648 ± 0.0162 0.2687 ± 0.0692 0.6059 ± 0.0837 0.6704 ± 0.0835 0.5532 ± 0.0213 0.5080 ± 0.0164 0.2963 ± 0.0389 0.6704 ± 0.0835 0.5383 ± 0.0287 0.2317 ± 0.0293 0.5125 ± 0.0426 0.7609 ± 0.0307 0.5162 ± 0.0730 0.4507 ± 0.0183 0.2544 ± 0.0087 0.7609 ± 0.0307 0.5015 ± 0.0057 0.2339 ± 0.0355 0.6157 ± 0.0681 0.7118 ± 0.0690 0.6183 ± 0.1044 0.5172 ± 0.0618 0.3186 ± 0.0270 0.7118 ± 0.0690 0.5372 ± 0.0475 0.2845 ± 0.0666 0.5660 ± 0.0729 0.6233 ± 0.1209 0.5065 ± 0.0079 0.4668 ± 0.0233 0.2670 ± 0.0411 0.6233 ± 0.1209 0.5021 ± 0.0026 0.2448 ± 0.0256 0.5257 ± 0.0200 0.7552 ± 0.0256 0.5720 ± 0.0168 0.5004 ± 0.0404 0.3357 ± 0.0444 0.7552 ± 0.0256 0.5257 ± 0.0200 0.1841 ± 0.0249 0.6148 ± 0.0398 0.7562 ± 0.0274 0.5734 ± 0.0149 0.4979 ± 0.0410 0.3096 ± 0.0100 0.7562 ± 0.0274 0.5244 ± 0.0200|
|---|---|
||**LR** <br>**XGB**<br>**MLP** <br>**CNN**<br>**EEGNet** <br>**ConvLSTM**<br>**ConvTransformer**<br>**Binary voting** <br>**Mean voting**|


|MSE ROC Accuracy Precision F1 PRAUC Sensitivity Speci i fcity|0.2440 ± 0.0703 0.7758 ± 0.1120 0.6418 ± 0.1082 0.5130 ± 0.0063 0.4177 ± 0.0500 0.1378 ± 0.0551 0.6418 ± 0.1082 0.7029 ± 0.0951 0.4447 ± 0.1103 0.5931 ± 0.1621 0.4781 ± 0.1547 0.5045 ± 0.0094 0.3344 ± 0.0758 0.3008 ± 0.2043 0.4781 ± 0.1547 0.6061 ± 0.1440 0.2372 ± 0.0354 0.7724 ± 0.1527 0.6070 ± 0.1075 0.5116 ± 0.0071 0.4023 ± 0.0492 0.1186 ± 0.0397 0.6070 ± 0.1075 0.6979 ± 0.1183 0.1346 ± 0.1377 0.7686 ± 0.1029 0.7822 ± 0.2513 0.5148 ± 0.0109 0.4536 ± 0.1077 0.0733 ± 0.0534 0.7822 ± 0.2513 0.6512 ± 0.0229 0.6912 ± 0.1986 0.4219 ± 0.1970 0.0937 ± 0.0671 0.4960 ± 0.0036 0.0829 ± 0.0581 0.0238 ± 0.0305 0.0937 ± 0.0671 0.4220 ± 0.0915 0.0983 ± 0.1061 0.7902 ± 0.1182 0.8838 ± 0.1331 0.5154 ± 0.0116 0.4943 ± 0.0574 0.0411 ± 0.0227 0.8838 ± 0.1331 0.6665 ± 0.0392 0.0804 ± 0.0861 0.8650 ± 0.1006 0.8842 ± 0.1292 0.5175 ± 0.0148 0.4988 ± 0.0600 0.0428 ± 0.0173 0.8842 ± 0.1292 0.6760 ± 0.0446 0.3440 ± 0.1390 0.6639 ± 0.1149 0.6560 ± 0.1390 0.5102 ± 0.0076 0.4176 ± 0.0619 0.3527 ± 0.1318 0.6560 ± 0.1390 0.6639 ± 0.1149 0.1833 ± 0.0421 0.7618 ± 0.1753 0.7875 ± 0.1320 0.5190 ± 0.0146 0.4783 ± 0.0597 0.1383 ± 0.1232 0.7875 ± 0.1320 0.6997 ± 0.1246|
|---|---|
||**LR** <br>**XGB**<br>**MLP** <br>**CNN**<br>**EEGNet** <br>**ConvLSTM**<br>**ConvTransformer**<br>**Binary voting** <br>**Mean voting**|



72


