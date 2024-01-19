## 2017 (1 paper)



1. [Attention Is All You Need](http://arxiv.org/abs/1706.03762v7), Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, 12-06-2017
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks in an encoder-decoder configuration. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014
English-to-German translation task, improving over the existing best results,
including ensembles by over 2 BLEU. On the WMT 2014 English-to-French
translation task, our model establishes a new single-model state-of-the-art
BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction
of the training costs of the best models from the literature. We show that the
Transformer generalizes well to other tasks by applying it successfully to
English constituency parsing both with large and limited training data.
## 2018 (1 paper)



2. [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](http://arxiv.org/abs/1810.04805v2), Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, 11-10-2018
     ### Categories
     Computation and Language
    ### Abstract
    We introduce a new language representation model called BERT, which stands
for Bidirectional Encoder Representations from Transformers. Unlike recent
language representation models, BERT is designed to pre-train deep
bidirectional representations from unlabeled text by jointly conditioning on
both left and right context in all layers. As a result, the pre-trained BERT
model can be fine-tuned with just one additional output layer to create
state-of-the-art models for a wide range of tasks, such as question answering
and language inference, without substantial task-specific architecture
modifications.
  BERT is conceptually simple and empirically powerful. It obtains new
state-of-the-art results on eleven natural language processing tasks, including
pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI
accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering
Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).
## 2019 (7 papers)



3. [Generating Long Sequences with Sparse Transformers](http://arxiv.org/abs/1904.10509v1), Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever, 23-04-2019
     ### Categories
     Machine Learning
    ### Abstract
    Transformers are powerful sequence models, but require time and memory that
grows quadratically with the sequence length. In this paper we introduce sparse
factorizations of the attention matrix which reduce this to $O(n \sqrt{n})$. We
also introduce a) a variation on architecture and initialization to train
deeper networks, b) the recomputation of attention matrices to save memory, and
c) fast attention kernels for training. We call networks with these changes
Sparse Transformers, and show they can model sequences tens of thousands of
timesteps long using hundreds of layers. We use the same architecture to model
images, audio, and text from raw bytes, setting a new state of the art for
density modeling of Enwik8, CIFAR-10, and ImageNet-64. We generate
unconditional samples that demonstrate global coherence and great diversity,
and show it is possible in principle to use self-attention to model sequences
of length one million or more.


3. [Language Models as Knowledge Bases?](http://arxiv.org/abs/1909.01066v2), Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel, 03-09-2019
     ### Categories
     Computation and Language
    ### Abstract
    Recent progress in pretraining language models on large textual corpora led
to a surge of improvements for downstream NLP tasks. Whilst learning linguistic
knowledge, these models may also be storing relational knowledge present in the
training data, and may be able to answer queries structured as
"fill-in-the-blank" cloze statements. Language models have many advantages over
structured knowledge bases: they require no schema engineering, allow
practitioners to query about an open class of relations, are easy to extend to
more data, and require no human supervision to train. We present an in-depth
analysis of the relational knowledge already present (without fine-tuning) in a
wide range of state-of-the-art pretrained language models. We find that (i)
without fine-tuning, BERT contains relational knowledge competitive with
traditional NLP methods that have some access to oracle knowledge, (ii) BERT
also does remarkably well on open-domain question answering against a
supervised baseline, and (iii) certain types of factual knowledge are learned
much more readily than others by standard language model pretraining
approaches. The surprisingly strong ability of these models to recall factual
knowledge without any fine-tuning demonstrates their potential as unsupervised
open-domain QA systems. The code to reproduce our analysis is available at
https://github.com/facebookresearch/LAMA.


3. [Megatron-LM: Training Multi-Billion Parameter Language Models Using
  Model Parallelism](http://arxiv.org/abs/1909.08053v4), Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro, 17-09-2019
     ### Categories
     Computation and Language
    ### Abstract
    Recent work in language modeling demonstrates that training large transformer
models advances the state of the art in Natural Language Processing
applications. However, very large models can be quite difficult to train due to
memory constraints. In this work, we present our techniques for training very
large transformer models and implement a simple, efficient intra-layer model
parallel approach that enables training transformer models with billions of
parameters. Our approach does not require a new compiler or library changes, is
orthogonal and complimentary to pipeline model parallelism, and can be fully
implemented with the insertion of a few communication operations in native
PyTorch. We illustrate this approach by converging transformer based models up
to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the
entire application with 76% scaling efficiency when compared to a strong single
GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To
demonstrate that large language models can further advance the state of the art
(SOTA), we train an 8.3 billion parameter transformer language model similar to
GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful
attention to the placement of layer normalization in BERT-like models is
critical to achieving increased performance as the model size grows. Using the
GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA
perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%)
datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9%
compared to SOTA accuracy of 89.4%).


3. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer](http://arxiv.org/abs/1910.10683v4), Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu, 23-10-2019
     ### Categories
     Machine Learning, Computation and Language
    ### Abstract
    Transfer learning, where a model is first pre-trained on a data-rich task
before being fine-tuned on a downstream task, has emerged as a powerful
technique in natural language processing (NLP). The effectiveness of transfer
learning has given rise to a diversity of approaches, methodology, and
practice. In this paper, we explore the landscape of transfer learning
techniques for NLP by introducing a unified framework that converts all
text-based language problems into a text-to-text format. Our systematic study
compares pre-training objectives, architectures, unlabeled data sets, transfer
approaches, and other factors on dozens of language understanding tasks. By
combining the insights from our exploration with scale and our new ``Colossal
Clean Crawled Corpus'', we achieve state-of-the-art results on many benchmarks
covering summarization, question answering, text classification, and more. To
facilitate future work on transfer learning for NLP, we release our data set,
pre-trained models, and code.


3. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
  Generation, Translation, and Comprehension](http://arxiv.org/abs/1910.13461v1), Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer, 29-10-2019
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    We present BART, a denoising autoencoder for pretraining sequence-to-sequence
models. BART is trained by (1) corrupting text with an arbitrary noising
function, and (2) learning a model to reconstruct the original text. It uses a
standard Tranformer-based neural machine translation architecture which,
despite its simplicity, can be seen as generalizing BERT (due to the
bidirectional encoder), GPT (with the left-to-right decoder), and many other
more recent pretraining schemes. We evaluate a number of noising approaches,
finding the best performance by both randomly shuffling the order of the
original sentences and using a novel in-filling scheme, where spans of text are
replaced with a single mask token. BART is particularly effective when fine
tuned for text generation but also works well for comprehension tasks. It
matches the performance of RoBERTa with comparable training resources on GLUE
and SQuAD, achieves new state-of-the-art results on a range of abstractive
dialogue, question answering, and summarization tasks, with gains of up to 6
ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system
for machine translation, with only target language pretraining. We also report
ablation experiments that replicate other pretraining schemes within the BART
framework, to better measure which factors most influence end-task performance.


3. [How Can We Know What Language Models Know?](http://arxiv.org/abs/1911.12543v2), Zhengbao Jiang, Frank F. Xu, Jun Araki, Graham Neubig, 28-11-2019
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Recent work has presented intriguing results examining the knowledge
contained in language models (LM) by having the LM fill in the blanks of
prompts such as "Obama is a _ by profession". These prompts are usually
manually created, and quite possibly sub-optimal; another prompt such as "Obama
worked as a _" may result in more accurately predicting the correct profession.
Because of this, given an inappropriate prompt, we might fail to retrieve facts
that the LM does know, and thus any given prompt only provides a lower bound
estimate of the knowledge contained in an LM. In this paper, we attempt to more
accurately estimate the knowledge contained in LMs by automatically discovering
better prompts to use in this querying process. Specifically, we propose
mining-based and paraphrasing-based methods to automatically generate
high-quality and diverse prompts, as well as ensemble methods to combine
answers from different prompts. Extensive experiments on the LAMA benchmark for
extracting relational knowledge from LMs demonstrate that our methods can
improve accuracy from 31.1% to 39.6%, providing a tighter lower bound on what
LMs know. We have released the code and the resulting LM Prompt And Query
Archive (LPAQA) at https://github.com/jzbjyb/LPAQA.


3. [Zero-shot Text Classification With Generative Language Models](http://arxiv.org/abs/1912.10165v1), Raul Puri, Bryan Catanzaro, 10-12-2019
     ### Categories
     Computation and Language
    ### Abstract
    This work investigates the use of natural language to enable zero-shot model
adaptation to new tasks. We use text and metadata from social commenting
platforms as a source for a simple pretraining task. We then provide the
language model with natural language descriptions of classification tasks as
input and train it to generate the correct answer in natural language via a
language modeling objective. This allows the model to generalize to new
classification tasks without the need for multiple multitask classification
heads. We show the zero-shot performance of these generative language models,
trained with weak supervision, on six benchmark text classification datasets
from the torchtext library. Despite no access to training data, we achieve up
to a 45% absolute improvement in classification accuracy over random or
majority class baselines. These results show that natural language can serve as
simple and powerful descriptors for task adaptation. We believe this points the
way to new metalearning strategies for text problems.
## 2020 (8 papers)



4. [Longformer: The Long-Document Transformer](http://arxiv.org/abs/2004.05150v2), Iz Beltagy, Matthew E. Peters, Arman Cohan, 10-04-2020
     ### Categories
     Computation and Language
    ### Abstract
    Transformer-based models are unable to process long sequences due to their
self-attention operation, which scales quadratically with the sequence length.
To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process
documents of thousands of tokens or longer. Longformer's attention mechanism is
a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work
on long-sequence transformers, we evaluate Longformer on character-level
language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a
variety of downstream tasks. Our pretrained Longformer consistently outperforms
RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop
and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a
Longformer variant for supporting long document generative sequence-to-sequence
tasks, and demonstrate its effectiveness on the arXiv summarization dataset.


4. [ColBERT: Efficient and Effective Passage Search via Contextualized Late
  Interaction over BERT](http://arxiv.org/abs/2004.12832v2), Omar Khattab, Matei Zaharia, 27-04-2020
     ### Categories
     Computation and Language
    ### Abstract
    Recent progress in Natural Language Understanding (NLU) is driving fast-paced
advances in Information Retrieval (IR), largely owed to fine-tuning deep
language models (LMs) for document ranking. While remarkably effective, the
ranking models based on these LMs increase computational cost by orders of
magnitude over prior approaches, particularly as they must feed each
query-document pair through a massive neural network to compute a single
relevance score. To tackle this, we present ColBERT, a novel ranking model that
adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT
introduces a late interaction architecture that independently encodes the query
and the document using BERT and then employs a cheap yet powerful interaction
step that models their fine-grained similarity. By delaying and yet retaining
this fine-granular interaction, ColBERT can leverage the expressiveness of deep
LMs while simultaneously gaining the ability to pre-compute document
representations offline, considerably speeding up query processing. Beyond
reducing the cost of re-ranking the documents retrieved by a traditional model,
ColBERT's pruning-friendly interaction mechanism enables leveraging
vector-similarity indexes for end-to-end retrieval directly from a large
document collection. We extensively evaluate ColBERT using two recent passage
search datasets. Results show that ColBERT's effectiveness is competitive with
existing BERT-based models (and outperforms every non-BERT baseline), while
executing two orders-of-magnitude faster and requiring four orders-of-magnitude
fewer FLOPs per query.


4. [It's Not Just Size That Matters: Small Language Models Are Also Few-Shot
  Learners](http://arxiv.org/abs/2009.07118v2), Timo Schick, Hinrich Schütze, 15-09-2020
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    When scaled to hundreds of billions of parameters, pretrained language models
such as GPT-3 (Brown et al., 2020) achieve remarkable few-shot performance.
However, enormous amounts of compute are required for training and applying
such big models, resulting in a large carbon footprint and making it difficult
for researchers and practitioners to use them. We show that performance similar
to GPT-3 can be obtained with language models that are much "greener" in that
their parameter count is several orders of magnitude smaller. This is achieved
by converting textual inputs into cloze questions that contain a task
description, combined with gradient-based optimization; exploiting unlabeled
data gives further improvements. We identify key factors required for
successful natural language understanding with small language models.


4. [FLIN: A Flexible Natural Language Interface for Web Navigation](http://arxiv.org/abs/2010.12844v2), Sahisnu Mazumder, Oriana Riva, 24-10-2020
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    AI assistants can now carry out tasks for users by directly interacting with
website UIs. Current semantic parsing and slot-filling techniques cannot
flexibly adapt to many different websites without being constantly re-trained.
We propose FLIN, a natural language interface for web navigation that maps user
commands to concept-level actions (rather than low-level UI actions), thus
being able to flexibly adapt to different websites and handle their transient
nature. We frame this as a ranking problem: given a user command and a webpage,
FLIN learns to score the most relevant navigation instruction (involving action
and parameter values). To train and evaluate FLIN, we collect a dataset using
nine popular websites from three domains. Our results show that FLIN was able
to adapt to new websites in a given domain.


4. [Automatically Identifying Words That Can Serve as Labels for Few-Shot
  Text Classification](http://arxiv.org/abs/2010.13641v1), Timo Schick, Helmut Schmid, Hinrich Schütze, 26-10-2020
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    A recent approach for few-shot text classification is to convert textual
inputs to cloze questions that contain some form of task description, process
them with a pretrained language model and map the predicted words to labels.
Manually defining this mapping between words and labels requires both domain
expertise and an understanding of the language model's abilities. To mitigate
this issue, we devise an approach that automatically finds such a mapping given
small amounts of training data. For a number of tasks, the mapping found by our
approach performs almost as well as hand-crafted label-to-word mappings.


4. [AutoPrompt: Eliciting Knowledge from Language Models with Automatically
  Generated Prompts](http://arxiv.org/abs/2010.15980v2), Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh, 29-10-2020
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    The remarkable success of pretrained language models has motivated the study
of what kinds of knowledge these models learn during pretraining. Reformulating
tasks as fill-in-the-blanks problems (e.g., cloze tests) is a natural approach
for gauging such knowledge, however, its usage is limited by the manual effort
and guesswork required to write suitable prompts. To address this, we develop
AutoPrompt, an automated method to create prompts for a diverse set of tasks,
based on a gradient-guided search. Using AutoPrompt, we show that masked
language models (MLMs) have an inherent capability to perform sentiment
analysis and natural language inference without additional parameters or
finetuning, sometimes achieving performance on par with recent state-of-the-art
supervised models. We also show that our prompts elicit more accurate factual
knowledge from MLMs than the manually created prompts on the LAMA benchmark,
and that MLMs can be used as relation extractors more effectively than
supervised relation extraction models. These results demonstrate that
automatically generated prompts are a viable parameter-free alternative to
existing probing methods, and as pretrained LMs become more sophisticated and
capable, potentially a replacement for finetuning.


4. [Making Pre-trained Language Models Better Few-shot Learners](http://arxiv.org/abs/2012.15723v2), Tianyu Gao, Adam Fisch, Danqi Chen, 31-12-2020
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    The recent GPT-3 model (Brown et al., 2020) achieves remarkable few-shot
performance solely by leveraging a natural-language prompt and a few task
demonstrations as input context. Inspired by their findings, we study few-shot
learning in a more practical scenario, where we use smaller language models for
which fine-tuning is computationally efficient. We present LM-BFF--better
few-shot fine-tuning of language models--a suite of simple and complementary
techniques for fine-tuning language models on a small number of annotated
examples. Our approach includes (1) prompt-based fine-tuning together with a
novel pipeline for automating prompt generation; and (2) a refined strategy for
dynamically and selectively incorporating demonstrations into each context.
Finally, we present a systematic evaluation for analyzing few-shot performance
on a range of NLP tasks, including classification and regression. Our
experiments demonstrate that our methods combine to dramatically outperform
standard fine-tuning procedures in this low resource setting, achieving up to
30% absolute improvement, and 11% on average across all tasks. Our approach
makes minimal assumptions on task resources and domain expertise, and hence
constitutes a strong task-agnostic method for few-shot learning.


4. [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](http://arxiv.org/abs/2101.00027v1), Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy, 31-12-2020
     ### Categories
     Computation and Language
    ### Abstract
    Recent work has demonstrated that increased training dataset diversity
improves general cross-domain knowledge and downstream generalization
capability for large-scale language models. With this in mind, we present
\textit{the Pile}: an 825 GiB English text corpus targeted at training
large-scale language models. The Pile is constructed from 22 diverse
high-quality subsets -- both existing and newly constructed -- many of which
derive from academic or professional sources. Our evaluation of the untuned
performance of GPT-2 and GPT-3 on the Pile shows that these models struggle on
many of its components, such as academic writing. Conversely, models trained on
the Pile improve significantly over both Raw CC and CC-100 on all components of
the Pile, while improving performance on downstream evaluations. Through an
in-depth exploratory analysis, we document potentially concerning aspects of
the data for prospective users. We make publicly available the code used in its
construction.
## 2021 (45 papers)



5. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](http://arxiv.org/abs/2101.00190v1), Xiang Lisa Li, Percy Liang, 01-01-2021
     ### Categories
     Computation and Language
    ### Abstract
    Fine-tuning is the de facto way to leverage large pretrained language models
to perform downstream tasks. However, it modifies all the language model
parameters and therefore necessitates storing a full copy for each task. In
this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning
for natural language generation tasks, which keeps language model parameters
frozen, but optimizes a small continuous task-specific vector (called the
prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent
tokens to attend to this prefix as if it were "virtual tokens". We apply
prefix-tuning to GPT-2 for table-to-text generation and to BART for
summarization. We find that by learning only 0.1\% of the parameters,
prefix-tuning obtains comparable performance in the full data setting,
outperforms fine-tuning in low-data settings, and extrapolates better to
examples with topics unseen during training.


5. [Prompt Programming for Large Language Models: Beyond the Few-Shot
  Paradigm](http://arxiv.org/abs/2102.07350v1), Laria Reynolds, Kyle McDonell, 15-02-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Prevailing methods for mapping large generative language models to supervised
tasks may fail to sufficiently probe models' novel capabilities. Using GPT-3 as
a case study, we show that 0-shot prompts can significantly outperform few-shot
prompts. We suggest that the function of few-shot examples in these cases is
better described as locating an already learned task rather than meta-learning.
This analysis motivates rethinking the role of prompts in controlling and
evaluating powerful language models. In this work, we discuss methods of prompt
programming, emphasizing the usefulness of considering prompts through the lens
of natural language. We explore techniques for exploiting the capacity of
narratives and cultural anchors to encode nuanced intentions and techniques for
encouraging deconstruction of a problem into components before producing a
verdict. Informed by this more encompassing theory of prompt programming, we
also introduce the idea of a metaprompt that seeds the model to generate its
own natural language prompts for a range of tasks. Finally, we discuss how
these more general methods of interacting with language models can be
incorporated into existing and future benchmarks and practical applications.


5. [Calibrate Before Use: Improving Few-Shot Performance of Language Models](http://arxiv.org/abs/2102.09690v2), Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh, 19-02-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    GPT-3 can perform numerous tasks when provided a natural language prompt that
contains a few training examples. We show that this type of few-shot learning
can be unstable: the choice of prompt format, training examples, and even the
order of the training examples can cause accuracy to vary from near chance to
near state-of-the-art. We demonstrate that this instability arises from the
bias of language models towards predicting certain answers, e.g., those that
are placed near the end of the prompt or are common in the pre-training data.
To mitigate this, we first estimate the model's bias towards each answer by
asking for its prediction when given the training prompt and a content-free
test input such as "N/A". We then fit calibration parameters that cause the
prediction for this input to be uniform across answers. On a diverse set of
tasks, this contextual calibration procedure substantially improves GPT-3 and
GPT-2's average accuracy (up to 30.0% absolute) and reduces variance across
different choices of the prompt.


5. [PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen
  Domains](http://arxiv.org/abs/2102.12206v4), Eyal Ben-David, Nadav Oved, Roi Reichart, 24-02-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Natural Language Processing algorithms have made incredible progress, but
they still struggle when applied to out-of-distribution examples. We address a
challenging and underexplored version of this domain adaptation problem, where
an algorithm is trained on several source domains, and then applied to examples
from unseen domains that are unknown at training time. Particularly, no
examples, labeled or unlabeled, or any other knowledge about the target domain
are available to the algorithm at training time. We present PADA: An
example-based autoregressive Prompt learning algorithm for on-the-fly
Any-Domain Adaptation, based on the T5 language model. Given a test example,
PADA first generates a unique prompt for it and then, conditioned on this
prompt, labels the example with respect to the NLP prediction task. PADA is
trained to generate a prompt which is a token sequence of unrestricted length,
consisting of Domain Related Features (DRFs) that characterize each of the
source domains. Intuitively, the generated prompt is a unique signature that
maps the test example to a semantic space spanned by the source domains. In
experiments with 3 tasks (text classification and sequence tagging), for a
total of 14 multi-source adaptation scenarios, PADA substantially outperforms
strong baselines.


5. [Learning Transferable Visual Models From Natural Language Supervision](http://arxiv.org/abs/2103.00020v1), Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, 26-02-2021
     ### Categories
     Machine Learning
    ### Abstract
    State-of-the-art computer vision systems are trained to predict a fixed set
of predetermined object categories. This restricted form of supervision limits
their generality and usability since additional labeled data is needed to
specify any other visual concept. Learning directly from raw text about images
is a promising alternative which leverages a much broader source of
supervision. We demonstrate that the simple pre-training task of predicting
which caption goes with which image is an efficient and scalable way to learn
SOTA image representations from scratch on a dataset of 400 million (image,
text) pairs collected from the internet. After pre-training, natural language
is used to reference learned visual concepts (or describe new ones) enabling
zero-shot transfer of the model to downstream tasks. We study the performance
of this approach by benchmarking on over 30 different existing computer vision
datasets, spanning tasks such as OCR, action recognition in videos,
geo-localization, and many types of fine-grained object classification. The
model transfers non-trivially to most tasks and is often competitive with a
fully supervised baseline without the need for any dataset specific training.
For instance, we match the accuracy of the original ResNet-50 on ImageNet
zero-shot without needing to use any of the 1.28 million training examples it
was trained on. We release our code and pre-trained model weights at
https://github.com/OpenAI/CLIP.


5. [How Many Data Points is a Prompt Worth?](http://arxiv.org/abs/2103.08493v2), Teven Le Scao, Alexander M. Rush, 15-03-2021
     ### Categories
     Machine Learning
    ### Abstract
    When fine-tuning pretrained models for classification, researchers either use
a generic model head or a task-specific prompt for prediction. Proponents of
prompting have argued that prompts provide a method for injecting task-specific
guidance, which is beneficial in low-data regimes. We aim to quantify this
benefit through rigorous testing of prompts in a fair setting: comparing
prompted and head-based fine-tuning in equal conditions across many tasks and
data sizes. By controlling for many sources of advantage, we find that
prompting does indeed provide a benefit, and that this benefit can be
quantified per task. Results show that prompting is often worth 100s of data
points on average across classification tasks.


5. [GPT Understands, Too](http://arxiv.org/abs/2103.10385v2), Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang, 18-03-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Prompting a pretrained language model with natural language patterns has been
proved effective for natural language understanding (NLU). However, our
preliminary study reveals that manual discrete prompts often lead to unstable
performance -- e.g., changing a single word in the prompt might result in
substantial performance drop. We propose a novel method P-Tuning that employs
trainable continuous prompt embeddings in concatenation with discrete prompts.
Empirically, P-Tuning not only stabilizes training by minimizing the gap
between various discrete prompts, but also improves performance by a sizeable
margin on a wide range of NLU tasks including LAMA and SuperGLUE. P-Tuning is
generally effective for both frozen and tuned language models, under both the
fully-supervised and few-shot settings.


5. [Adapting Language Models for Zero-shot Learning by Meta-tuning on
  Dataset and Prompt Collections](http://arxiv.org/abs/2104.04670v5), Ruiqi Zhong, Kristy Lee, Zheng Zhang, Dan Klein, 10-04-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large pre-trained language models (LMs) such as GPT-3 have acquired a
surprising ability to perform zero-shot learning. For example, to classify
sentiment without any training examples, we can "prompt" the LM with the review
and the label description "Does the user like this movie?", and ask whether the
next word is "yes" or "no". However, the next word prediction training
objective is still misaligned with the target zero-shot learning objective. To
address this weakness, we propose meta-tuning, which directly optimizes the
zero-shot learning objective by fine-tuning pre-trained language models on a
collection of datasets. We focus on classification tasks, and construct the
meta-dataset by aggregating 43 existing datasets and annotating 441 label
descriptions in a question-answering (QA) format. When evaluated on unseen
tasks, meta-tuned models outperform a same-sized QA model and the previous SOTA
zero-shot learning system based on natural language inference. Additionally,
increasing parameter count from 220M to 770M improves AUC-ROC scores by 6.3%,
and we forecast that even larger models would perform better. Therefore,
measuring zero-shot learning performance on language models out-of-the-box
might underestimate their true potential, and community-wide efforts on
aggregating datasets and unifying their formats can help build models that
answer prompts better.


5. [Learning How to Ask: Querying LMs with Mixtures of Soft Prompts](http://arxiv.org/abs/2104.06599v1), Guanghui Qin, Jason Eisner, 14-04-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Natural-language prompts have recently been used to coax pretrained language
models into performing other AI tasks, using a fill-in-the-blank paradigm
(Petroni et al., 2019) or a few-shot extrapolation paradigm (Brown et al.,
2020). For example, language models retain factual knowledge from their
training corpora that can be extracted by asking them to "fill in the blank" in
a sentential prompt. However, where does this prompt come from? We explore the
idea of learning prompts by gradient descent -- either fine-tuning prompts
taken from previous work, or starting from random initialization. Our prompts
consist of "soft words," i.e., continuous vectors that are not necessarily word
type embeddings from the language model. Furthermore, for each task, we
optimize a mixture of prompts, learning which prompts are most effective and
how to ensemble them. Across multiple English LMs and tasks, our approach
hugely outperforms previous methods, showing that the implicit factual
knowledge in language models was previously underestimated. Moreover, this
knowledge is cheap to elicit: random initialization is nearly as good as
informed initialization.


5. [Generating Datasets with Pretrained Language Models](http://arxiv.org/abs/2104.07540v3), Timo Schick, Hinrich Schütze, 15-04-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    To obtain high-quality sentence embeddings from pretrained language models
(PLMs), they must either be augmented with additional pretraining objectives or
finetuned on a large set of labeled text pairs. While the latter approach
typically outperforms the former, it requires great human effort to generate
suitable datasets of sufficient size. In this paper, we show how PLMs can be
leveraged to obtain high-quality sentence embeddings without the need for
labeled data, finetuning or modifications to the pretraining objective: We
utilize the generative abilities of large and high-performing PLMs to generate
entire datasets of labeled text pairs from scratch, which we then use for
finetuning much smaller and more efficient models. Our fully unsupervised
approach outperforms strong baselines on several semantic textual similarity
datasets.


5. [Surface Form Competition: Why the Highest Probability Answer Isn't
  Always Right](http://arxiv.org/abs/2104.08315v9), Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, Luke Zettlemoyer, 16-04-2021
     ### Categories
     Computation and Language
    ### Abstract
    Large language models have shown promising results in zero-shot settings
(Brown et al.,2020; Radford et al., 2019). For example, they can perform
multiple choice tasks simply by conditioning on a question and selecting the
answer with the highest probability.
  However, ranking by string probability can be problematic due to surface form
competition-wherein different surface forms compete for probability mass, even
if they represent the same underlying concept, e.g. "computer" and "PC." Since
probability mass is finite, this lowers the probability of the correct answer,
due to competition from other strings that are valid answers (but not one of
the multiple choice options).
  We introduce Domain Conditional Pointwise Mutual Information, an alternative
scoring function that directly compensates for surface form competition by
simply reweighing each option according to a term that is proportional to its a
priori likelihood within the context of the specific zero-shot task. It
achieves consistent gains in zero-shot performance over both calibrated (Zhao
et al., 2021) and uncalibrated scoring functions on all GPT-2 and GPT-3 models
over a variety of multiple choice datasets.


5. [Fantastically Ordered Prompts and Where to Find Them: Overcoming
  Few-Shot Prompt Order Sensitivity](http://arxiv.org/abs/2104.08786v2), Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp, 18-04-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    When primed with only a handful of training samples, very large, pretrained
language models such as GPT-3 have shown competitive results when compared to
fully-supervised, fine-tuned, large, pretrained language models. We demonstrate
that the order in which the samples are provided can make the difference
between near state-of-the-art and random guess performance: essentially some
permutations are "fantastic" and some not. We analyse this phenomenon in
detail, establishing that: it is present across model sizes (even for the
largest current models), it is not related to a specific subset of samples, and
that a given good permutation for one model is not transferable to another.
While one could use a development set to determine which permutations are
performant, this would deviate from the true few-shot setting as it requires
additional annotated data. Instead, we use the generative nature of language
models to construct an artificial development set and based on entropy
statistics of the candidate permutations on this set, we identify performant
prompts. Our method yields a 13% relative improvement for GPT-family models
across eleven different established text classification tasks.


5. [GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation](http://arxiv.org/abs/2104.08826v2), Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, Woomyeong Park, 18-04-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large-scale language models such as GPT-3 are excellent few-shot learners,
allowing them to be controlled via natural text prompts. Recent studies report
that prompt-based direct classification eliminates the need for fine-tuning but
lacks data and inference scalability. This paper proposes a novel data
augmentation technique that leverages large-scale language models to generate
realistic text samples from a mixture of real samples. We also propose
utilizing soft-labels predicted by the language models, effectively distilling
knowledge from the large-scale language models and creating textual
perturbations simultaneously. We perform data augmentation experiments on
diverse classification tasks and show that our method hugely outperforms
existing text augmentation methods. Ablation studies and a qualitative analysis
provide more insights into our approach.


5. [The Power of Scale for Parameter-Efficient Prompt Tuning](http://arxiv.org/abs/2104.08691v2), Brian Lester, Rami Al-Rfou, Noah Constant, 18-04-2021
     ### Categories
     Computation and Language
    ### Abstract
    In this work, we explore "prompt tuning", a simple yet effective mechanism
for learning "soft prompts" to condition frozen language models to perform
specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft
prompts are learned through backpropagation and can be tuned to incorporate
signal from any number of labeled examples. Our end-to-end learned approach
outperforms GPT-3's "few-shot" learning by a large margin. More remarkably,
through ablations on model size using T5, we show that prompt tuning becomes
more competitive with scale: as models exceed billions of parameters, our
method "closes the gap" and matches the strong performance of model tuning
(where all model weights are tuned). This finding is especially relevant in
that large models are costly to share and serve, and the ability to reuse one
frozen model for multiple downstream tasks can ease this burden. Our method can
be seen as a simplification of the recently proposed "prefix tuning" of Li and
Liang (2021), and we provide a comparison to this and other similar approaches.
Finally, we show that conditioning a frozen model with soft prompts confers
benefits in robustness to domain transfer, as compared to full model tuning.


5. [PTR: Prompt Tuning with Rules for Text Classification](http://arxiv.org/abs/2105.11259v3), Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, Maosong Sun, 24-05-2021
     ### Categories
     Computation and Language
    ### Abstract
    Fine-tuned pre-trained language models (PLMs) have achieved awesome
performance on almost all NLP tasks. By using additional prompts to fine-tune
PLMs, we can further stimulate the rich knowledge distributed in PLMs to better
serve downstream tasks. Prompt tuning has achieved promising results on some
few-class classification tasks such as sentiment classification and natural
language inference. However, manually designing lots of language prompts is
cumbersome and fallible. For those auto-generated prompts, it is also expensive
and time-consuming to verify their effectiveness in non-few-shot scenarios.
Hence, it is still challenging for prompt tuning to address many-class
classification tasks. To this end, we propose prompt tuning with rules (PTR)
for many-class text classification and apply logic rules to construct prompts
with several sub-prompts. In this way, PTR is able to encode prior knowledge of
each class into prompt tuning. We conduct experiments on relation
classification, a typical and complicated many-class classification task, and
the results show that PTR can significantly and consistently outperform
existing state-of-the-art baselines. This indicates that PTR is a promising
approach to take advantage of both human prior knowledge and PLMs for those
complicated classification tasks.


5. [True Few-Shot Learning with Language Models](http://arxiv.org/abs/2105.11447v1), Ethan Perez, Douwe Kiela, Kyunghyun Cho, 24-05-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Pretrained language models (LMs) perform well on many tasks even when
learning from a few examples, but prior work uses many held-out examples to
tune various aspects of learning, such as hyperparameters, training objectives,
and natural language templates ("prompts"). Here, we evaluate the few-shot
ability of LMs when such held-out examples are unavailable, a setting we call
true few-shot learning. We test two model selection criteria, cross-validation
and minimum description length, for choosing LM prompts and hyperparameters in
the true few-shot setting. On average, both marginally outperform random
selection and greatly underperform selection based on held-out examples.
Moreover, selection criteria often prefer models that perform significantly
worse than randomly-selected ones. We find similar results even when taking
into account our uncertainty in a model's true performance during selection, as
well as when varying the amount of computation and number of examples used for
selection. Overall, our findings suggest that prior work significantly
overestimated the true few-shot ability of LMs given the difficulty of few-shot
model selection.


5. [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685v2), Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, 17-06-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    An important paradigm of natural language processing consists of large-scale
pre-training on general domain data and adaptation to particular tasks or
domains. As we pre-train larger models, full fine-tuning, which retrains all
model parameters, becomes less feasible. Using GPT-3 175B as an example --
deploying independent instances of fine-tuned models, each with 175B
parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or
LoRA, which freezes the pre-trained model weights and injects trainable rank
decomposition matrices into each layer of the Transformer architecture, greatly
reducing the number of trainable parameters for downstream tasks. Compared to
GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable
parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA
performs on-par or better than fine-tuning in model quality on RoBERTa,
DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher
training throughput, and, unlike adapters, no additional inference latency. We
also provide an empirical investigation into rank-deficiency in language model
adaptation, which sheds light on the efficacy of LoRA. We release a package
that facilitates the integration of LoRA with PyTorch models and provide our
implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at
https://github.com/microsoft/LoRA.


5. [Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis
  of Head and Prompt Tuning](http://arxiv.org/abs/2106.09226v2), Colin Wei, Sang Michael Xie, Tengyu Ma, 17-06-2021
     ### Categories
     Machine Learning
    ### Abstract
    Pretrained language models have achieved state-of-the-art performance when
adapted to a downstream NLP task. However, theoretical analysis of these models
is scarce and challenging since the pretraining and downstream tasks can be
very different. We propose an analysis framework that links the pretraining and
downstream tasks with an underlying latent variable generative model of text --
the downstream classifier must recover a function of the posterior distribution
over the latent variables. We analyze head tuning (learning a classifier on top
of the frozen pretrained model) and prompt tuning in this setting. The
generative model in our analysis is either a Hidden Markov Model (HMM) or an
HMM augmented with a latent memory component, motivated by long-term
dependencies in natural language. We show that 1) under certain non-degeneracy
conditions on the HMM, simple classification heads can solve the downstream
task, 2) prompt tuning obtains downstream guarantees with weaker non-degeneracy
conditions, and 3) our recovery guarantees for the memory-augmented HMM are
stronger than for the vanilla HMM because task-relevant information is easier
to recover from the long-term memory. Experiments on synthetically generated
data from HMMs back our theoretical findings.


5. [Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with
  Language Models](http://arxiv.org/abs/2106.13353v2), Robert L. Logan IV, Ivana Balažević, Eric Wallace, Fabio Petroni, Sameer Singh, Sebastian Riedel, 24-06-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Prompting language models (LMs) with training examples and task descriptions
has been seen as critical to recent successes in few-shot learning. In this
work, we show that finetuning LMs in the few-shot setting can considerably
reduce the need for prompt engineering. In fact, one can use null prompts,
prompts that contain neither task-specific templates nor training examples, and
achieve competitive accuracy to manually-tuned prompts across a wide range of
tasks. While finetuning LMs does introduce new parameters for each downstream
task, we show that this memory overhead can be substantially reduced:
finetuning only the bias terms can achieve comparable or better accuracy than
standard finetuning while only updating 0.1% of the parameters. All in all, we
recommend finetuning LMs for few-shot learning as it is more accurate, robust
to different prompts, and can be made nearly as efficient as using frozen LMs.


5. [Deduplicating Training Data Makes Language Models Better](http://arxiv.org/abs/2107.06499v2), Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini, 14-07-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    We find that existing language modeling datasets contain many near-duplicate
examples and long repetitive substrings. As a result, over 1% of the unprompted
output of language models trained on these datasets is copied verbatim from the
training data. We develop two tools that allow us to deduplicate training
datasets -- for example removing from C4 a single 61 word English sentence that
is repeated over 60,000 times. Deduplication allows us to train models that
emit memorized text ten times less frequently and require fewer train steps to
achieve the same or better accuracy. We can also reduce train-test overlap,
which affects over 4% of the validation set of standard datasets, thus allowing
for more accurate evaluation. We release code for reproducing our work and
performing dataset deduplication at
https://github.com/google-research/deduplicate-text-datasets.


5. [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods
  in Natural Language Processing](http://arxiv.org/abs/2107.13586v1), Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig, 28-07-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    This paper surveys and organizes research works in a new paradigm in natural
language processing, which we dub "prompt-based learning". Unlike traditional
supervised learning, which trains a model to take in an input x and predict an
output y as P(y|x), prompt-based learning is based on language models that
model the probability of text directly. To use these models to perform
prediction tasks, the original input x is modified using a template into a
textual string prompt x' that has some unfilled slots, and then the language
model is used to probabilistically fill the unfilled information to obtain a
final string x, from which the final output y can be derived. This framework is
powerful and attractive for a number of reasons: it allows the language model
to be pre-trained on massive amounts of raw text, and by defining a new
prompting function the model is able to perform few-shot or even zero-shot
learning, adapting to new scenarios with few or no labeled data. In this paper
we introduce the basics of this promising paradigm, describe a unified set of
mathematical notations that can cover a wide variety of existing work, and
organize existing work along several dimensions, e.g.the choice of pre-trained
models, prompts, and tuning strategies. To make the field more accessible to
interested beginners, we not only make a systematic review of existing works
and a highly structured typology of prompt-based concepts, but also release
other resources, e.g., a website http://pretrain.nlpedia.ai/ including
constantly-updated survey, and paperlist.


5. [Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt
  Verbalizer for Text Classification](http://arxiv.org/abs/2108.02035v2), Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li, Wei Wu, Maosong Sun, 04-08-2021
     ### Categories
     Computation and Language
    ### Abstract
    Tuning pre-trained language models (PLMs) with task-specific prompts has been
a promising approach for text classification. Particularly, previous studies
suggest that prompt-tuning has remarkable superiority in the low-data scenario
over the generic fine-tuning methods with extra classifiers. The core idea of
prompt-tuning is to insert text pieces, i.e., template, to the input and
transform a classification problem into a masked language modeling problem,
where a crucial step is to construct a projection, i.e., verbalizer, between a
label space and a label word space. A verbalizer is usually handcrafted or
searched by gradient descent, which may lack coverage and bring considerable
bias and high variances to the results. In this work, we focus on incorporating
external knowledge into the verbalizer, forming a knowledgeable prompt-tuning
(KPT), to improve and stabilize prompt-tuning. Specifically, we expand the
label word space of the verbalizer using external knowledge bases (KBs) and
refine the expanded label word space with the PLM itself before predicting with
the expanded label word space. Extensive experiments on zero and few-shot text
classification tasks demonstrate the effectiveness of knowledgeable
prompt-tuning.


5. [Noisy Channel Language Model Prompting for Few-Shot Text Classification](http://arxiv.org/abs/2108.04106v3), Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer, 09-08-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We introduce a noisy channel approach for language model prompting in
few-shot text classification. Instead of computing the likelihood of the label
given the input (referred as direct models), channel models compute the
conditional probability of the input given the label, and are thereby required
to explain every word in the input. We use channel models for recently proposed
few-shot learning methods with no or very limited updates to the language model
parameters, via either in-context demonstration or prompt tuning. Our
experiments show that, for both methods, channel models significantly
outperform their direct counterparts, which we attribute to their stability,
i.e., lower variance and higher worst-case accuracy. We also present extensive
ablations that provide recommendations for when to use channel prompt tuning
instead of other competitive methods (e.g., direct head tuning): channel prompt
tuning is preferred when the number of training examples is small, labels in
the training data are imbalanced, or generalization to unseen labels is
required.


5. [FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated
  Learning](http://arxiv.org/abs/2108.06098v3), Nam Hyeon-Woo, Moon Ye-Bin, Tae-Hyun Oh, 13-08-2021
     ### Categories
     Machine Learning
    ### Abstract
    In this work, we propose a communication-efficient parameterization, FedPara,
for federated learning (FL) to overcome the burdens on frequent model uploads
and downloads. Our method re-parameterizes weight parameters of layers using
low-rank weights followed by the Hadamard product. Compared to the conventional
low-rank parameterization, our FedPara method is not restricted to low-rank
constraints, and thereby it has a far larger capacity. This property enables to
achieve comparable performance while requiring 3 to 10 times lower
communication costs than the model with the original layers, which is not
achievable by the traditional low-rank methods. The efficiency of our method
can be further improved by combining with other efficient FL optimizers. In
addition, we extend our method to a personalized FL application, pFedPara,
which separates parameters into global and local ones. We show that pFedPara
outperforms competing personalized FL methods with more than three times fewer
parameters.


5. [Differentiable Prompt Makes Pre-trained Language Models Better Few-shot
  Learners](http://arxiv.org/abs/2108.13161v7), Ningyu Zhang, Luoqiu Li, Xiang Chen, Shumin Deng, Zhen Bi, Chuanqi Tan, Fei Huang, Huajun Chen, 30-08-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large-scale pre-trained language models have contributed significantly to
natural language processing by demonstrating remarkable abilities as few-shot
learners. However, their effectiveness depends mainly on scaling the model
parameters and prompt design, hindering their implementation in most real-world
applications. This study proposes a novel pluggable, extensible, and efficient
approach named DifferentiAble pRompT (DART), which can convert small language
models into better few-shot learners without any prompt engineering. The main
principle behind this approach involves reformulating potential natural
language processing tasks into the task of a pre-trained language model and
differentially optimizing the prompt template as well as the target label with
backpropagation. Furthermore, the proposed approach can be: (i) Plugged to any
pre-trained language models; (ii) Extended to widespread classification tasks.
A comprehensive evaluation of standard NLP tasks demonstrates that the proposed
approach achieves a better few-shot performance. Code is available in
https://github.com/zjunlp/DART.


5. [Want To Reduce Labeling Cost? GPT-3 Can Help](http://arxiv.org/abs/2108.13487v1), Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, Michael Zeng, 30-08-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Data annotation is a time-consuming and labor-intensive process for many NLP
tasks. Although there exist various methods to produce pseudo data labels, they
are often task-specific and require a decent amount of labeled data to start
with. Recently, the immense language model GPT-3 with 175 billion parameters
has achieved tremendous improvement across many few-shot learning tasks. In
this paper, we explore ways to leverage GPT-3 as a low-cost data labeler to
train other models. We find that, to make the downstream model achieve the same
performance on a variety of NLU and NLG tasks, it costs 50% to 96% less to use
labels from GPT-3 than using labels from humans. Furthermore, we propose a
novel framework of combining pseudo labels from GPT-3 with human labels, which
leads to even better performance with limited labeling budget. These results
present a cost-effective data labeling methodology that is generalizable to
many practical applications.


5. [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](http://arxiv.org/abs/2109.01247v2), Albert Webson, Ellie Pavlick, 02-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    Recently, a boom of papers has shown extraordinary progress in zero-shot and
few-shot learning with various prompt-based models. It is commonly argued that
prompts help models to learn faster in the same way that humans learn faster
when provided with task instructions expressed in natural language. In this
study, we experiment with over 30 prompt templates manually written for natural
language inference (NLI). We find that models learn just as fast with many
prompts that are intentionally irrelevant or even pathologically misleading as
they do with instructively "good" prompts. Further, such patterns hold even for
models as large as 175 billion parameters (Brown et al., 2020) as well as the
recently proposed instruction-tuned models which are trained on hundreds of
prompts (Sanh et al., 2022). That is, instruction-tuned models often produce
good predictions with irrelevant and misleading prompts even at zero shots. In
sum, notwithstanding prompt-based models' impressive improvement, we find
evidence of serious limitations that question the degree to which such
improvement is derived from models understanding task instructions in ways
analogous to humans' use of task instructions.


5. [Finetuned Language Models Are Zero-Shot Learners](http://arxiv.org/abs/2109.01652v5), Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le, 03-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    This paper explores a simple method for improving the zero-shot learning
abilities of language models. We show that instruction tuning -- finetuning
language models on a collection of tasks described via instructions --
substantially improves zero-shot performance on unseen tasks.
  We take a 137B parameter pretrained language model and instruction-tune it on
over 60 NLP tasks verbalized via natural language instruction templates. We
evaluate this instruction-tuned model, which we call FLAN, on unseen task
types. FLAN substantially improves the performance of its unmodified
counterpart and surpasses zero-shot 175B GPT-3 on 20 of 25 tasks that we
evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE,
BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number
of finetuning datasets, model scale, and natural language instructions are key
to the success of instruction tuning.


5. [General-Purpose Question-Answering with Macaw](http://arxiv.org/abs/2109.02593v1), Oyvind Tafjord, Peter Clark, 06-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Despite the successes of pretrained language models, there are still few
high-quality, general-purpose QA systems that are freely available. In
response, we present Macaw, a versatile, generative question-answering (QA)
system that we are making available to the community. Macaw is built on
UnifiedQA, itself built on T5, and exhibits strong performance, zero-shot, on a
wide variety of topics, including outperforming GPT-3 by over 10% (absolute) on
Challenge300, a suite of 300 challenge questions, despite being an order of
magnitude smaller (11 billion vs. 175 billion parameters). In addition, Macaw
allows different permutations ("angles") of its inputs and outputs to be used,
for example Macaw can take a question and produce an answer; or take an answer
and produce a question; or take an answer and question, and produce
multiple-choice options. We describe the system, and illustrate a variety of
question types where it produces surprisingly good answers, well outside the
training setup. We also identify question classes where it still appears to
struggle, offering insights into the limitations of pretrained language models.
Macaw is freely available, and we hope that it proves useful to the community.
Macaw is available at https://github.com/allenai/macaw


5. [Discrete and Soft Prompting for Multilingual Models](http://arxiv.org/abs/2109.03630v1), Mengjie Zhao, Hinrich Schütze, 08-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    It has been shown for English that discrete and soft prompting perform
strongly in few-shot learning with pretrained language models (PLMs). In this
paper, we show that discrete and soft prompting perform better than finetuning
in multilingual cases: Crosslingual transfer and in-language training of
multilingual natural language inference. For example, with 48 English training
examples, finetuning obtains 33.74% accuracy in crosslingual transfer, barely
surpassing the majority baseline (33.33%). In contrast, discrete and soft
prompting outperform finetuning, achieving 36.43% and 38.79%. We also
demonstrate good performance of prompting with training data in multiple
languages other than English.


5. [Open Aspect Target Sentiment Classification with Natural Language
  Prompts](http://arxiv.org/abs/2109.03685v1), Ronald Seoh, Ian Birle, Mrinal Tak, Haw-Shiuan Chang, Brian Pinette, Alfred Hough, 08-09-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    For many business applications, we often seek to analyze sentiments
associated with any arbitrary aspects of commercial products, despite having a
very limited amount of labels or even without any labels at all. However,
existing aspect target sentiment classification (ATSC) models are not trainable
if annotated datasets are not available. Even with labeled data, they fall
short of reaching satisfactory performance. To address this, we propose simple
approaches that better solve ATSC with natural language prompts, enabling the
task under zero-shot cases and enhancing supervised settings, especially for
few-shot cases. Under the few-shot setting for SemEval 2014 Task 4 laptop
domain, our method of reformulating ATSC as an NLI task outperforms supervised
SOTA approaches by up to 24.13 accuracy points and 33.14 macro F1 points.
Moreover, we demonstrate that our prompts could handle implicitly stated
aspects as well: our models reach about 77% accuracy on detecting sentiments
for aspect categories (e.g., food), which do not necessarily appear within the
text, even though we trained the models only with explicitly mentioned aspect
terms (e.g., fajitas) from just 16 reviews - while the accuracy of the
no-prompt baseline is only around 65%.


5. [Avoiding Inference Heuristics in Few-shot Prompt-based Finetuning](http://arxiv.org/abs/2109.04144v1), Prasetya Ajie Utama, Nafise Sadat Moosavi, Victor Sanh, Iryna Gurevych, 09-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Recent prompt-based approaches allow pretrained language models to achieve
strong performances on few-shot finetuning by reformulating downstream tasks as
a language modeling problem. In this work, we demonstrate that, despite its
advantages on low data regimes, finetuned prompt-based models for sentence pair
classification tasks still suffer from a common pitfall of adopting inference
heuristics based on lexical overlap, e.g., models incorrectly assuming a
sentence pair is of the same meaning because they consist of the same set of
words. Interestingly, we find that this particular inference heuristic is
significantly less present in the zero-shot evaluation of the prompt-based
model, indicating how finetuning can be destructive to useful knowledge learned
during the pretraining. We then show that adding a regularization that
preserves pretraining weights is effective in mitigating this destructive
tendency of few-shot finetuning. Our evaluation on three datasets demonstrates
promising improvements on the three corresponding challenge datasets used to
diagnose the inference heuristics.


5. [PPT: Pre-trained Prompt Tuning for Few-shot Learning](http://arxiv.org/abs/2109.04332v3), Yuxian Gu, Xu Han, Zhiyuan Liu, Minlie Huang, 09-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    Prompts for pre-trained language models (PLMs) have shown remarkable
performance by bridging the gap between pre-training tasks and various
downstream tasks. Among these methods, prompt tuning, which freezes PLMs and
only tunes soft prompts, provides an efficient and effective solution for
adapting large-scale PLMs to downstream tasks. However, prompt tuning is yet to
be fully explored. In our pilot experiments, we find that prompt tuning
performs comparably with conventional full-model fine-tuning when downstream
data are sufficient, whereas it performs much worse under few-shot learning
settings, which may hinder the application of prompt tuning in practice. We
attribute this low performance to the manner of initializing soft prompts.
Therefore, in this work, we propose to pre-train prompts by adding soft prompts
into the pre-training stage to obtain a better initialization. We name this
Pre-trained Prompt Tuning framework "PPT". To ensure the generalization of PPT,
we formulate similar classification tasks into a unified task form and
pre-train soft prompts for this unified task. Extensive experiments show that
tuning pre-trained prompts for downstream tasks can reach or even outperform
full-model fine-tuning under both full-data and few-shot settings. Our approach
is effective and efficient for using large-scale PLMs in practice.


5. [CINS: Comprehensive Instruction for Few-shot Learning in Task-oriented
  Dialog Systems](http://arxiv.org/abs/2109.04645v4), Fei Mi, Yitong Li, Yasheng Wang, Xin Jiang, Qun Liu, 10-09-2021
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    As labeling cost for different modules in task-oriented dialog (ToD) systems
is high, a major challenge in practice is to learn different tasks with the
least amount of labeled data. Recently, prompting methods over pre-trained
language models (PLMs) have shown promising results for few-shot learning in
ToD. To better utilize the power of PLMs, this paper proposes Comprehensive
Instruction (CINS) that exploits PLMs with extra task-specific instructions. We
design a schema (definition, constraint, prompt) of instructions and their
customized realizations for three important downstream tasks in ToD, i.e.
intent classification, dialog state tracking, and natural language generation.
A sequence-to-sequence model (T5) is adopted to solve these three tasks in a
unified framework. Extensive experiments are conducted on these ToD tasks in
realistic few-shot learning scenarios with small validation data. Empirical
results demonstrate that the proposed CINS approach consistently improves
techniques that finetune PLMs with raw input or short prompts.


5. [PoKE: A Prompt-based Knowledge Eliciting Approach for Event Argument
  Extraction](http://arxiv.org/abs/2109.05190v3), Jiaju Lin, Qin Chen, 11-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Eliciting knowledge from pre-trained language models via prompt-based
learning has shown great potential in many natural language processing tasks.
Whereas, the applications for more complex tasks such as event extraction are
less studied since the design of prompt is not straightforward for the
structured event containing various triggers and arguments. % Meanwhile,
current conditional generation methods employ large encoder-decoder models,
which are costly to train and serve. In this paper, we present a novel
prompt-based approach, which elicits both the independent and joint knowledge
about different events for event argument extraction. The experimental results
on the benchmark ACE2005 dataset show the great advantages of our proposed
approach. In particular, our approach is superior to the recent advanced
methods in both fully-supervised and low-resource scenarios.


5. [Exploring Prompt-based Few-shot Learning for Grounded Dialog Generation](http://arxiv.org/abs/2109.06513v2), Chujie Zheng, Minlie Huang, 14-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    Dialog models can be greatly strengthened through grounding on various
external information, but grounded dialog corpora are usually not naturally
accessible. In this work, we focus on the few-shot learning for grounded dialog
generation (GDG). We first propose a simple prompting method for GDG tasks,
where different constructs of model input, such as the grounding source and the
conversation context, are distinguished through continuous or discrete prompts.
On three typical GDG tasks, we empirically demonstrate and analyze in-depth the
effectiveness of our method. We then conduct extensive experiments to
thoroughly investigate how our prompting method works with different
pre-trained models. We show that prompted language models perform superiorly to
conversational models, and further analyze various factors that influence the
effects of prompting. Overall, our work introduces a prompt-based perspective
to the few-shot learning for GDG tasks, and provides valuable findings and
insights for future research.


5. [Can Language Models be Biomedical Knowledge Bases?](http://arxiv.org/abs/2109.07154v1), Mujeen Sung, Jinhyuk Lee, Sean Yi, Minji Jeon, Sungdong Kim, Jaewoo Kang, 15-09-2021
     ### Categories
     Computation and Language
    ### Abstract
    Pre-trained language models (LMs) have become ubiquitous in solving various
natural language processing (NLP) tasks. There has been increasing interest in
what knowledge these LMs contain and how we can extract that knowledge,
treating LMs as knowledge bases (KBs). While there has been much work on
probing LMs in the general domain, there has been little attention to whether
these powerful LMs can be used as domain-specific KBs. To this end, we create
the BioLAMA benchmark, which is comprised of 49K biomedical factual knowledge
triples for probing biomedical LMs. We find that biomedical LMs with recently
proposed probing methods can achieve up to 18.51% Acc@5 on retrieving
biomedical knowledge. Although this seems promising given the task difficulty,
our detailed analyses reveal that most predictions are highly correlated with
prompt templates without any subjects, hence producing similar results on each
relation and hindering their capabilities to be used as domain-specific KBs. We
hope that BioLAMA can serve as a challenging benchmark for biomedical factual
probing.


5. [Language Models are Few-shot Multilingual Learners](http://arxiv.org/abs/2109.07684v1), Genta Indra Winata, Andrea Madotto, Zhaojiang Lin, Rosanne Liu, Jason Yosinski, Pascale Fung, 16-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    General-purpose language models have demonstrated impressive capabilities,
performing on par with state-of-the-art approaches on a range of downstream
natural language processing (NLP) tasks and benchmarks when inferring
instructions from very few examples. Here, we evaluate the multilingual skills
of the GPT and T5 models in conducting multi-class classification on
non-English languages without any parameter updates. We show that, given a few
English examples as context, pre-trained language models can predict not only
English test samples but also non-English ones. Finally, we find the in-context
few-shot cross-lingual prediction results of language models are significantly
better than random prediction, and they are competitive compared to the
existing state-of-the-art cross-lingual models.


5. [Reframing Instructional Prompts to GPTk's Language](http://arxiv.org/abs/2109.07830v3), Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, Hannaneh Hajishirzi, 16-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    What kinds of instructional prompts are easier to follow for Language Models
(LMs)? We study this question by conducting extensive empirical analysis that
shed light on important features of successful instructional prompts.
Specifically, we study several classes of reframing techniques for manual
reformulation of prompts into more effective ones. Some examples include
decomposing a complex task instruction into multiple simpler tasks or itemizing
instructions into sequential steps. Our experiments compare the zero-shot and
few-shot performance of LMs prompted with reframed instructions on 12 NLP tasks
across 6 categories. Compared with original instructions, our reframed
instructions lead to significant improvements across LMs with different sizes.
For example, the same reframed prompts boost few-shot performance of
GPT3-series and GPT2-series by 12.5% and 6.7% respectively averaged over all
tasks. Furthermore, reframed instructions reduce the number of examples
required to prompt LMs in the few-shot setting. We hope these
empirically-driven techniques will pave the way towards more effective future
prompting algorithms.


5. [SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based
  Sentiment Analysis](http://arxiv.org/abs/2109.08306v1), Chengxi Li, Feiyu Gao, Jiajun Bu, Lu Xu, Xiang Chen, Yu Gu, Zirui Shao, Qi Zheng, Ningyu Zhang, Yongpan Wang, Zhi Yu, 17-09-2021
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Aspect-based sentiment analysis (ABSA) is an emerging fine-grained sentiment
analysis task that aims to extract aspects, classify corresponding sentiment
polarities and find opinions as the causes of sentiment. The latest research
tends to solve the ABSA task in a unified way with end-to-end frameworks. Yet,
these frameworks get fine-tuned from downstream tasks without any task-adaptive
modification. Specifically, they do not use task-related knowledge well or
explicitly model relations between aspect and opinion terms, hindering them
from better performance. In this paper, we propose SentiPrompt to use sentiment
knowledge enhanced prompts to tune the language model in the unified framework.
We inject sentiment knowledge regarding aspects, opinions, and polarities into
prompt and explicitly model term relations via constructing consistency and
polarity judgment templates from the ground truth triplets. Experimental
results demonstrate that our approach can outperform strong baselines on
Triplet Extraction, Pair Extraction, and Aspect Term Extraction with Sentiment
Classification by a notable margin.


5. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally
  Across Scales and Tasks](http://arxiv.org/abs/2110.07602v3), Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang, 14-10-2021
     ### Categories
     Computation and Language
    ### Abstract
    Prompt tuning, which only tunes continuous prompts with a frozen language
model, substantially reduces per-task storage and memory usage at training.
However, in the context of NLU, prior work reveals that prompt tuning does not
perform well for normal-sized pretrained models. We also find that existing
methods of prompt tuning cannot handle hard sequence labeling tasks, indicating
a lack of universality. We present a novel empirical finding that properly
optimized prompt tuning can be universally effective across a wide range of
model scales and NLU tasks. It matches the performance of finetuning while
having only 0.1%-3% tuned parameters. Our method P-Tuning v2 is an
implementation of Deep Prompt Tuning \cite{li2021prefix,qin2021learning}
optimized and adapted for NLU. Given the universality and simplicity of
P-Tuning v2, we believe it can serve as an alternative to finetuning and a
strong baseline for future research.Our code and data are released at
https://github.com/THUDM/P-tuning-v2.


5. [Generated Knowledge Prompting for Commonsense Reasoning](http://arxiv.org/abs/2110.08387v3), Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, Hannaneh Hajishirzi, 15-10-2021
     ### Categories
     Computation and Language
    ### Abstract
    It remains an open question whether incorporating external knowledge benefits
commonsense reasoning while maintaining the flexibility of pretrained sequence
models. To investigate this question, we develop generated knowledge prompting,
which consists of generating knowledge from a language model, then providing
the knowledge as additional input when answering a question. Our method does
not require task-specific supervision for knowledge integration, or access to a
structured knowledge base, yet it improves performance of large-scale,
state-of-the-art models on four commonsense reasoning tasks, achieving
state-of-the-art results on numerical commonsense (NumerSense), general
commonsense (CommonsenseQA 2.0), and scientific commonsense (QASC) benchmarks.
Generated knowledge prompting highlights large-scale language models as
flexible sources of external knowledge for improving commonsense reasoning. Our
code is available at https://github.com/liujch1998/GKP


5. [Multitask Prompted Training Enables Zero-Shot Task Generalization](http://arxiv.org/abs/2110.08207v3), Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, Alexander M. Rush, 15-10-2021
     ### Categories
     Machine Learning, Computation and Language
    ### Abstract
    Large language models have recently been shown to attain reasonable zero-shot
generalization on a diverse set of tasks (Brown et al., 2020). It has been
hypothesized that this is a consequence of implicit multitask learning in
language models' pretraining (Radford et al., 2019). Can zero-shot
generalization instead be directly induced by explicit multitask learning? To
test this question at scale, we develop a system for easily mapping any natural
language tasks into a human-readable prompted form. We convert a large set of
supervised datasets, each with multiple prompts with diverse wording. These
prompted datasets allow for benchmarking the ability of a model to perform
completely held-out tasks. We fine-tune a pretrained encoder-decoder model
(Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a
wide variety of tasks. The model attains strong zero-shot performance on
several standard datasets, often outperforming models up to 16x its size.
Further, our approach attains strong performance on a subset of tasks from the
BIG-bench benchmark, outperforming models up to 6x its size. All trained models
are available at https://github.com/bigscience-workshop/t-zero and all prompts
are available at https://github.com/bigscience-workshop/promptsource.


5. [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
  Interaction](http://arxiv.org/abs/2112.01488v3), Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia, 02-12-2021
     ### Categories
     Computation and Language
    ### Abstract
    Neural information retrieval (IR) has greatly advanced search and other
knowledge-intensive language tasks. While many neural IR methods encode queries
and documents into single-vector representations, late interaction models
produce multi-vector representations at the granularity of each token and
decompose relevance modeling into scalable token-level computations. This
decomposition has been shown to make late interaction more effective, but it
inflates the space footprint of these models by an order of magnitude. In this
work, we introduce ColBERTv2, a retriever that couples an aggressive residual
compression mechanism with a denoised supervision strategy to simultaneously
improve the quality and space footprint of late interaction. We evaluate
ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art
quality within and outside the training domain while reducing the space
footprint of late interaction models by 6--10$\times$.


5. [WebGPT: Browser-assisted question-answering with human feedback](http://arxiv.org/abs/2112.09332v3), Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, John Schulman, 17-12-2021
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    We fine-tune GPT-3 to answer long-form questions using a text-based
web-browsing environment, which allows the model to search and navigate the
web. By setting up the task so that it can be performed by humans, we are able
to train models on the task using imitation learning, and then optimize answer
quality with human feedback. To make human evaluation of factual accuracy
easier, models must collect references while browsing in support of their
answers. We train and evaluate our models on ELI5, a dataset of questions asked
by Reddit users. Our best model is obtained by fine-tuning GPT-3 using behavior
cloning, and then performing rejection sampling against a reward model trained
to predict human preferences. This model's answers are preferred by humans 56%
of the time to those of our human demonstrators, and 69% of the time to the
highest-voted answer from Reddit.
## 2022 (47 papers)



6. [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge
  for Embodied Agents](http://arxiv.org/abs/2201.07207v2), Wenlong Huang, Pieter Abbeel, Deepak Pathak, Igor Mordatch, 18-01-2022
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Can world knowledge learned by large language models (LLMs) be used to act in
interactive environments? In this paper, we investigate the possibility of
grounding high-level tasks, expressed in natural language (e.g. "make
breakfast"), to a chosen set of actionable steps (e.g. "open fridge"). While
prior work focused on learning from explicit step-by-step examples of how to
act, we surprisingly find that if pre-trained LMs are large enough and prompted
appropriately, they can effectively decompose high-level tasks into mid-level
plans without any further training. However, the plans produced naively by LLMs
often cannot map precisely to admissible actions. We propose a procedure that
conditions on existing demonstrations and semantically translates the plans to
admissible actions. Our evaluation in the recent VirtualHome environment shows
that the resulting method substantially improves executability over the LLM
baseline. The conducted human evaluation reveals a trade-off between
executability and correctness but shows a promising sign towards extracting
actionable knowledge from language models. Website at
https://huangwl18.github.io/language-planner


6. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](http://arxiv.org/abs/2201.11903v6), Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou, 28-01-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We explore how generating a chain of thought -- a series of intermediate
reasoning steps -- significantly improves the ability of large language models
to perform complex reasoning. In particular, we show how such reasoning
abilities emerge naturally in sufficiently large language models via a simple
method called chain of thought prompting, where a few chain of thought
demonstrations are provided as exemplars in prompting. Experiments on three
large language models show that chain of thought prompting improves performance
on a range of arithmetic, commonsense, and symbolic reasoning tasks. The
empirical gains can be striking. For instance, prompting a 540B-parameter
language model with just eight chain of thought exemplars achieves state of the
art accuracy on the GSM8K benchmark of math word problems, surpassing even
finetuned GPT-3 with a verifier.


6. [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155v1), Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe, 04-03-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Making language models bigger does not inherently make them better at
following a user's intent. For example, large language models can generate
outputs that are untruthful, toxic, or simply not helpful to the user. In other
words, these models are not aligned with their users. In this paper, we show an
avenue for aligning language models with user intent on a wide range of tasks
by fine-tuning with human feedback. Starting with a set of labeler-written
prompts and prompts submitted through the OpenAI API, we collect a dataset of
labeler demonstrations of the desired model behavior, which we use to fine-tune
GPT-3 using supervised learning. We then collect a dataset of rankings of model
outputs, which we use to further fine-tune this supervised model using
reinforcement learning from human feedback. We call the resulting models
InstructGPT. In human evaluations on our prompt distribution, outputs from the
1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3,
despite having 100x fewer parameters. Moreover, InstructGPT models show
improvements in truthfulness and reductions in toxic output generation while
having minimal performance regressions on public NLP datasets. Even though
InstructGPT still makes simple mistakes, our results show that fine-tuning with
human feedback is a promising direction for aligning language models with human
intent.


6. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](http://arxiv.org/abs/2203.11171v4), Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou, 21-03-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Chain-of-thought prompting combined with pre-trained large language models
has achieved encouraging results on complex reasoning tasks. In this paper, we
propose a new decoding strategy, self-consistency, to replace the naive greedy
decoding used in chain-of-thought prompting. It first samples a diverse set of
reasoning paths instead of only taking the greedy one, and then selects the
most consistent answer by marginalizing out the sampled reasoning paths.
Self-consistency leverages the intuition that a complex reasoning problem
typically admits multiple different ways of thinking leading to its unique
correct answer. Our extensive empirical evaluation shows that self-consistency
boosts the performance of chain-of-thought prompting with a striking margin on
a range of popular arithmetic and commonsense reasoning benchmarks, including
GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and
ARC-challenge (+3.9%).


6. [Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language](http://arxiv.org/abs/2204.00598v2), Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, Pete Florence, 01-04-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Large pretrained (e.g., "foundation") models exhibit distinct capabilities
depending on the domain of data they are trained on. While these domains are
generic, they may only barely overlap. For example, visual-language models
(VLMs) are trained on Internet-scale image captions, but large language models
(LMs) are further trained on Internet-scale text with no images (e.g.,
spreadsheets, SAT questions, code). As a result, these models store different
forms of commonsense knowledge across different domains. In this work, we show
that this diversity is symbiotic, and can be leveraged through Socratic Models
(SMs): a modular framework in which multiple pretrained models may be composed
zero-shot i.e., via multimodal-informed prompting, to exchange information with
each other and capture new multimodal capabilities, without requiring
finetuning. With minimal engineering, SMs are not only competitive with
state-of-the-art zero-shot image captioning and video-to-text retrieval, but
also enable new applications such as (i) answering free-form questions about
egocentric video, (ii) engaging in multimodal assistive dialogue with people
(e.g., for cooking recipes) by interfacing with external APIs and databases
(e.g., web search), and (iii) robot perception and planning.


6. [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](http://arxiv.org/abs/2204.01691v2), Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, Andy Zeng, 04-04-2022
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Large language models can encode a wealth of semantic knowledge about the
world. Such knowledge could be extremely useful to robots aiming to act upon
high-level, temporally extended instructions expressed in natural language.
However, a significant weakness of language models is that they lack real-world
experience, which makes it difficult to leverage them for decision making
within a given embodiment. For example, asking a language model to describe how
to clean a spill might result in a reasonable narrative, but it may not be
applicable to a particular agent, such as a robot, that needs to perform this
task in a particular environment. We propose to provide real-world grounding by
means of pretrained skills, which are used to constrain the model to propose
natural language actions that are both feasible and contextually appropriate.
The robot can act as the language model's "hands and eyes," while the language
model supplies high-level semantic knowledge about the task. We show how
low-level skills can be combined with large language models so that the
language model provides high-level knowledge about the procedures for
performing complex and temporally-extended instructions, while value functions
associated with these skills provide the grounding necessary to connect this
knowledge to a particular physical environment. We evaluate our method on a
number of real-world robotic tasks, where we show the need for real-world
grounding and that this approach is capable of completing long-horizon,
abstract, natural language instructions on a mobile manipulator. The project's
website and the video can be found at https://say-can.github.io/.


6. [Training a Helpful and Harmless Assistant with Reinforcement Learning
  from Human Feedback](http://arxiv.org/abs/2204.05862v1), Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan, 12-04-2022
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    We apply preference modeling and reinforcement learning from human feedback
(RLHF) to finetune language models to act as helpful and harmless assistants.
We find this alignment training improves performance on almost all NLP
evaluations, and is fully compatible with training for specialized skills such
as python coding and summarization. We explore an iterated online mode of
training, where preference models and RL policies are updated on a weekly
cadence with fresh human feedback data, efficiently improving our datasets and
models. Finally, we investigate the robustness of RLHF training, and identify a
roughly linear relation between the RL reward and the square root of the KL
divergence between the policy and its initialization. Alongside our main
results, we perform peripheral analyses on calibration, competing objectives,
and the use of OOD detection, compare our models with human writers, and
provide samples from our models using prompts appearing in recent related work.


6. [Inferring Implicit Relations in Complex Questions with Language Models](http://arxiv.org/abs/2204.13778v2), Uri Katz, Mor Geva, Jonathan Berant, 28-04-2022
     ### Categories
     Computation and Language
    ### Abstract
    A prominent challenge for modern language understanding systems is the
ability to answer implicit reasoning questions, where the required reasoning
steps for answering the question are not mentioned in the text explicitly. In
this work, we investigate why current models struggle with implicit reasoning
question answering (QA) tasks, by decoupling inference of reasoning steps from
their execution. We define a new task of implicit relation inference and
construct a benchmark, IMPLICITRELATIONS, where given a question, a model
should output a list of concept-relation pairs, where the relations describe
the implicit reasoning steps required for answering the question. Using
IMPLICITRELATIONS, we evaluate models from the GPT-3 family and find that,
while these models struggle on the implicit reasoning QA task, they often
succeed at inferring implicit relations. This suggests that the challenge in
implicit reasoning questions does not stem from the need to plan a reasoning
strategy alone, but to do it while also retrieving and reasoning over relevant
information.


6. [The Unreliability of Explanations in Few-shot Prompting for Textual
  Reasoning](http://arxiv.org/abs/2205.03401v2), Xi Ye, Greg Durrett, 06-05-2022
     ### Categories
     Computation and Language
    ### Abstract
    Does prompting a large language model (LLM) like GPT-3 with explanations
improve in-context learning? We study this question on two NLP tasks that
involve reasoning over text, namely question answering and natural language
inference. We test the performance of four LLMs on three textual reasoning
datasets using prompts that include explanations in multiple different styles.
For these tasks, we find that including explanations in the prompts for OPT,
GPT-3 (davinci), and InstructGPT (text-davinci-001) only yields small to
moderate accuracy improvements over standard few-show learning. However,
text-davinci-002 is able to benefit more substantially.
  We further show that explanations generated by the LLMs may not entail the
models' predictions nor be factually grounded in the input, even on simple
tasks with extractive explanations. However, these flawed explanations can
still be useful as a way to verify LLMs' predictions post-hoc. Through analysis
in our three settings, we show that explanations judged by humans to be
good--logically consistent with the input and the prediction--more likely
cooccur with accurate predictions. Following these observations, we train
calibrators using automatically extracted scores that assess the reliability of
explanations, allowing us to improve performance post-hoc across all of our
datasets.


6. [UL2: Unifying Language Learning Paradigms](http://arxiv.org/abs/2205.05131v3), Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, Donald Metzler, 10-05-2022
     ### Categories
     Computation and Language
    ### Abstract
    Existing pre-trained models are generally geared towards a particular class
of problems. To date, there seems to be still no consensus on what the right
architecture and pre-training setup should be. This paper presents a unified
framework for pre-training models that are universally effective across
datasets and setups. We begin by disentangling architectural archetypes with
pre-training objectives -- two concepts that are commonly conflated. Next, we
present a generalized & unified perspective for self-supervision in NLP and
show how different pre-training objectives can be cast as one another and how
interpolating between different objectives can be effective. We then propose
Mixture-of-Denoisers (MoD), a pre-training objective that combines diverse
pre-training paradigms together. We furthermore introduce a notion of mode
switching, wherein downstream fine-tuning is associated with specific
pre-training schemes. We conduct extensive ablative experiments to compare
multiple pre-training objectives and find that our method pushes the
Pareto-frontier by outperforming T5 & GPT-like models across multiple diverse
setups. By scaling our model up to 20B parameters, we achieve SOTA performance
on 50 well-established supervised finetuning based NLP tasks. Our model also
achieve strong results at in-context learning, outperforming 175B GPT-3 on
zero-shot SuperGLUE and tripling the performance of T5-XXL on one-shot
summarization. On 0-shot MMLU, UL2 20B outperforms T0 and T5 models. UL2 20B
also works well with chain-of-thought prompting and reasoning, making it an
appealing choice for research into reasoning at a small to medium scale of 20B
parameters. Finally, we apply FLAN instruction tuning to the UL2 20B model,
achieving MMLU and Big-Bench scores competitive to FLAN-PaLM 62B. We release
Flax-based T5X checkpoints for the UL2 20B & Flan-UL2 20B.


6. [A Generalist Agent](http://arxiv.org/abs/2205.06175v3), Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, Nando de Freitas, 12-05-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Inspired by progress in large-scale language modeling, we apply a similar
approach towards building a single generalist agent beyond the realm of text
outputs. The agent, which we refer to as Gato, works as a multi-modal,
multi-task, multi-embodiment generalist policy. The same network with the same
weights can play Atari, caption images, chat, stack blocks with a real robot
arm and much more, deciding based on its context whether to output text, joint
torques, button presses, or other tokens. In this report we describe the model
and the data, and document the current capabilities of Gato.


6. [Selection-Inference: Exploiting Large Language Models for Interpretable
  Logical Reasoning](http://arxiv.org/abs/2205.09712v1), Antonia Creswell, Murray Shanahan, Irina Higgins, 19-05-2022
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    Large language models (LLMs) have been shown to be capable of impressive
few-shot generalisation to new tasks. However, they still tend to perform
poorly on multi-step logical reasoning problems. Here we carry out a
comprehensive evaluation of LLMs on 50 tasks that probe different aspects of
logical reasoning. We show that language models tend to perform fairly well at
single step inference or entailment tasks, but struggle to chain together
multiple reasoning steps to solve more complex problems. In light of this, we
propose a Selection-Inference (SI) framework that exploits pre-trained LLMs as
general processing modules, and alternates between selection and inference to
generate a series of interpretable, casual reasoning steps leading to the final
answer. We show that a 7B parameter LLM used within the SI framework in a
5-shot generalisation setting, with no fine-tuning, yields a performance
improvement of over 100% compared to an equivalent vanilla baseline on a suite
of 10 logical reasoning tasks. The same model in the same setting even
outperforms a significantly larger 280B parameter baseline on the same suite of
tasks. Moreover, answers produced by the SI framework are accompanied by a
causal natural-language-based reasoning trace, which has important implications
for the safety and trustworthiness of the system.


6. [Least-to-Most Prompting Enables Complex Reasoning in Large Language
  Models](http://arxiv.org/abs/2205.10625v3), Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi, 21-05-2022
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    Chain-of-thought prompting has demonstrated remarkable performance on various
natural language reasoning tasks. However, it tends to perform poorly on tasks
which requires solving problems harder than the exemplars shown in the prompts.
To overcome this challenge of easy-to-hard generalization, we propose a novel
prompting strategy, least-to-most prompting. The key idea in this strategy is
to break down a complex problem into a series of simpler subproblems and then
solve them in sequence. Solving each subproblem is facilitated by the answers
to previously solved subproblems. Our experimental results on tasks related to
symbolic manipulation, compositional generalization, and math reasoning reveal
that least-to-most prompting is capable of generalizing to more difficult
problems than those seen in the prompts. A notable finding is that when the
GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve
the compositional generalization benchmark SCAN in any split (including length
split) with an accuracy of at least 99% using just 14 exemplars, compared to
only 16% accuracy with chain-of-thought prompting. This is particularly
noteworthy because neural-symbolic models in the literature that specialize in
solving SCAN are trained on the entire training set containing over 15,000
examples. We have included prompts for all the tasks in the Appendix.


6. [FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness](http://arxiv.org/abs/2205.14135v2), Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, 27-05-2022
     ### Categories
     Machine Learning
    ### Abstract
    Transformers are slow and memory-hungry on long sequences, since the time and
memory complexity of self-attention are quadratic in sequence length.
Approximate attention methods have attempted to address this problem by trading
off model quality to reduce the compute complexity, but often do not achieve
wall-clock speedup. We argue that a missing principle is making attention
algorithms IO-aware -- accounting for reads and writes between levels of GPU
memory. We propose FlashAttention, an IO-aware exact attention algorithm that
uses tiling to reduce the number of memory reads/writes between GPU high
bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of
FlashAttention, showing that it requires fewer HBM accesses than standard
attention, and is optimal for a range of SRAM sizes. We also extend
FlashAttention to block-sparse attention, yielding an approximate attention
algorithm that is faster than any existing approximate attention method.
FlashAttention trains Transformers faster than existing baselines: 15%
end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the
MLPerf 1.1 training speed record, 3$\times$ speedup on GPT-2 (seq. length 1K),
and 2.4$\times$ speedup on long-range arena (seq. length 1K-4K). FlashAttention
and block-sparse FlashAttention enable longer context in Transformers, yielding
higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on
long-document classification) and entirely new capabilities: the first
Transformers to achieve better-than-chance performance on the Path-X challenge
(seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1%
accuracy).


6. [Making Large Language Models Better Reasoners with Step-Aware Verifier](http://arxiv.org/abs/2206.02336v3), Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, Weizhu Chen, 06-06-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Few-shot learning is a challenging task that requires language models to
generalize from limited examples. Large language models like GPT-3 and PaLM
have made impressive progress in this area, but they still face difficulties in
reasoning tasks such as GSM8K, a benchmark for arithmetic problems. To improve
their reasoning skills, previous work has proposed to guide the language model
with prompts that elicit a series of reasoning steps before giving the final
answer, achieving a significant improvement on GSM8K from 17.9% to 58.1% in
problem-solving rate. In this paper, we present DIVERSE (Diverse Verifier on
Reasoning Step), a novel approach that further enhances the reasoning
capability of language models. DIVERSE has three main components: first, it
generates diverse prompts to explore different reasoning paths for the same
question; second, it uses a verifier to filter out incorrect answers based on a
weighted voting scheme; and third, it verifies each reasoning step individually
instead of the whole chain. We evaluate DIVERSE on the latest language model
code-davinci-002 and show that it achieves new state-of-the-art results on six
of eight reasoning benchmarks (e.g., GSM8K 74.4% to 83.2%).


6. [From Human Days to Machine Seconds: Automatically Answering and
  Generating Machine Learning Final Exams](http://arxiv.org/abs/2206.05442v7), Iddo Drori, Sarah J. Zhang, Reece Shuttleworth, Sarah Zhang, Keith Tyser, Zad Chin, Pedro Lantigua, Saisamrit Surbehera, Gregory Hunter, Derek Austin, Leonard Tang, Yann Hicke, Sage Simhon, Sathwik Karnik, Darnell Granberry, Madeleine Udell, 11-06-2022
     ### Categories
     Machine Learning
    ### Abstract
    A final exam in machine learning at a top institution such as MIT, Harvard,
or Cornell typically takes faculty days to write, and students hours to solve.
We demonstrate that large language models pass machine learning finals at a
human level, on finals available online after the models were trained, and
automatically generate new human-quality final exam questions in seconds.
Previous work has developed program synthesis and few-shot learning methods to
solve university-level problem set questions in mathematics and STEM courses.
In this work, we develop and compare methods that solve final exams, which
differ from problem sets in several ways: the questions are longer, have
multiple parts, are more complicated, and span a broader set of topics. We
curate a dataset and benchmark of questions from machine learning final exams
available online and code for answering these questions and generating new
questions. We show how to generate new questions from other questions and
course notes. For reproducibility and future research on this final exam
benchmark, we use automatic checkers for multiple-choice, numeric, and
questions with expression answers. We perform ablation studies comparing
zero-shot learning with few-shot learning and chain-of-thought prompting using
GPT-3, OPT, Codex, and ChatGPT across machine learning topics and find that
few-shot learning methods perform best. We highlight the transformative
potential of language models to streamline the writing and solution of
large-scale assessments, significantly reducing the workload from human days to
mere machine seconds. Our results suggest that rather than banning large
language models such as ChatGPT in class, instructors should teach students to
harness them by asking students meta-questions about correctness, completeness,
and originality of the responses generated, encouraging critical thinking in
academic studies.


6. [Emergent Abilities of Large Language Models](http://arxiv.org/abs/2206.07682v2), Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus, 15-06-2022
     ### Categories
     Computation and Language
    ### Abstract
    Scaling up language models has been shown to predictably improve performance
and sample efficiency on a wide range of downstream tasks. This paper instead
discusses an unpredictable phenomenon that we refer to as emergent abilities of
large language models. We consider an ability to be emergent if it is not
present in smaller models but is present in larger models. Thus, emergent
abilities cannot be predicted simply by extrapolating the performance of
smaller models. The existence of such emergence implies that additional scaling
could further expand the range of capabilities of language models.


6. [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale
  Knowledge](http://arxiv.org/abs/2206.08853v2), Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, Anima Anandkumar, 17-06-2022
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Autonomous agents have made great strides in specialist domains like Atari
games and Go. However, they typically learn tabula rasa in isolated
environments with limited and manually conceived objectives, thus failing to
generalize across a wide spectrum of tasks and capabilities. Inspired by how
humans continually learn and adapt in the open world, we advocate a trinity of
ingredients for building generalist agents: 1) an environment that supports a
multitude of tasks and goals, 2) a large-scale database of multimodal
knowledge, and 3) a flexible and scalable agent architecture. We introduce
MineDojo, a new framework built on the popular Minecraft game that features a
simulation suite with thousands of diverse open-ended tasks and an
internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and
forum discussions. Using MineDojo's data, we propose a novel agent learning
algorithm that leverages large pre-trained video-language models as a learned
reward function. Our agent is able to solve a variety of open-ended tasks
specified in free-form language without any manually designed dense shaping
reward. We open-source the simulation suite, knowledge bases, algorithm
implementation, and pretrained models (https://minedojo.org) to promote
research towards the goal of generally capable embodied agents.


6. [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online
  Videos](http://arxiv.org/abs/2206.11795v1), Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, Jeff Clune, 23-06-2022
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    Pretraining on noisy, internet-scale datasets has been heavily studied as a
technique for training models with broad, general capabilities for text,
images, and other modalities. However, for many sequential decision domains
such as robotics, video games, and computer use, publicly available data does
not contain the labels required to train behavioral priors in the same way. We
extend the internet-scale pretraining paradigm to sequential decision domains
through semi-supervised imitation learning wherein agents learn to act by
watching online unlabeled videos. Specifically, we show that with a small
amount of labeled data we can train an inverse dynamics model accurate enough
to label a huge unlabeled source of online data -- here, online videos of
people playing Minecraft -- from which we can then train a general behavioral
prior. Despite using the native human interface (mouse and keyboard at 20Hz),
we show that this behavioral prior has nontrivial zero-shot capabilities and
that it can be fine-tuned, with both imitation learning and reinforcement
learning, to hard-exploration tasks that are impossible to learn from scratch
via reinforcement learning. For many tasks our models exhibit human-level
performance, and we are the first to report computer agents that can craft
diamond tools, which can take proficient humans upwards of 20 minutes (24,000
environment actions) of gameplay to accomplish.


6. [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language,
  Vision, and Action](http://arxiv.org/abs/2207.04429v2), Dhruv Shah, Blazej Osinski, Brian Ichter, Sergey Levine, 10-07-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Goal-conditioned policies for robotic navigation can be trained on large,
unannotated datasets, providing for good generalization to real-world settings.
However, particularly in vision-based settings where specifying goals requires
an image, this makes for an unnatural interface. Language provides a more
convenient modality for communication with robots, but contemporary methods
typically require expensive supervision, in the form of trajectories annotated
with language descriptions. We present a system, LM-Nav, for robotic navigation
that enjoys the benefits of training on unannotated large datasets of
trajectories, while still providing a high-level interface to the user. Instead
of utilizing a labeled instruction following dataset, we show that such a
system can be constructed entirely out of pre-trained models for navigation
(ViNG), image-language association (CLIP), and language modeling (GPT-3),
without requiring any fine-tuning or language-annotated robot data. We
instantiate LM-Nav on a real-world mobile robot and demonstrate long-horizon
navigation through complex, outdoor environments from natural language
instructions. For videos of our experiments, code release, and an interactive
Colab notebook that runs in your browser, please check out our project page
https://sites.google.com/view/lmnav


6. [Inner Monologue: Embodied Reasoning through Planning with Language
  Models](http://arxiv.org/abs/2207.05608v1), Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, Brian Ichter, 12-07-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Recent works have shown how the reasoning capabilities of Large Language
Models (LLMs) can be applied to domains beyond natural language processing,
such as planning and interaction for robots. These embodied problems require an
agent to understand many semantic aspects of the world: the repertoire of
skills available, how these skills influence the world, and how changes to the
world map back to the language. LLMs planning in embodied environments need to
consider not just what skills to do, but also how and when to do them - answers
that change over time in response to the agent's own choices. In this work, we
investigate to what extent LLMs used in such embodied contexts can reason over
sources of feedback provided through natural language, without any additional
training. We propose that by leveraging environment feedback, LLMs are able to
form an inner monologue that allows them to more richly process and plan in
robotic control scenarios. We investigate a variety of sources of feedback,
such as success detection, scene description, and human interaction. We find
that closed-loop language feedback significantly improves high-level
instruction completion on three domains, including simulated and real table top
rearrangement tasks and long-horizon mobile manipulation tasks in a kitchen
environment in the real world.


6. [BlenderBot 3: a deployed conversational agent that continually learns to
  responsibly engage](http://arxiv.org/abs/2208.03188v3), Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, Morteza Behrooz, William Ngan, Spencer Poff, Naman Goyal, Arthur Szlam, Y-Lan Boureau, Melanie Kambadur, Jason Weston, 05-08-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We present BlenderBot 3, a 175B parameter dialogue model capable of
open-domain conversation with access to the internet and a long-term memory,
and having been trained on a large number of user defined tasks. We release
both the model weights and code, and have also deployed the model on a public
web page to interact with organic users. This technical report describes how
the model was built (architecture, model and training scheme), and details of
its deployment, including safety mechanisms. Human evaluations show its
superiority to existing open-domain dialogue agents, including its predecessors
(Roller et al., 2021; Komeili et al., 2022). Finally, we detail our plan for
continual learning using the data collected from deployment, which will also be
publicly released. The goal of this research program is thus to enable the
community to study ever-improving responsible agents that learn through
interaction.


6. [Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango](http://arxiv.org/abs/2209.07686v2), Aman Madaan, Amir Yazdanbakhsh, 16-09-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    The past decade has witnessed dramatic gains in natural language processing
and an unprecedented scaling of large language models. These developments have
been accelerated by the advent of few-shot techniques such as chain of thought
(CoT) prompting. Specifically, CoT pushes the performance of large language
models in a few-shot setup by augmenting the prompts with intermediate steps.
Despite impressive results across various tasks, the reasons behind their
success have not been explored. This work uses counterfactual prompting to
develop a deeper understanding of CoT-based few-shot prompting mechanisms in
large language models. We first systematically identify and define the key
components of a prompt: symbols, patterns, and text. Then, we devise and
conduct an exhaustive set of experiments across four different tasks, by
querying the model with counterfactual prompts where only one of these
components is altered. Our experiments across three models (PaLM, GPT-3, and
CODEX) reveal several surprising findings and brings into question the
conventional wisdom around few-shot prompting. First, the presence of factual
patterns in a prompt is practically immaterial to the success of CoT. Second,
our results conclude that the primary role of intermediate steps may not be to
facilitate learning how to solve a task. The intermediate steps are rather a
beacon for the model to realize what symbols to replicate in the output to form
a factual answer. Further, text imbues patterns with commonsense knowledge and
meaning. Our empirical and qualitative analysis reveals that a symbiotic
relationship between text and patterns explains the success of few-shot
prompting: text helps extract commonsense from the question to help patterns,
and patterns enforce task understanding and direct text generation.


6. [Learn to Explain: Multimodal Reasoning via Thought Chains for Science
  Question Answering](http://arxiv.org/abs/2209.09513v2), Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan, 20-09-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    When answering a question, humans utilize the information available across
different modalities to synthesize a consistent and complete chain of thought
(CoT). This process is normally a black box in the case of deep learning models
like large-scale language models. Recently, science question benchmarks have
been used to diagnose the multi-hop reasoning ability and interpretability of
an AI system. However, existing datasets fail to provide annotations for the
answers, or are restricted to the textual-only modality, small scales, and
limited domain diversity. To this end, we present Science Question Answering
(ScienceQA), a new benchmark that consists of ~21k multimodal multiple choice
questions with a diverse set of science topics and annotations of their answers
with corresponding lectures and explanations. We further design language models
to learn to generate lectures and explanations as the chain of thought (CoT) to
mimic the multi-hop reasoning process when answering ScienceQA questions.
ScienceQA demonstrates the utility of CoT in language models, as CoT improves
the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in
fine-tuned UnifiedQA. We also explore the upper bound for models to leverage
explanations by feeding those in the input; we observe that it improves the
few-shot performance of GPT-3 by 18.96%. Our analysis further shows that
language models, similar to humans, benefit from explanations to learn from
fewer data and achieve the same performance with just 40% of the data. The data
and code are available at https://scienceqa.github.io.


6. [ProgPrompt: Generating Situated Robot Task Plans using Large Language
  Models](http://arxiv.org/abs/2209.11302v1), Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg, 22-09-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Task planning can require defining myriad domain knowledge about the world in
which a robot needs to act. To ameliorate that effort, large language models
(LLMs) can be used to score potential next actions during task planning, and
even generate action sequences directly, given an instruction in natural
language with no additional domain information. However, such methods either
require enumerating all possible next steps for scoring, or generate free-form
text that may contain actions not possible on a given robot in its current
context. We present a programmatic LLM prompt structure that enables plan
generation functional across situated environments, robot capabilities, and
tasks. Our key insight is to prompt the LLM with program-like specifications of
the available actions and objects in an environment, as well as with example
programs that can be executed. We make concrete recommendations about prompt
structure and generation constraints through ablation experiments, demonstrate
state of the art success rates in VirtualHome household tasks, and deploy our
method on a physical robot arm for tabletop tasks. Website at
progprompt.github.io


6. [Promptagator: Few-shot Dense Retrieval From 8 Examples](http://arxiv.org/abs/2209.11755v1), Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, Ming-Wei Chang, 23-09-2022
     ### Categories
     Computation and Language
    ### Abstract
    Much recent research on information retrieval has focused on how to transfer
from one task (typically with abundant supervised data) to various other tasks
where supervision is limited, with the implicit assumption that it is possible
to generalize from one task to all the rest. However, this overlooks the fact
that there are many diverse and unique retrieval tasks, each targeting
different search intents, queries, and search domains. In this paper, we
suggest to work on Few-shot Dense Retrieval, a setting where each task comes
with a short description and a few examples. To amplify the power of a few
examples, we propose Prompt-base Query Generation for Retriever (Promptagator),
which leverages large language models (LLM) as a few-shot query generator, and
creates task-specific retrievers based on the generated data. Powered by LLM's
generalization ability, Promptagator makes it possible to create task-specific
end-to-end retrievers solely based on a few examples {without} using Natural
Questions or MS MARCO to train %question generators or dual encoders.
Surprisingly, LLM prompting with no more than 8 examples allows dual encoders
to outperform heavily engineered models trained on MS MARCO like ColBERT v2 by
more than 1.2 nDCG on average on 11 retrieval sets. Further training
standard-size re-rankers using the same generated data yields another 5.0 point
nDCG improvement. Our studies determine that query generation can be far more
effective than previously observed, especially when a small amount of
task-specific knowledge is given.


6. [Ask Me Anything: A simple strategy for prompting language models](http://arxiv.org/abs/2210.02441v3), Simran Arora, Avanika Narayan, Mayee F. Chen, Laurel Orr, Neel Guha, Kush Bhatia, Ines Chami, Frederic Sala, Christopher Ré, 05-10-2022
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) transfer well to new tasks out-of-the-box simply
given a natural language prompt that demonstrates how to perform the task and
no additional training. Prompting is a brittle process wherein small
modifications to the prompt can cause large variations in the model
predictions, and therefore significant effort is dedicated towards designing a
painstakingly "perfect prompt" for a task. To mitigate the high degree of
effort involved in prompt-design, we instead ask whether producing multiple
effective, yet imperfect, prompts and aggregating them can lead to a high
quality prompting strategy. Our observations motivate our proposed prompting
method, ASK ME ANYTHING (AMA). We first develop an understanding of the
effective prompt formats, finding that question-answering (QA) prompts, which
encourage open-ended generation ("Who went to the park?") tend to outperform
those that restrict the model outputs ("John went to the park. Output True or
False."). Our approach recursively uses the LLM itself to transform task inputs
to the effective QA format. We apply the collected prompts to obtain several
noisy votes for the input's true label. We find that the prompts can have very
different accuracies and complex dependencies and thus propose to use weak
supervision, a procedure for combining the noisy predictions, to produce the
final predictions for the inputs. We evaluate AMA across open-source model
families (e.g., EleutherAI, BLOOM, OPT, and T0) and model sizes (125M-175B
parameters), demonstrating an average performance lift of 10.2% over the
few-shot baseline. This simple strategy enables the open-source GPT-J-6B model
to match and exceed the performance of few-shot GPT3-175B on 15 of 20 popular
benchmarks. Averaged across these tasks, the GPT-J-6B model outperforms
few-shot GPT3-175B. We release our code here:
https://github.com/HazyResearch/ama_prompting


6. [Language Models are Multilingual Chain-of-Thought Reasoners](http://arxiv.org/abs/2210.03057v1), Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei, 06-10-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    We evaluate the reasoning abilities of large language models in multilingual
settings. We introduce the Multilingual Grade School Math (MGSM) benchmark, by
manually translating 250 grade-school math problems from the GSM8K dataset
(Cobbe et al., 2021) into ten typologically diverse languages. We find that the
ability to solve MGSM problems via chain-of-thought prompting emerges with
increasing model scale, and that models have strikingly strong multilingual
reasoning abilities, even in underrepresented languages such as Bengali and
Swahili. Finally, we show that the multilingual reasoning abilities of language
models extend to other tasks such as commonsense reasoning and word-in-context
semantic judgment. The MGSM benchmark is publicly available at
https://github.com/google-research/url-nlp.


6. [ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629v3), Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao, 06-10-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    While large language models (LLMs) have demonstrated impressive capabilities
across tasks in language understanding and interactive decision making, their
abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g.
action plan generation) have primarily been studied as separate topics. In this
paper, we explore the use of LLMs to generate both reasoning traces and
task-specific actions in an interleaved manner, allowing for greater synergy
between the two: reasoning traces help the model induce, track, and update
action plans as well as handle exceptions, while actions allow it to interface
with external sources, such as knowledge bases or environments, to gather
additional information. We apply our approach, named ReAct, to a diverse set of
language and decision making tasks and demonstrate its effectiveness over
state-of-the-art baselines, as well as improved human interpretability and
trustworthiness over methods without reasoning or acting components.
Concretely, on question answering (HotpotQA) and fact verification (Fever),
ReAct overcomes issues of hallucination and error propagation prevalent in
chain-of-thought reasoning by interacting with a simple Wikipedia API, and
generates human-like task-solving trajectories that are more interpretable than
baselines without reasoning traces. On two interactive decision making
benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and
reinforcement learning methods by an absolute success rate of 34% and 10%
respectively, while being prompted with only one or two in-context examples.
Project site with code: https://react-lm.github.io


6. [Automatic Chain of Thought Prompting in Large Language Models](http://arxiv.org/abs/2210.03493v1), Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola, 07-10-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) can perform complex reasoning by generating
intermediate reasoning steps. Providing these steps for prompting
demonstrations is called chain-of-thought (CoT) prompting. CoT prompting has
two major paradigms. One leverages a simple prompt like "Let's think step by
step" to facilitate step-by-step thinking before answering a question. The
other uses a few manual demonstrations one by one, each composed of a question
and a reasoning chain that leads to an answer. The superior performance of the
second paradigm hinges on the hand-crafting of task-specific demonstrations one
by one. We show that such manual efforts may be eliminated by leveraging LLMs
with the "Let's think step by step" prompt to generate reasoning chains for
demonstrations one by one, i.e., let's think not just step by step, but also
one by one. However, these generated chains often come with mistakes. To
mitigate the effect of such mistakes, we find that diversity matters for
automatically constructing demonstrations. We propose an automatic CoT
prompting method: Auto-CoT. It samples questions with diversity and generates
reasoning chains to construct demonstrations. On ten public benchmark reasoning
tasks with GPT-3, Auto-CoT consistently matches or exceeds the performance of
the CoT paradigm that requires manual designs of demonstrations. Code is
available at https://github.com/amazon-research/auto-cot


6. [Interactive Language: Talking to Robots in Real Time](http://arxiv.org/abs/2210.06407v1), Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong, Pete Florence, 12-10-2022
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    We present a framework for building interactive, real-time, natural
language-instructable robots in the real world, and we open source related
assets (dataset, environment, benchmark, and policies). Trained with behavioral
cloning on a dataset of hundreds of thousands of language-annotated
trajectories, a produced policy can proficiently execute an order of magnitude
more commands than previous works: specifically we estimate a 93.5% success
rate on a set of 87,000 unique natural language strings specifying raw
end-to-end visuo-linguo-motor skills in the real world. We find that the same
policy is capable of being guided by a human via real-time language to address
a wide range of precise long-horizon rearrangement goals, e.g. "make a smiley
face out of blocks". The dataset we release comprises nearly 600,000
language-labeled trajectories, an order of magnitude larger than prior
available datasets. We hope the demonstrated results and associated assets
enable further advancement of helpful, capable, natural-language-interactable
robots. See videos at https://interactive-language.github.io.


6. [Language Models of Code are Few-Shot Commonsense Learners](http://arxiv.org/abs/2210.07128v3), Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, Graham Neubig, 13-10-2022
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    We address the general task of structured commonsense reasoning: given a
natural language input, the goal is to generate a graph such as an event -- or
a reasoning-graph. To employ large language models (LMs) for this task,
existing approaches ``serialize'' the output graph as a flat list of nodes and
edges. Although feasible, these serialized graphs strongly deviate from the
natural language corpora that LMs were pre-trained on, hindering LMs from
generating them correctly. In this paper, we show that when we instead frame
structured commonsense reasoning tasks as code generation tasks, pre-trained
LMs of code are better structured commonsense reasoners than LMs of natural
language, even when the downstream task does not involve source code at all. We
demonstrate our approach across three diverse structured commonsense reasoning
tasks. In all these natural language tasks, we show that using our approach, a
code generation LM (CODEX) outperforms natural-LMs that are fine-tuned on the
target task (e.g., T5) and other strong LMs such as GPT-3 in the few-shot
setting.


6. [Crosslingual Generalization through Multitask Finetuning](http://arxiv.org/abs/2211.01786v2), Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, Colin Raffel, 03-11-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Multitask prompted finetuning (MTF) has been shown to help large language
models generalize to new tasks in a zero-shot setting, but so far explorations
of MTF have focused on English data and models. We apply MTF to the pretrained
multilingual BLOOM and mT5 model families to produce finetuned variants called
BLOOMZ and mT0. We find finetuning large multilingual language models on
English tasks with English prompts allows for task generalization to
non-English languages that appear only in the pretraining corpus. Finetuning on
multilingual tasks with English prompts further improves performance on English
and non-English tasks leading to various state-of-the-art zero-shot results. We
also investigate finetuning on multilingual tasks with prompts that have been
machine-translated from English to match the language of each dataset. We find
training on these machine-translated prompts leads to better performance on
human-written prompts in the respective languages. Surprisingly, we find models
are capable of zero-shot generalization to tasks in languages they have never
intentionally seen. We conjecture that the models are learning higher-level
capabilities that are both task- and language-agnostic. In addition, we
introduce xP3, a composite of supervised datasets in 46 languages with English
and machine-translated prompts. Our code, datasets and models are freely
available at https://github.com/bigscience-workshop/xmtf.


6. [Large Language Models Are Human-Level Prompt Engineers](http://arxiv.org/abs/2211.01910v2), Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba, 03-11-2022
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    By conditioning on natural language instructions, large language models
(LLMs) have displayed impressive capabilities as general-purpose computers.
However, task performance depends significantly on the quality of the prompt
used to steer the model, and most effective prompts have been handcrafted by
humans. Inspired by classical program synthesis and the human approach to
prompt engineering, we propose Automatic Prompt Engineer (APE) for automatic
instruction generation and selection. In our method, we treat the instruction
as the "program," optimized by searching over a pool of instruction candidates
proposed by an LLM in order to maximize a chosen score function. To evaluate
the quality of the selected instruction, we evaluate the zero-shot performance
of another LLM following the selected instruction. Experiments on 24 NLP tasks
show that our automatically generated instructions outperform the prior LLM
baseline by a large margin and achieve better or comparable performance to the
instructions generated by human annotators on 19/24 tasks. We conduct extensive
qualitative and quantitative analyses to explore the performance of APE. We
show that APE-engineered prompts can be applied to steer models toward
truthfulness and/or informativeness, as well as to improve few-shot learning
performance by simply prepending them to standard in-context learning prompts.
Please check out our webpage at
https://sites.google.com/view/automatic-prompt-engineer.


6. [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](http://arxiv.org/abs/2211.05100v4), BigScience Workshop,  :, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo González Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jörg Frohberg, Joseph Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber, Long Phan, Loubna Ben allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario Šaško, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Mohammad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis López, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Davut Emre Taşar, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiangru Tang, Zheng-Xin Yong, Zhiqing Sun, Shaked Brody, Yallow Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre François Lavallée, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aurélie Névéol, Charles Lovering, Dan Garrette, Deepak Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, Jan-Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura, Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, Shani Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdeněk Kasner, Alice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena, Carlos Muñoz Ferrandis, Daniel McDuff, Danish Contractor, David Lansky, Davis David, Douwe Kiela, Duong A. Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatima Mirza, Frankline Ononiwu, Habib Rezanejad, Hessie Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jesse Passmore, Josh Seltzer, Julio Bonis Sanz, Livia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nour Fahmy, Olanrewaju Samuel, Ran An, Rasmus Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas Wang, Sourav Roy, Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le, Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada, Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel León Periñán, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Imane Bello, Ishani Dash, Jihyun Kang, John Giorgi, Jonas Golde, Jose David Posada, Karthik Rangasai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, Maria A Castillo, Marianna Nezhurina, Mario Sänger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Bharati, Tanmay Laud, Théo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yash Shailesh Bajaj, Yash Venkatraman, Yifan Xu, Yingxin Xu, Yu Xu, Zhe Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, Thomas Wolf, 09-11-2022
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) have been shown to be able to perform new tasks
based on a few demonstrations or natural language instructions. While these
capabilities have led to widespread adoption, most LLMs are developed by
resource-rich organizations and are frequently kept from the public. As a step
towards democratizing this powerful technology, we present BLOOM, a
176B-parameter open-access language model designed and built thanks to a
collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer
language model that was trained on the ROOTS corpus, a dataset comprising
hundreds of sources in 46 natural and 13 programming languages (59 in total).
We find that BLOOM achieves competitive performance on a wide variety of
benchmarks, with stronger results after undergoing multitask prompted
finetuning. To facilitate future research and applications using LLMs, we
publicly release our models and code under the Responsible AI License.


6. [Ignore Previous Prompt: Attack Techniques For Language Models](http://arxiv.org/abs/2211.09527v1), Fábio Perez, Ian Ribeiro, 17-11-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Transformer-based large language models (LLMs) provide a powerful foundation
for natural language tasks in large-scale customer-facing applications.
However, studies that explore their vulnerabilities emerging from malicious
user interaction are scarce. By proposing PromptInject, a prosaic alignment
framework for mask-based iterative adversarial prompt composition, we examine
how GPT-3, the most widely deployed language model in production, can be easily
misaligned by simple handcrafted inputs. In particular, we investigate two
types of attacks -- goal hijacking and prompt leaking -- and demonstrate that
even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit
GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject
is available at https://github.com/agencyenterprise/PromptInject.


6. [PAL: Program-aided Language Models](http://arxiv.org/abs/2211.10435v2), Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig, 18-11-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) have recently demonstrated an impressive ability
to perform arithmetic and symbolic reasoning tasks, when provided with a few
examples at test time ("few-shot prompting"). Much of this success can be
attributed to prompting methods such as "chain-of-thought'', which employ LLMs
for both understanding the problem description by decomposing it into steps, as
well as solving each step of the problem. While LLMs seem to be adept at this
sort of step-by-step decomposition, LLMs often make logical and arithmetic
mistakes in the solution part, even when the problem is decomposed correctly.
In this paper, we present Program-Aided Language models (PAL): a novel approach
that uses the LLM to read natural language problems and generate programs as
the intermediate reasoning steps, but offloads the solution step to a runtime
such as a Python interpreter. With PAL, decomposing the natural language
problem into runnable steps remains the only learning task for the LLM, while
solving is delegated to the interpreter. We demonstrate this synergy between a
neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and
algorithmic reasoning tasks from BIG-Bench Hard and other benchmarks. In all
these natural language reasoning tasks, generating code using an LLM and
reasoning using a Python interpreter leads to more accurate results than much
larger models. For example, PAL using Codex achieves state-of-the-art few-shot
accuracy on the GSM8K benchmark of math word problems, surpassing PaLM-540B
which uses chain-of-thought by absolute 15% top-1. Our code and data are
publicly available at http://reasonwithpal.com/ .


6. [Program of Thoughts Prompting: Disentangling Computation from Reasoning
  for Numerical Reasoning Tasks](http://arxiv.org/abs/2211.12588v4), Wenhu Chen, Xueguang Ma, Xinyi Wang, William W. Cohen, 22-11-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Recently, there has been significant progress in teaching language models to
perform step-by-step reasoning to solve complex numerical reasoning tasks.
Chain-of-thoughts prompting (CoT) is by far the state-of-art method for these
tasks. CoT uses language models to perform both reasoning and computation in
the multi-step `thought' process. To disentangle computation from reasoning, we
propose `Program of Thoughts' (PoT), which uses language models (mainly Codex)
to express the reasoning process as a program. The computation is relegated to
an external computer, which executes the generated programs to derive the
answer. We evaluate PoT on five math word problem datasets (GSM, AQuA, SVAMP,
TabMWP, MultiArith) and three financial-QA datasets (FinQA, ConvFinQA, TATQA)
for both few-shot and zero-shot setups. Under both few-shot and zero-shot
settings, PoT can show an average performance gain over CoT by around 12\%
across all the evaluated datasets. By combining PoT with self-consistency
decoding, we can achieve SoTA performance on all math problem datasets and
near-SoTA performance on financial datasets. All of our data and code are
released in Github https://github.com/wenhuchen/Program-of-Thoughts


6. [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large
  Language Models](http://arxiv.org/abs/2212.04088v3), Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, Yu Su, 08-12-2022
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    This study focuses on using large language models (LLMs) as a planner for
embodied agents that can follow natural language instructions to complete
complex tasks in a visually-perceived environment. The high data cost and poor
sample efficiency of existing methods hinders the development of versatile
agents that are capable of many tasks and can learn new tasks quickly. In this
work, we propose a novel method, LLM-Planner, that harnesses the power of large
language models to do few-shot planning for embodied agents. We further propose
a simple but effective way to enhance LLMs with physical grounding to generate
and update plans that are grounded in the current environment. Experiments on
the ALFRED dataset show that our method can achieve very competitive few-shot
performance: Despite using less than 0.5% of paired training data, LLM-Planner
achieves competitive performance with recent baselines that are trained using
the full training data. Existing methods can barely complete any task
successfully under the same few-shot setting. Our work opens the door for
developing versatile and sample-efficient embodied agents that can quickly
learn many tasks. Website: https://dki-lab.github.io/LLM-Planner


6. [Constitutional AI: Harmlessness from AI Feedback](http://arxiv.org/abs/2212.08073v1), Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan, 15-12-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    As AI systems become more capable, we would like to enlist their help to
supervise other AIs. We experiment with methods for training a harmless AI
assistant through self-improvement, without any human labels identifying
harmful outputs. The only human oversight is provided through a list of rules
or principles, and so we refer to the method as 'Constitutional AI'. The
process involves both a supervised learning and a reinforcement learning phase.
In the supervised phase we sample from an initial model, then generate
self-critiques and revisions, and then finetune the original model on revised
responses. In the RL phase, we sample from the finetuned model, use a model to
evaluate which of the two samples is better, and then train a preference model
from this dataset of AI preferences. We then train with RL using the preference
model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a
result we are able to train a harmless but non-evasive AI assistant that
engages with harmful queries by explaining its objections to them. Both the SL
and RL methods can leverage chain-of-thought style reasoning to improve the
human-judged performance and transparency of AI decision making. These methods
make it possible to control AI behavior more precisely and with far fewer human
labels.


6. [Reasoning with Language Model Prompting: A Survey](http://arxiv.org/abs/2212.09597v8), Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, Huajun Chen, 19-12-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Reasoning, as an essential ability for complex problem-solving, can provide
back-end support for various real-world applications, such as medical
diagnosis, negotiation, etc. This paper provides a comprehensive survey of
cutting-edge research on reasoning with language model prompting. We introduce
research works with comparisons and summaries and provide systematic resources
to help beginners. We also discuss the potential reasons for emerging such
reasoning abilities and highlight future research directions. Resources are
available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated
periodically).


6. [KronA: Parameter Efficient Tuning with Kronecker Adapter](http://arxiv.org/abs/2212.10650v1), Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J. Clark, Mehdi Rezagholizadeh, 20-12-2022
     ### Categories
     Computation and Language
    ### Abstract
    Fine-tuning a Pre-trained Language Model (PLM) on a specific downstream task
has been a well-known paradigm in Natural Language Processing. However, with
the ever-growing size of PLMs, training the entire model on several downstream
tasks becomes very expensive and resource-hungry. Recently, different Parameter
Efficient Tuning (PET) techniques are proposed to improve the efficiency of
fine-tuning PLMs. One popular category of PET methods is the low-rank
adaptation methods which insert learnable truncated SVD modules into the
original model either sequentially or in parallel. However, low-rank
decomposition suffers from limited representation power. In this work, we
address this problem using the Kronecker product instead of the low-rank
representation. We introduce KronA, a Kronecker product-based adapter module
for efficient fine-tuning of Transformer-based PLMs. We apply the proposed
methods for fine-tuning T5 on the GLUE benchmark to show that incorporating the
Kronecker-based modules can outperform state-of-the-art PET methods.


6. [Large Language Models Are Reasoning Teachers](http://arxiv.org/abs/2212.10071v2), Namgyu Ho, Laura Schmid, Se-Young Yun, 20-12-2022
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Recent works have shown that chain-of-thought (CoT) prompting can elicit
language models to solve complex reasoning tasks, step-by-step. However,
prompt-based CoT methods are dependent on very large models such as GPT-3 175B
which are prohibitive to deploy at scale. In this paper, we use these large
models as reasoning teachers to enable complex reasoning in smaller models and
reduce model size requirements by several orders of magnitude. We propose
Fine-tune-CoT, a method that generates reasoning samples from very large
teacher models to fine-tune smaller models. We evaluate our method on a wide
range of public models and complex tasks. We find that Fine-tune-CoT enables
substantial reasoning capability in small models, far outperforming
prompt-based baselines and even the teacher model in many tasks. Additionally,
we extend our method by leveraging the teacher model's ability to generate
multiple distinct rationales for each original sample. Enriching the
fine-tuning data with such diverse reasoning results in a substantial
performance boost across datasets, even for very small models. We conduct
ablations and sample studies to understand the emergence of reasoning
capabilities of student models. Our code implementation and data are available
at https://github.com/itsnamgyu/reasoning-teacher.


6. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](http://arxiv.org/abs/2212.10560v2), Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi, 20-12-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large "instruction-tuned" language models (i.e., finetuned to respond to
instructions) have demonstrated a remarkable ability to generalize zero-shot to
new tasks. Nevertheless, they depend heavily on human-written instruction data
that is often limited in quantity, diversity, and creativity, therefore
hindering the generality of the tuned model. We introduce Self-Instruct, a
framework for improving the instruction-following capabilities of pretrained
language models by bootstrapping off their own generations. Our pipeline
generates instructions, input, and output samples from a language model, then
filters invalid or similar ones before using them to finetune the original
model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute
improvement over the original model on Super-NaturalInstructions, on par with
the performance of InstructGPT-001, which was trained with private user data
and human annotations. For further evaluation, we curate a set of
expert-written instructions for novel tasks, and show through human evaluation
that tuning GPT3 with Self-Instruct outperforms using existing public
instruction datasets by a large margin, leaving only a 5% absolute gap behind
InstructGPT-001. Self-Instruct provides an almost annotation-free method for
aligning pre-trained language models with instructions, and we release our
large synthetic dataset to facilitate future studies on instruction tuning. Our
code and data are available at https://github.com/yizhongw/self-instruct.


6. [Towards Reasoning in Large Language Models: A Survey](http://arxiv.org/abs/2212.10403v2), Jie Huang, Kevin Chen-Chuan Chang, 20-12-2022
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Reasoning is a fundamental aspect of human intelligence that plays a crucial
role in activities such as problem solving, decision making, and critical
thinking. In recent years, large language models (LLMs) have made significant
progress in natural language processing, and there is observation that these
models may exhibit reasoning abilities when they are sufficiently large.
However, it is not yet clear to what extent LLMs are capable of reasoning. This
paper provides a comprehensive overview of the current state of knowledge on
reasoning in LLMs, including techniques for improving and eliciting reasoning
in these models, methods and benchmarks for evaluating reasoning abilities,
findings and implications of previous research in this field, and suggestions
on future directions. Our aim is to provide a detailed and up-to-date review of
this topic and stimulate meaningful discussion and future work.


6. [Towards Understanding Chain-of-Thought Prompting: An Empirical Study of
  What Matters](http://arxiv.org/abs/2212.10001v2), Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, Huan Sun, 20-12-2022
     ### Categories
     Computation and Language
    ### Abstract
    Chain-of-Thought (CoT) prompting can dramatically improve the multi-step
reasoning abilities of large language models (LLMs). CoT explicitly encourages
the LLM to generate intermediate rationales for solving a problem, by providing
a series of reasoning steps in the demonstrations. Despite its success, there
is still little understanding of what makes CoT prompting effective and which
aspects of the demonstrated reasoning steps contribute to its performance. In
this paper, we show that CoT reasoning is possible even with invalid
demonstrations - prompting with invalid reasoning steps can achieve over 80-90%
of the performance obtained using CoT under various metrics, while still
generating coherent lines of reasoning during inference. Further experiments
show that other aspects of the rationales, such as being relevant to the query
and correctly ordering the reasoning steps, are much more important for
effective CoT reasoning. Overall, these findings both deepen our understanding
of CoT prompting, and open up new questions regarding LLMs' capability to learn
to reason in context.


6. [Cramming: Training a Language Model on a Single GPU in One Day](http://arxiv.org/abs/2212.14034v1), Jonas Geiping, Tom Goldstein, 28-12-2022
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Recent trends in language modeling have focused on increasing performance
through scaling, and have resulted in an environment where training language
models is out of reach for most researchers and practitioners. While most in
the community are asking how to push the limits of extreme computation, we ask
the opposite question: How far can we get with a single GPU in just one day?
  We investigate the downstream performance achievable with a transformer-based
language model trained completely from scratch with masked language modeling
for a single day on a single consumer GPU. Aside from re-analyzing nearly all
components of the pretraining pipeline for this scenario and providing a
modified pipeline with performance close to BERT, we investigate why scaling
down is hard, and which modifications actually improve performance in this
scenario. We provide evidence that even in this constrained setting,
performance closely follows scaling laws observed in large-compute settings.
Through the lens of scaling laws, we categorize a range of recent improvements
to training and architecture and discuss their merit and practical
applicability (or lack thereof) for the limited compute setting.
## 2023 (195 papers)



7. [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](http://arxiv.org/abs/2301.00774v3), Elias Frantar, Dan Alistarh, 02-01-2023
     ### Categories
     Machine Learning
    ### Abstract
    We show for the first time that large-scale generative pretrained transformer
(GPT) family models can be pruned to at least 50% sparsity in one-shot, without
any retraining, at minimal loss of accuracy. This is achieved via a new pruning
method called SparseGPT, specifically designed to work efficiently and
accurately on massive GPT-family models. We can execute SparseGPT on the
largest available open-source models, OPT-175B and BLOOM-176B, in under 4.5
hours, and can reach 60% unstructured sparsity with negligible increase in
perplexity: remarkably, more than 100 billion weights from these models can be
ignored at inference time. SparseGPT generalizes to semi-structured (2:4 and
4:8) patterns, and is compatible with weight quantization approaches. The code
is available at: https://github.com/IST-DASLab/sparsegpt.


7. [Faithful Chain-of-Thought Reasoning](http://arxiv.org/abs/2301.13379v3), Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, Chris Callison-Burch, 31-01-2023
     ### Categories
     Computation and Language
    ### Abstract
    While Chain-of-Thought (CoT) prompting boosts Language Models' (LM)
performance on a gamut of complex reasoning tasks, the generated reasoning
chain does not necessarily reflect how the model arrives at the answer (aka.
faithfulness). We propose Faithful CoT, a reasoning framework involving two
stages: Translation (Natural Language query $\rightarrow$ symbolic reasoning
chain) and Problem Solving (reasoning chain $\rightarrow$ answer), using an LM
and a deterministic solver respectively. This guarantees that the reasoning
chain provides a faithful explanation of the final answer. Aside from
interpretability, Faithful CoT also improves empirical performance: it
outperforms standard CoT on 9 of 10 benchmarks from 4 diverse domains, with a
relative accuracy gain of 6.3% on Math Word Problems (MWP), 3.4% on Planning,
5.5% on Multi-hop Question Answering (QA), and 21.4% on Relational Inference.
Furthermore, with GPT-4 and Codex, it sets the new state-of-the-art few-shot
performance on 7 datasets (with 95.0+ accuracy on 6 of them), showing a strong
synergy between faithfulness and accuracy.


7. [Large Language Models Can Be Easily Distracted by Irrelevant Context](http://arxiv.org/abs/2302.00093v3), Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, Denny Zhou, 31-01-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models have achieved impressive performance on various natural
language processing tasks. However, so far they have been evaluated primarily
on benchmarks where all information in the input context is relevant for
solving the task. In this work, we investigate the distractibility of large
language models, i.e., how the model problem-solving accuracy can be influenced
by irrelevant context. In particular, we introduce Grade-School Math with
Irrelevant Context (GSM-IC), an arithmetic reasoning dataset with irrelevant
information in the problem description. We use this benchmark to measure the
distractibility of cutting-edge prompting techniques for large language models,
and find that the model performance is dramatically decreased when irrelevant
information is included. We also identify several approaches for mitigating
this deficiency, such as decoding with self-consistency and adding to the
prompt an instruction that tells the language model to ignore the irrelevant
information.


7. [Large Language Models are Versatile Decomposers: Decompose Evidence and
  Questions for Table-based Reasoning](http://arxiv.org/abs/2301.13808v3), Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, Yongbin Li, 31-01-2023
     ### Categories
     Computation and Language
    ### Abstract
    Table-based reasoning has shown remarkable progress in combining deep models
with discrete reasoning, which requires reasoning over both free-form natural
language (NL) questions and structured tabular data. However, previous
table-based reasoning solutions usually suffer from significant performance
degradation on huge evidence (tables). In addition, most existing methods
struggle to reason over complex questions since the required information is
scattered in different places. To alleviate the above challenges, we exploit
large language models (LLMs) as decomposers for effective table-based
reasoning, which (i) decompose huge evidence (a huge table) into sub-evidence
(a small table) to mitigate the interference of useless information for table
reasoning; and (ii) decompose complex questions into simpler sub-questions for
text reasoning. Specifically, we first use the LLMs to break down the evidence
(tables) involved in the current question, retaining the relevant evidence and
excluding the remaining irrelevant evidence from the huge table. In addition,
we propose a "parsing-execution-filling" strategy to alleviate the
hallucination dilemma of the chain of thought by decoupling logic and numerical
computation in each step. Extensive experiments show that our method can
effectively leverage decomposed evidence and questions and outperforms the
strong baselines on TabFact, WikiTableQuestion, and FetaQA datasets. Notably,
our model outperforms human performance for the first time on the TabFact
dataset.


7. [Collaborating with language models for embodied reasoning](http://arxiv.org/abs/2302.00763v1), Ishita Dasgupta, Christine Kaeser-Chen, Kenneth Marino, Arun Ahuja, Sheila Babayan, Felix Hill, Rob Fergus, 01-02-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Reasoning in a complex and ambiguous environment is a key goal for
Reinforcement Learning (RL) agents. While some sophisticated RL agents can
successfully solve difficult tasks, they require a large amount of training
data and often struggle to generalize to new unseen environments and new tasks.
On the other hand, Large Scale Language Models (LSLMs) have exhibited strong
reasoning ability and the ability to to adapt to new tasks through in-context
learning. However, LSLMs do not inherently have the ability to interrogate or
intervene on the environment. In this work, we investigate how to combine these
complementary abilities in a single system consisting of three parts: a
Planner, an Actor, and a Reporter. The Planner is a pre-trained language model
that can issue commands to a simple embodied agent (the Actor), while the
Reporter communicates with the Planner to inform its next command. We present a
set of tasks that require reasoning, test this system's ability to generalize
zero-shot and investigate failure cases, and demonstrate how components of this
system can be trained with reinforcement-learning to improve performance.


7. [Describe, Explain, Plan and Select: Interactive Planning with Large
  Language Models Enables Open-World Multi-Task Agents](http://arxiv.org/abs/2302.01560v2), Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang, 03-02-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    We investigate the challenge of task planning for multi-task embodied agents
in open-world environments. Two main difficulties are identified: 1) executing
plans in an open-world environment (e.g., Minecraft) necessitates accurate and
multi-step reasoning due to the long-term nature of tasks, and 2) as vanilla
planners do not consider how easy the current agent can achieve a given
sub-task when ordering parallel sub-goals within a complicated plan, the
resulting plan could be inefficient or even infeasible. To this end, we propose
"$\underline{D}$escribe, $\underline{E}$xplain, $\underline{P}$lan and
$\underline{S}$elect" ($\textbf{DEPS}$), an interactive planning approach based
on Large Language Models (LLMs). DEPS facilitates better error correction on
initial LLM-generated $\textit{plan}$ by integrating $\textit{description}$ of
the plan execution process and providing self-$\textit{explanation}$ of
feedback when encountering failures during the extended planning phases.
Furthermore, it includes a goal $\textit{selector}$, which is a trainable
module that ranks parallel candidate sub-goals based on the estimated steps of
completion, consequently refining the initial plan. Our experiments mark the
milestone of the first zero-shot multi-task agent that can robustly accomplish
70+ Minecraft tasks and nearly double the overall performances. Further testing
reveals our method's general effectiveness in popularly adopted non-open-ended
domains as well (i.e., ALFWorld and tabletop manipulation). The ablation and
exploratory studies detail how our design beats the counterparts and provide a
promising update on the $\texttt{ObtainDiamond}$ grand challenge with our
approach. The code is released at https://github.com/CraftJarvis/MC-Planner.


7. [Read and Reap the Rewards: Learning to Play Atari with the Help of
  Instruction Manuals](http://arxiv.org/abs/2302.04449v3), Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom M. Mitchell, 09-02-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    High sample complexity has long been a challenge for RL. On the other hand,
humans learn to perform tasks not only from interaction or demonstrations, but
also by reading unstructured text documents, e.g., instruction manuals.
Instruction manuals and wiki pages are among the most abundant data that could
inform agents of valuable features and policies or task-specific environmental
dynamics and reward structures. Therefore, we hypothesize that the ability to
utilize human-written instruction manuals to assist learning policies for
specific tasks should lead to a more efficient and better-performing agent. We
propose the Read and Reward framework. Read and Reward speeds up RL algorithms
on Atari games by reading manuals released by the Atari game developers. Our
framework consists of a QA Extraction module that extracts and summarizes
relevant information from the manual and a Reasoning module that evaluates
object-agent interactions based on information from the manual. An auxiliary
reward is then provided to a standard A2C RL agent, when interaction is
detected. Experimentally, various RL algorithms obtain significant improvement
in performance and training speed when assisted by our design.


7. [Transformer models: an introduction and catalog](http://arxiv.org/abs/2302.07730v3), Xavier Amatriain, Ananth Sankar, Jie Bing, Praveen Kumar Bodigutla, Timothy J. Hazen, Michaeel Kazi, 12-02-2023
     ### Categories
     Computation and Language
    ### Abstract
    In the past few years we have seen the meteoric appearance of dozens of
foundation models of the Transformer family, all of which have memorable and
sometimes funny, but not self-explanatory, names. The goal of this paper is to
offer a somewhat comprehensive but simple catalog and classification of the
most popular Transformer models. The paper also includes an introduction to the
most important aspects and innovations in Transformer models. Our catalog will
include models that are trained using self-supervised learning (e.g., BERT or
GPT3) as well as those that are further trained using a human-in-the-loop (e.g.
the InstructGPT model used by ChatGPT).


7. [Guiding Pretraining in Reinforcement Learning with Large Language Models](http://arxiv.org/abs/2302.06692v2), Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, Jacob Andreas, 13-02-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Reinforcement learning algorithms typically struggle in the absence of a
dense, well-shaped reward function. Intrinsically motivated exploration methods
address this limitation by rewarding agents for visiting novel states or
transitions, but these methods offer limited benefits in large environments
where most discovered novelty is irrelevant for downstream tasks. We describe a
method that uses background knowledge from text corpora to shape exploration.
This method, called ELLM (Exploring with LLMs) rewards an agent for achieving
goals suggested by a language model prompted with a description of the agent's
current state. By leveraging large-scale language model pretraining, ELLM
guides agents toward human-meaningful and plausibly useful behaviors without
requiring a human in the loop. We evaluate ELLM in the Crafter game environment
and the Housekeep robotic simulator, showing that ELLM-trained agents have
better coverage of common-sense behaviors during pretraining and usually match
or improve performance on a range of downstream tasks. Code available at
https://github.com/yuqingd/ellm.


7. [GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural
  Networks](http://arxiv.org/abs/2302.08043v3), Zemin Liu, Xingtong Yu, Yuan Fang, Xinming Zhang, 16-02-2023
     ### Categories
     Machine Learning, Computation and Language
    ### Abstract
    Graphs can model complex relationships between objects, enabling a myriad of
Web applications such as online page/article classification and social
recommendation. While graph neural networks(GNNs) have emerged as a powerful
tool for graph representation learning, in an end-to-end supervised setting,
their performance heavily rely on a large amount of task-specific supervision.
To reduce labeling requirement, the "pre-train, fine-tune" and "pre-train,
prompt" paradigms have become increasingly common. In particular, prompting is
a popular alternative to fine-tuning in natural language processing, which is
designed to narrow the gap between pre-training and downstream objectives in a
task-specific manner. However, existing study of prompting on graphs is still
limited, lacking a universal treatment to appeal to different downstream tasks.
In this paper, we propose GraphPrompt, a novel pre-training and prompting
framework on graphs. GraphPrompt not only unifies pre-training and downstream
tasks into a common task template, but also employs a learnable prompt to
assist a downstream task in locating the most relevant knowledge from the
pre-train model in a task-specific manner. Finally, we conduct extensive
experiments on five public datasets to evaluate and analyze GraphPrompt.


7. [A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT](http://arxiv.org/abs/2302.11382v1), Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt, 21-02-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Prompt engineering is an increasingly important skill set needed to converse
effectively with large language models (LLMs), such as ChatGPT. Prompts are
instructions given to an LLM to enforce rules, automate processes, and ensure
specific qualities (and quantities) of generated output. Prompts are also a
form of programming that can customize the outputs and interactions with an
LLM. This paper describes a catalog of prompt engineering techniques presented
in pattern form that have been applied to solve common problems when conversing
with LLMs. Prompt patterns are a knowledge transfer method analogous to
software patterns since they provide reusable solutions to common problems
faced in a particular context, i.e., output generation and interaction when
working with LLMs. This paper provides the following contributions to research
on prompt engineering that apply LLMs to automate software development tasks.
First, it provides a framework for documenting patterns for structuring prompts
to solve a range of problems so that they can be adapted to different domains.
Second, it presents a catalog of patterns that have been applied successfully
to improve the outputs of LLM conversations. Third, it explains how prompts can
be built from multiple patterns and illustrates prompt patterns that benefit
from combination with other prompt patterns.


7. [Guiding Large Language Models via Directional Stimulus Prompting](http://arxiv.org/abs/2302.11520v4), Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan, 22-02-2023
     ### Categories
     Computation and Language
    ### Abstract
    We introduce Directional Stimulus Prompting, a novel framework for guiding
black-box large language models (LLMs) toward specific desired outputs. Instead
of directly adjusting LLMs, our method employs a small tunable policy model
(e.g., T5) to generate an auxiliary directional stimulus prompt for each input
instance. These directional stimulus prompts act as nuanced, instance-specific
hints and clues to guide LLMs in generating desired outcomes, such as including
specific keywords in the generated summary. Our approach sidesteps the
challenges of direct LLM tuning by optimizing the policy model to explore
directional stimulus prompts that align LLMs with desired behaviors. The policy
model can be optimized through 1) supervised fine-tuning using labeled data and
2) reinforcement learning from offline or online rewards based on the LLM's
output. We assess our method across summarization, dialogue response
generation, and chain-of-thought reasoning tasks. Our experiments demonstrate
that the framework consistently improves LLMs' (e.g., ChatGPT, Codex,
InstructGPT) performance on these supervised tasks using minimal labeled data.
Notably, using just 80 dialogues on the MultiWOZ dataset, our approach enhances
ChatGPT's performance by an impressive 41.4%, matching or surpassing some fully
supervised start-of-the-art models. Additionally, the instance-specific
chain-of-thought prompt generated by our approach improves InstructGPT's
reasoning accuracy compared to human-crafted or automatically generated
prompts. The code and data are publicly available at
\url{https://github.com/Leezekun/Directional-Stimulus-Prompting}.


7. [Active Prompting with Chain-of-Thought for Large Language Models](http://arxiv.org/abs/2302.12246v3), Shizhe Diao, Pengcheng Wang, Yong Lin, Tong Zhang, 23-02-2023
     ### Categories
     Computation and Language
    ### Abstract
    The increasing scale of large language models (LLMs) brings emergent
abilities to various complex tasks requiring reasoning, such as arithmetic and
commonsense reasoning. It is known that the effective design of task-specific
prompts is critical for LLMs' ability to produce high-quality answers. In
particular, an effective approach for complex question-and-answer tasks is
example-based prompting with chain-of-thought (CoT) reasoning, which
significantly improves the performance of LLMs. However, current CoT methods
rely on a fixed set of human-annotated exemplars, which are not necessarily the
most effective examples for different tasks. This paper proposes a new method,
Active-Prompt, to adapt LLMs to different tasks with task-specific example
prompts (annotated with human-designed CoT reasoning). For this purpose, we
propose a solution to the key problem of determining which questions are the
most important and helpful ones to annotate from a pool of task-specific
queries. By borrowing ideas from the related problem of uncertainty-based
active learning, we introduce several metrics to characterize the uncertainty
so as to select the most uncertain questions for annotation. Experimental
results demonstrate the superiority of our proposed method, achieving
state-of-the-art on eight complex reasoning tasks. Further analyses of
different uncertainty metrics, pool sizes, zero-shot learning, and
accuracy-uncertainty relationship demonstrate the effectiveness of our method.
Our code will be available at https://github.com/shizhediao/active-prompt.


7. [LLaMA: Open and Efficient Foundation Language Models](http://arxiv.org/abs/2302.13971v1), Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample, 27-02-2023
     ### Categories
     Computation and Language
    ### Abstract
    We introduce LLaMA, a collection of foundation language models ranging from
7B to 65B parameters. We train our models on trillions of tokens, and show that
it is possible to train state-of-the-art models using publicly available
datasets exclusively, without resorting to proprietary and inaccessible
datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks,
and LLaMA-65B is competitive with the best models, Chinchilla-70B and
PaLM-540B. We release all our models to the research community.


7. [Language Is Not All You Need: Aligning Perception with Language Models](http://arxiv.org/abs/2302.14045v2), Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei, 27-02-2023
     ### Categories
     Computation and Language
    ### Abstract
    A big convergence of language, multimodal perception, action, and world
modeling is a key step toward artificial general intelligence. In this work, we
introduce Kosmos-1, a Multimodal Large Language Model (MLLM) that can perceive
general modalities, learn in context (i.e., few-shot), and follow instructions
(i.e., zero-shot). Specifically, we train Kosmos-1 from scratch on web-scale
multimodal corpora, including arbitrarily interleaved text and images,
image-caption pairs, and text data. We evaluate various settings, including
zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range
of tasks without any gradient updates or finetuning. Experimental results show
that Kosmos-1 achieves impressive performance on (i) language understanding,
generation, and even OCR-free NLP (directly fed with document images), (ii)
perception-language tasks, including multimodal dialogue, image captioning,
visual question answering, and (iii) vision tasks, such as image recognition
with descriptions (specifying classification via text instructions). We also
show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge
from language to multimodal, and from multimodal to language. In addition, we
introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning
capability of MLLMs.


7. [Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning](http://arxiv.org/abs/2303.02861v1), Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, Yoon Kim, 06-03-2023
     ### Categories
     Computation and Language
    ### Abstract
    Prompt tuning, in which a base pretrained model is adapted to each task via
conditioning on learned prompt vectors, has emerged as a promising approach for
efficiently adapting large language models to multiple downstream tasks.
However, existing methods typically learn soft prompt vectors from scratch, and
it has not been clear how to exploit the rich cross-task knowledge with prompt
vectors in a multitask learning setting. We propose multitask prompt tuning
(MPT), which first learns a single transferable prompt by distilling knowledge
from multiple task-specific source prompts. We then learn multiplicative low
rank updates to this shared prompt to efficiently adapt it to each downstream
target task. Extensive experiments on 23 NLP datasets demonstrate that our
proposed approach outperforms the state-of-the-art methods, including the full
finetuning baseline in some cases, despite only tuning 0.035% as many
task-specific parameters.


7. [Foundation Models for Decision Making: Problems, Methods, and
  Opportunities](http://arxiv.org/abs/2303.04129v1), Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, Dale Schuurmans, 07-03-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    Foundation models pretrained on diverse data at scale have demonstrated
extraordinary capabilities in a wide range of vision and language tasks. When
such models are deployed in real world environments, they inevitably interface
with other entities and agents. For example, language models are often used to
interact with human beings through dialogue, and visual perception models are
used to autonomously navigate neighborhood streets. In response to these
developments, new paradigms are emerging for training foundation models to
interact with other agents and perform long-term reasoning. These paradigms
leverage the existence of ever-larger datasets curated for multimodal,
multitask, and generalist interaction. Research at the intersection of
foundation models and decision making holds tremendous promise for creating
powerful new systems that can interact effectively across a diverse range of
applications such as dialogue, autonomous driving, healthcare, education, and
robotics. In this manuscript, we examine the scope of foundation models for
decision making, and provide conceptual tools and technical background for
understanding the problem space and exploring new research directions. We
review recent approaches that ground foundation models in practical decision
making applications through a variety of methods such as prompting, conditional
generative modeling, planning, optimal control, and reinforcement learning, and
discuss common challenges and open problems in the field.


7. [Large Language Models in the Workplace: A Case Study on Prompt
  Engineering for Job Type Classification](http://arxiv.org/abs/2303.07142v3), Benjamin Clavié, Alexandru Ciceu, Frederick Naylor, Guillaume Soulié, Thomas Brightwell, 13-03-2023
     ### Categories
     Computation and Language
    ### Abstract
    This case study investigates the task of job classification in a real-world
setting, where the goal is to determine whether an English-language job posting
is appropriate for a graduate or entry-level position. We explore multiple
approaches to text classification, including supervised approaches such as
traditional models like Support Vector Machines (SVMs) and state-of-the-art
deep learning methods such as DeBERTa. We compare them with Large Language
Models (LLMs) used in both few-shot and zero-shot classification settings. To
accomplish this task, we employ prompt engineering, a technique that involves
designing prompts to guide the LLMs towards the desired output. Specifically,
we evaluate the performance of two commercially available state-of-the-art
GPT-3.5-based language models, text-davinci-003 and gpt-3.5-turbo. We also
conduct a detailed analysis of the impact of different aspects of prompt
engineering on the model's performance. Our results show that, with a
well-designed prompt, a zero-shot gpt-3.5-turbo classifier outperforms all
other models, achieving a 6% increase in Precision@95% Recall compared to the
best supervised approach. Furthermore, we observe that the wording of the
prompt is a critical factor in eliciting the appropriate "reasoning" in the
model, and that seemingly minor aspects of the prompt significantly affect the
model's performance.


7. [GPT-4 Technical Report](http://arxiv.org/abs/2303.08774v4),  OpenAI,  :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto,  Michael,  Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, Barret Zoph, 15-03-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance on
various professional and academic benchmarks, including passing a simulated bar
exam with a score around the top 10% of test takers. GPT-4 is a
Transformer-based model pre-trained to predict the next token in a document.
The post-training alignment process results in improved performance on measures
of factuality and adherence to desired behavior. A core component of this
project was developing infrastructure and optimization methods that behave
predictably across a wide range of scales. This allowed us to accurately
predict some aspects of GPT-4's performance based on models trained with no
more than 1/1,000th the compute of GPT-4.


7. [ART: Automatic multi-step reasoning and tool-use for large language
  models](http://arxiv.org/abs/2303.09014v1), Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, Marco Tulio Ribeiro, 16-03-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) can perform complex reasoning in few- and
zero-shot settings by generating intermediate chain of thought (CoT) reasoning
steps. Further, each reasoning step can rely on external tools to support
computation beyond the core LLM capabilities (e.g. search/running code). Prior
work on CoT prompting and tool use typically requires hand-crafting
task-specific demonstrations and carefully scripted interleaving of model
generations with tool use. We introduce Automatic Reasoning and Tool-use (ART),
a framework that uses frozen LLMs to automatically generate intermediate
reasoning steps as a program. Given a new task to solve, ART selects
demonstrations of multi-step reasoning and tool use from a task library. At
test time, ART seamlessly pauses generation whenever external tools are called,
and integrates their output before resuming generation. ART achieves a
substantial improvement over few-shot prompting and automatic CoT on unseen
tasks in the BigBench and MMLU benchmarks, and matches performance of
hand-crafted CoT prompts on a majority of these tasks. ART is also extensible,
and makes it easy for humans to improve performance by correcting errors in
task-specific programs or incorporating new tools, which we demonstrate by
drastically improving performance on select tasks with minimal human
intervention.


7. [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](http://arxiv.org/abs/2303.10512v2), Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao, 18-03-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    Fine-tuning large pre-trained language models on downstream tasks has become
an important paradigm in NLP. However, common practice fine-tunes all of the
parameters in a pre-trained model, which becomes prohibitive when a large
number of downstream tasks are present. Therefore, many fine-tuning methods are
proposed to learn incremental updates of pre-trained weights in a parameter
efficient way, e.g., low-rank increments. These methods often evenly distribute
the budget of incremental updates across all pre-trained weight matrices, and
overlook the varying importance of different weight parameters. As a
consequence, the fine-tuning performance is suboptimal. To bridge this gap, we
propose AdaLoRA, which adaptively allocates the parameter budget among weight
matrices according to their importance score. In particular, AdaLoRA
parameterizes the incremental updates in the form of singular value
decomposition. Such a novel approach allows us to effectively prune the
singular values of unimportant updates, which is essentially to reduce their
parameter budget but circumvent intensive exact SVD computations. We conduct
extensive experiments with several pre-trained models on natural language
processing, question answering, and natural language generation to validate the
effectiveness of AdaLoRA. Results demonstrate that AdaLoRA manifests notable
improvement over baselines, especially in the low budget settings. Our code is
publicly available at https://github.com/QingruZhang/AdaLoRA .


7. [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](http://arxiv.org/abs/2303.11381v1), Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, Lijuan Wang, 20-03-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    We propose MM-REACT, a system paradigm that integrates ChatGPT with a pool of
vision experts to achieve multimodal reasoning and action. In this paper, we
define and explore a comprehensive list of advanced vision tasks that are
intriguing to solve, but may exceed the capabilities of existing vision and
vision-language models. To achieve such advanced visual intelligence, MM-REACT
introduces a textual prompt design that can represent text descriptions,
textualized spatial coordinates, and aligned file names for dense visual
signals such as images and videos. MM-REACT's prompt design allows language
models to accept, associate, and process multimodal information, thereby
facilitating the synergetic combination of ChatGPT and various vision experts.
Zero-shot experiments demonstrate MM-REACT's effectiveness in addressing the
specified capabilities of interests and its wide application in different
scenarios that require advanced visual understanding. Furthermore, we discuss
and compare MM-REACT's system paradigm with an alternative approach that
extends language models for multimodal scenarios through joint finetuning.
Code, demo, video, and visualization are available at
https://multimodal-react.github.io/


7. [Sparks of Artificial General Intelligence: Early experiments with GPT-4](http://arxiv.org/abs/2303.12712v5), Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang, 22-03-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Artificial intelligence (AI) researchers have been developing and refining
large language models (LLMs) that exhibit remarkable capabilities across a
variety of domains and tasks, challenging our understanding of learning and
cognition. The latest model developed by OpenAI, GPT-4, was trained using an
unprecedented scale of compute and data. In this paper, we report on our
investigation of an early version of GPT-4, when it was still in active
development by OpenAI. We contend that (this early version of) GPT-4 is part of
a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that
exhibit more general intelligence than previous AI models. We discuss the
rising capabilities and implications of these models. We demonstrate that,
beyond its mastery of language, GPT-4 can solve novel and difficult tasks that
span mathematics, coding, vision, medicine, law, psychology and more, without
needing any special prompting. Moreover, in all of these tasks, GPT-4's
performance is strikingly close to human-level performance, and often vastly
surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's
capabilities, we believe that it could reasonably be viewed as an early (yet
still incomplete) version of an artificial general intelligence (AGI) system.
In our exploration of GPT-4, we put special emphasis on discovering its
limitations, and we discuss the challenges ahead for advancing towards deeper
and more comprehensive versions of AGI, including the possible need for
pursuing a new paradigm that moves beyond next-word prediction. We conclude
with reflections on societal influences of the recent technological leap and
future research directions.


7. [Natural Language Reasoning, A Survey](http://arxiv.org/abs/2303.14725v2), Fei Yu, Hongbo Zhang, Prayag Tiwari, Benyou Wang, 26-03-2023
     ### Categories
     Computation and Language
    ### Abstract
    This survey paper proposes a clearer view of natural language reasoning in
the field of Natural Language Processing (NLP), both conceptually and
practically. Conceptually, we provide a distinct definition for natural
language reasoning in NLP, based on both philosophy and NLP scenarios, discuss
what types of tasks require reasoning, and introduce a taxonomy of reasoning.
Practically, we conduct a comprehensive literature review on natural language
reasoning in NLP, mainly covering classical logical reasoning, natural language
inference, multi-hop question answering, and commonsense reasoning. The paper
also identifies and views backward reasoning, a powerful paradigm for
multi-step reasoning, and introduces defeasible reasoning as one of the most
important future directions in natural language reasoning research. We focus on
single-modality unstructured natural language text, excluding neuro-symbolic
techniques and mathematical reasoning.


7. [Ecosystem Graphs: The Social Footprint of Foundation Models](http://arxiv.org/abs/2303.15772v1), Rishi Bommasani, Dilara Soylu, Thomas I. Liao, Kathleen A. Creel, Percy Liang, 28-03-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    Foundation models (e.g. ChatGPT, StableDiffusion) pervasively influence
society, warranting immediate social attention. While the models themselves
garner much attention, to accurately characterize their impact, we must
consider the broader sociotechnical ecosystem. We propose Ecosystem Graphs as a
documentation framework to transparently centralize knowledge of this
ecosystem. Ecosystem Graphs is composed of assets (datasets, models,
applications) linked together by dependencies that indicate technical (e.g. how
Bing relies on GPT-4) and social (e.g. how Microsoft relies on OpenAI)
relationships. To supplement the graph structure, each asset is further
enriched with fine-grained metadata (e.g. the license or training emissions).
We document the ecosystem extensively at
https://crfm.stanford.edu/ecosystem-graphs/. As of March 16, 2023, we annotate
262 assets (64 datasets, 128 models, 70 applications) from 63 organizations
linked by 356 dependencies. We show Ecosystem Graphs functions as a powerful
abstraction and interface for achieving the minimum transparency required to
address myriad use cases. Therefore, we envision Ecosystem Graphs will be a
community-maintained resource that provides value to stakeholders spanning AI
researchers, industry professionals, social scientists, auditors and
policymakers.


7. [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init
  Attention](http://arxiv.org/abs/2303.16199v2), Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Yu Qiao, 28-03-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    We present LLaMA-Adapter, a lightweight adaption method to efficiently
fine-tune LLaMA into an instruction-following model. Using 52K self-instruct
demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon
the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8
A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and
prepend them to the word tokens at higher transformer layers. Then, a
zero-initialized attention mechanism with zero gating is proposed, which
adaptively injects the new instructional cues into LLaMA, while effectively
preserves its pre-trained knowledge. With our efficient training, LLaMA-Adapter
can generate high-quality responses, comparable to Alpaca with fully fine-tuned
7B parameters. Besides language commands, our approach can be simply extended
to multi-modal instructions for learning image-conditioned LLaMA model, which
achieves superior reasoning performance on ScienceQA and COCO Caption
benchmarks. Furthermore, we also evaluate the zero-initialized attention
mechanism for fine-tuning other pre-trained models (ViT, RoBERTa) on
traditional vision and language tasks, demonstrating the superior
generalization capacity of our approach. Code is released at
https://github.com/OpenGVLab/LLaMA-Adapter.


7. [BloombergGPT: A Large Language Model for Finance](http://arxiv.org/abs/2303.17564v3), Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann, 30-03-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    The use of NLP in the realm of financial technology is broad and complex,
with applications ranging from sentiment analysis and named entity recognition
to question answering. Large Language Models (LLMs) have been shown to be
effective on a variety of tasks; however, no LLM specialized for the financial
domain has been reported in literature. In this work, we present BloombergGPT,
a 50 billion parameter language model that is trained on a wide range of
financial data. We construct a 363 billion token dataset based on Bloomberg's
extensive data sources, perhaps the largest domain-specific dataset yet,
augmented with 345 billion tokens from general purpose datasets. We validate
BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite
of internal benchmarks that most accurately reflect our intended usage. Our
mixed dataset training leads to a model that outperforms existing models on
financial tasks by significant margins without sacrificing performance on
general LLM benchmarks. Additionally, we explain our modeling choices, training
process, and evaluation methodology. We release Training Chronicles (Appendix
C) detailing our experience in training BloombergGPT.


7. [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging
  Face](http://arxiv.org/abs/2303.17580v4), Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang, 30-03-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Solving complicated AI tasks with different domains and modalities is a key
step toward artificial general intelligence. While there are numerous AI models
available for various domains and modalities, they cannot handle complicated AI
tasks autonomously. Considering large language models (LLMs) have exhibited
exceptional abilities in language understanding, generation, interaction, and
reasoning, we advocate that LLMs could act as a controller to manage existing
AI models to solve complicated AI tasks, with language serving as a generic
interface to empower this. Based on this philosophy, we present HuggingGPT, an
LLM-powered agent that leverages LLMs (e.g., ChatGPT) to connect various AI
models in machine learning communities (e.g., Hugging Face) to solve AI tasks.
Specifically, we use ChatGPT to conduct task planning when receiving a user
request, select models according to their function descriptions available in
Hugging Face, execute each subtask with the selected AI model, and summarize
the response according to the execution results. By leveraging the strong
language capability of ChatGPT and abundant AI models in Hugging Face,
HuggingGPT can tackle a wide range of sophisticated AI tasks spanning different
modalities and domains and achieve impressive results in language, vision,
speech, and other challenging tasks, which paves a new way towards the
realization of artificial general intelligence.


7. [A Bibliometric Review of Large Language Models Research from 2017 to
  2023](http://arxiv.org/abs/2304.02020v1), Lizhou Fan, Lingyao Li, Zihui Ma, Sanggyu Lee, Huizi Yu, Libby Hemphill, 03-04-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) are a class of language models that have
demonstrated outstanding performance across a range of natural language
processing (NLP) tasks and have become a highly sought-after research area,
because of their ability to generate human-like language and their potential to
revolutionize science and technology. In this study, we conduct bibliometric
and discourse analyses of scholarly literature on LLMs. Synthesizing over 5,000
publications, this paper serves as a roadmap for researchers, practitioners,
and policymakers to navigate the current landscape of LLMs research. We present
the research trends from 2017 to early 2023, identifying patterns in research
paradigms and collaborations. We start with analyzing the core algorithm
developments and NLP tasks that are fundamental in LLMs research. We then
investigate the applications of LLMs in various fields and domains including
medicine, engineering, social science, and humanities. Our review also reveals
the dynamic, fast-paced evolution of LLMs research. Overall, this paper offers
valuable insights into the current state, impact, and potential of LLMs
research and its applications.


7. [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on
  Self-Chat Data](http://arxiv.org/abs/2304.01196v4), Canwen Xu, Daya Guo, Nan Duan, Julian McAuley, 03-04-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Chat models, such as ChatGPT, have shown impressive capabilities and have
been rapidly adopted across numerous domains. However, these models are only
accessible through a restricted API, creating barriers for new research and
progress in the field. We propose a pipeline that can automatically generate a
high-quality multi-turn chat corpus by leveraging ChatGPT to engage in a
conversation with itself. Subsequently, we employ parameter-efficient tuning to
enhance LLaMA, an open-source large language model. The resulting model, named
Baize, demonstrates good performance in multi-turn dialogues with guardrails
that minimize potential risks. Furthermore, we propose a new technique called
Self-Distill with Feedback, to further improve the performance of the Baize
models with feedback from ChatGPT. The Baize models and data are released for
research purposes only at https://github.com/project-baize/baize-chatbot. An
online demo is also available at
https://huggingface.co/spaces/project-baize/chat-with-baize.


7. [Pythia: A Suite for Analyzing Large Language Models Across Training and
  Scaling](http://arxiv.org/abs/2304.01373v2), Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, Oskar van der Wal, 03-04-2023
     ### Categories
     Computation and Language
    ### Abstract
    How do large language models (LLMs) develop and evolve over the course of
training? How do these patterns change as models scale? To answer these
questions, we introduce \textit{Pythia}, a suite of 16 LLMs all trained on
public data seen in the exact same order and ranging in size from 70M to 12B
parameters. We provide public access to 154 checkpoints for each one of the 16
models, alongside tools to download and reconstruct their exact training
dataloaders for further study. We intend \textit{Pythia} to facilitate research
in many areas, and we present several case studies including novel results in
memorization, term frequency effects on few-shot performance, and reducing
gender bias. We demonstrate that this highly controlled setup can be used to
yield novel insights toward LLMs and their training dynamics. Trained models,
analysis code, training code, and training data can be found at
\url{https://github.com/EleutherAI/pythia}.


7. [One Small Step for Generative AI, One Giant Leap for AGI: A Complete
  Survey on ChatGPT in AIGC Era](http://arxiv.org/abs/2304.06488v1), Chaoning Zhang, Chenshuang Zhang, Chenghao Li, Yu Qiao, Sheng Zheng, Sumit Kumar Dam, Mengchun Zhang, Jung Uk Kim, Seong Tae Kim, Jinwoo Choi, Gyeong-Moon Park, Sung-Ho Bae, Lik-Hang Lee, Pan Hui, In So Kweon, Choong Seon Hong, 04-04-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    OpenAI has recently released GPT-4 (a.k.a. ChatGPT plus), which is
demonstrated to be one small step for generative AI (GAI), but one giant leap
for artificial general intelligence (AGI). Since its official release in
November 2022, ChatGPT has quickly attracted numerous users with extensive
media coverage. Such unprecedented attention has also motivated numerous
researchers to investigate ChatGPT from various aspects. According to Google
scholar, there are more than 500 articles with ChatGPT in their titles or
mentioning it in their abstracts. Considering this, a review is urgently
needed, and our work fills this gap. Overall, this work is the first to survey
ChatGPT with a comprehensive review of its underlying technology, applications,
and challenges. Moreover, we present an outlook on how ChatGPT might evolve to
realize general-purpose AIGC (a.k.a. AI-generated content), which will be a
significant milestone for the development of AGI.


7. [Generative Agents: Interactive Simulacra of Human Behavior](http://arxiv.org/abs/2304.03442v2), Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein, 07-04-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    Believable proxies of human behavior can empower interactive applications
ranging from immersive environments to rehearsal spaces for interpersonal
communication to prototyping tools. In this paper, we introduce generative
agents--computational software agents that simulate believable human behavior.
Generative agents wake up, cook breakfast, and head to work; artists paint,
while authors write; they form opinions, notice each other, and initiate
conversations; they remember and reflect on days past as they plan the next
day. To enable generative agents, we describe an architecture that extends a
large language model to store a complete record of the agent's experiences
using natural language, synthesize those memories over time into higher-level
reflections, and retrieve them dynamically to plan behavior. We instantiate
generative agents to populate an interactive sandbox environment inspired by
The Sims, where end users can interact with a small town of twenty five agents
using natural language. In an evaluation, these generative agents produce
believable individual and emergent social behaviors: for example, starting with
only a single user-specified notion that one agent wants to throw a Valentine's
Day party, the agents autonomously spread invitations to the party over the
next two days, make new acquaintances, ask each other out on dates to the
party, and coordinate to show up for the party together at the right time. We
demonstrate through ablation that the components of our agent
architecture--observation, planning, and reflection--each contribute critically
to the believability of agent behavior. By fusing large language models with
computational, interactive agents, this work introduces architectural and
interaction patterns for enabling believable simulations of human behavior.


7. [OpenAssistant Conversations -- Democratizing Large Language Model
  Alignment](http://arxiv.org/abs/2304.07327v2), Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick, 14-04-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Aligning large language models (LLMs) with human preferences has proven to
drastically improve usability and has driven rapid adoption as demonstrated by
ChatGPT. Alignment techniques such as supervised fine-tuning (SFT) and
reinforcement learning from human feedback (RLHF) greatly reduce the required
skill and domain knowledge to effectively harness the capabilities of LLMs,
increasing their accessibility and utility across various domains. However,
state-of-the-art alignment techniques like RLHF rely on high-quality human
feedback data, which is expensive to create and often remains proprietary. In
an effort to democratize research on large-scale alignment, we release
OpenAssistant Conversations, a human-generated, human-annotated assistant-style
conversation corpus consisting of 161,443 messages in 35 different languages,
annotated with 461,292 quality ratings, resulting in over 10,000 complete and
fully annotated conversation trees. The corpus is a product of a worldwide
crowd-sourcing effort involving over 13,500 volunteers. Models trained on
OpenAssistant Conversations show consistent improvements on standard benchmarks
over respective base models. We release our code and data under a fully
permissive licence.


7. [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](http://arxiv.org/abs/2304.13712v2), Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu, 26-04-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    This paper presents a comprehensive and practical guide for practitioners and
end-users working with Large Language Models (LLMs) in their downstream natural
language processing (NLP) tasks. We provide discussions and insights into the
usage of LLMs from the perspectives of models, data, and downstream tasks.
Firstly, we offer an introduction and brief summary of current GPT- and
BERT-style LLMs. Then, we discuss the influence of pre-training data, training
data, and test data. Most importantly, we provide a detailed discussion about
the use and non-use cases of large language models for various natural language
processing tasks, such as knowledge-intensive tasks, traditional natural
language understanding tasks, natural language generation tasks, emergent
abilities, and considerations for specific tasks.We present various use cases
and non-use cases to illustrate the practical applications and limitations of
LLMs in real-world scenarios. We also try to understand the importance of data
and the specific challenges associated with each NLP task. Furthermore, we
explore the impact of spurious biases on LLMs and delve into other essential
considerations, such as efficiency, cost, and latency, to ensure a
comprehensive understanding of deploying LLMs in practice. This comprehensive
guide aims to provide researchers and practitioners with valuable insights and
best practices for working with LLMs, thereby enabling the successful
implementation of these models in a wide range of NLP tasks. A curated list of
practical guide resources of LLMs, regularly updated, can be found at
\url{https://github.com/Mooler0410/LLMsPracticalGuide}.


7. [MLCopilot: Unleashing the Power of Large Language Models in Solving
  Machine Learning Tasks](http://arxiv.org/abs/2304.14979v1), Lei Zhang, Yuge Zhang, Kan Ren, Dongsheng Li, Yuqing Yang, 28-04-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    The field of machine learning (ML) has gained widespread adoption, leading to
a significant demand for adapting ML to specific scenarios, which is yet
expensive and non-trivial. The predominant approaches towards the automation of
solving ML tasks (e.g., AutoML) are often time consuming and hard to understand
for human developers. In contrast, though human engineers have the incredible
ability to understand tasks and reason about solutions, their experience and
knowledge are often sparse and difficult to utilize by quantitative approaches.
In this paper, we aim to bridge the gap between machine intelligence and human
knowledge by introducing a novel framework MLCopilot, which leverages the
state-of-the-art LLMs to develop ML solutions for novel tasks. We showcase the
possibility of extending the capability of LLMs to comprehend structured inputs
and perform thorough reasoning for solving novel ML tasks. And we find that,
after some dedicated design, the LLM can (i) observe from the existing
experiences of ML tasks and (ii) reason effectively to deliver promising
results for new tasks. The solution generated can be used directly to achieve
high levels of competitiveness.


7. [Plan, Eliminate, and Track -- Language Models are Good Teachers for
  Embodied Agents](http://arxiv.org/abs/2305.02412v2), Yue Wu, So Yeon Min, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Yuanzhi Li, Tom Mitchell, Shrimai Prabhumoye, 03-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Pre-trained large language models (LLMs) capture procedural knowledge about
the world. Recent work has leveraged LLM's ability to generate abstract plans
to simplify challenging control tasks, either by action scoring, or action
modeling (fine-tuning). However, the transformer architecture inherits several
constraints that make it difficult for the LLM to directly serve as the agent:
e.g. limited input lengths, fine-tuning inefficiency, bias from pre-training,
and incompatibility with non-text environments. To maintain compatibility with
a low-level trainable actor, we propose to instead use the knowledge in LLMs to
simplify the control problem, rather than solving it. We propose the Plan,
Eliminate, and Track (PET) framework. The Plan module translates a task
description into a list of high-level sub-tasks. The Eliminate module masks out
irrelevant objects and receptacles from the observation for the current
sub-task. Finally, the Track module determines whether the agent has
accomplished each sub-task. On the AlfWorld instruction following benchmark,
the PET framework leads to a significant 15% improvement over SOTA for
generalization to human goal specifications.


7. [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](http://arxiv.org/abs/2305.03495v2), Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng, 04-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large Language Models (LLMs) have shown impressive performance as general
purpose agents, but their abilities remain highly dependent on prompts which
are hand written with onerous trial-and-error effort. We propose a simple and
nonparametric solution to this problem, Automatic Prompt Optimization (APO),
which is inspired by numerical gradient descent to automatically improve
prompts, assuming access to training data and an LLM API. The algorithm uses
minibatches of data to form natural language "gradients" that criticize the
current prompt. The gradients are then "propagated" into the prompt by editing
the prompt in the opposite semantic direction of the gradient. These gradient
descent steps are guided by a beam search and bandit selection procedure which
significantly improves algorithmic efficiency. Preliminary results across three
benchmark NLP tasks and the novel problem of LLM jailbreak detection suggest
that Automatic Prompt Optimization can outperform prior prompt editing
techniques and improve an initial prompt's performance by up to 31%, by using
data to rewrite vague task descriptions into more precise annotation
instructions.


7. [Exploring Human-Like Translation Strategy with Large Language Models](http://arxiv.org/abs/2305.04118v3), Zhiwei He, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Yujiu Yang, Rui Wang, Zhaopeng Tu, Shuming Shi, Xing Wang, 06-05-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) have demonstrated impressive capabilities in
general scenarios, exhibiting a level of aptitude that approaches, in some
aspects even surpasses, human-level intelligence. Among their numerous skills,
the translation abilities of LLMs have received considerable attention.
Compared to typical machine translation that focuses solely on source-to-target
mapping, LLM-based translation can potentially mimic the human translation
process which might take preparatory steps to ensure high-quality translation.
This work explores this possibility by proposing the MAPS framework, which
stands for Multi-Aspect Prompting and Selection. Specifically, we enable LLMs
first to analyze the given source sentence and induce three aspects of
translation-related knowledge: keywords, topics, and relevant demonstrations to
guide the final translation process. Moreover, we employ a selection mechanism
based on quality estimation to filter out noisy and unhelpful knowledge. Both
automatic (3 LLMs x 11 directions x 2 automatic metrics) and human evaluation
(preference study and MQM) demonstrate the effectiveness of MAPS. Further
analysis shows that by mimicking the human translation process, MAPS reduces
various translation errors such as hallucination, ambiguity, mistranslation,
awkward style, untranslated text, and omission. Source code is available at
https://github.com/zwhe99/MAPS-mt.


7. [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
  by Large Language Models](http://arxiv.org/abs/2305.04091v3), Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim, 06-05-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) have recently been shown to deliver impressive
performance in various NLP tasks. To tackle multi-step reasoning tasks,
few-shot chain-of-thought (CoT) prompting includes a few manually crafted
step-by-step reasoning demonstrations which enable LLMs to explicitly generate
reasoning steps and improve their reasoning task accuracy. To eliminate the
manual effort, Zero-shot-CoT concatenates the target problem statement with
"Let's think step by step" as an input prompt to LLMs. Despite the success of
Zero-shot-CoT, it still suffers from three pitfalls: calculation errors,
missing-step errors, and semantic misunderstanding errors. To address the
missing-step errors, we propose Plan-and-Solve (PS) Prompting. It consists of
two components: first, devising a plan to divide the entire task into smaller
subtasks, and then carrying out the subtasks according to the plan. To address
the calculation errors and improve the quality of generated reasoning steps, we
extend PS prompting with more detailed instructions and derive PS+ prompting.
We evaluate our proposed prompting strategy on ten datasets across three
reasoning problems. The experimental results over GPT-3 show that our proposed
zero-shot prompting consistently outperforms Zero-shot-CoT across all datasets
by a large margin, is comparable to or exceeds Zero-shot-Program-of-Thought
Prompting, and has comparable performance with 8-shot CoT prompting on the math
reasoning problem. The code can be found at
https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting.


7. [TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?](http://arxiv.org/abs/2305.07759v2), Ronen Eldan, Yuanzhi Li, 12-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Language models (LMs) are powerful tools for natural language processing, but
they often struggle to produce coherent and fluent text when they are small.
Models with around 125M parameters such as GPT-Neo (small) or GPT-2 (small) can
rarely generate coherent and consistent English text beyond a few words even
after extensive training. This raises the question of whether the emergence of
the ability to produce coherent English text only occurs at larger scales (with
hundreds of millions of parameters or more) and complex architectures (with
many layers of global attention).
  In this work, we introduce TinyStories, a synthetic dataset of short stories
that only contain words that a typical 3 to 4-year-olds usually understand,
generated by GPT-3.5 and GPT-4. We show that TinyStories can be used to train
and evaluate LMs that are much smaller than the state-of-the-art models (below
10 million total parameters), or have much simpler architectures (with only one
transformer block), yet still produce fluent and consistent stories with
several paragraphs that are diverse and have almost perfect grammar, and
demonstrate reasoning capabilities.
  We also introduce a new paradigm for the evaluation of language models: We
suggest a framework which uses GPT-4 to grade the content generated by these
models as if those were stories written by students and graded by a (human)
teacher. This new paradigm overcomes the flaws of standard benchmarks which
often requires the model's output to be very structures, and moreover provides
a multidimensional score for the model, providing scores for different
capabilities such as grammar, creativity and consistency.
  We hope that TinyStories can facilitate the development, analysis and
research of LMs, especially for low-resource or specialized domains, and shed
light on the emergence of language capabilities in LMs.


7. [Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM
  Inference with Transferable Prompt](http://arxiv.org/abs/2305.11186v2), Zhaozhuo Xu, Zirui Liu, Beidi Chen, Yuxin Tang, Jue Wang, Kaixiong Zhou, Xia Hu, Anshumali Shrivastava, 17-05-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    While the numerous parameters in Large Language Models (LLMs) contribute to
their superior performance, this massive scale makes them inefficient and
memory-hungry. Thus, they are hard to deploy on commodity hardware, such as one
single GPU. Given the memory and power constraints of such devices, model
compression methods are widely employed to reduce both the model size and
inference latency, which essentially trades off model quality in return for
improved efficiency. Thus, optimizing this accuracy-efficiency trade-off is
crucial for the LLM deployment on commodity hardware. In this paper, we
introduce a new perspective to optimize this trade-off by prompting compressed
models. Specifically, we first observe that for certain questions, the
generation quality of a compressed LLM can be significantly improved by adding
carefully designed hard prompts, though this isn't the case for all questions.
Based on this observation, we propose a soft prompt learning method where we
expose the compressed model to the prompt learning process, aiming to enhance
the performance of prompts. Our experimental analysis suggests our soft prompt
strategy greatly improves the performance of the 8x compressed LLaMA-7B model
(with a joint 4-bit quantization and 50% weight pruning compression), allowing
them to match their uncompressed counterparts on popular benchmarks. Also, we
demonstrate that these learned prompts can be transferred across various
datasets, tasks, and compression levels. Hence with this transferability, we
can stitch the soft prompt to a newly compressed model to improve the test-time
accuracy in an ``in-situ'' way.


7. [PaLM 2 Technical Report](http://arxiv.org/abs/2305.10403v3), Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, Yonghui Wu, 17-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We introduce PaLM 2, a new state-of-the-art language model that has better
multilingual and reasoning capabilities and is more compute-efficient than its
predecessor PaLM. PaLM 2 is a Transformer-based model trained using a mixture
of objectives. Through extensive evaluations on English and multilingual
language, and reasoning tasks, we demonstrate that PaLM 2 has significantly
improved quality on downstream tasks across different model sizes, while
simultaneously exhibiting faster and more efficient inference compared to PaLM.
This improved efficiency enables broader deployment while also allowing the
model to respond faster, for a more natural pace of interaction. PaLM 2
demonstrates robust reasoning capabilities exemplified by large improvements
over PaLM on BIG-Bench and other reasoning tasks. PaLM 2 exhibits stable
performance on a suite of responsible AI evaluations, and enables
inference-time control over toxicity without additional overhead or impact on
other capabilities. Overall, PaLM 2 achieves state-of-the-art performance
across a diverse set of tasks and capabilities.
  When discussing the PaLM 2 family, it is important to distinguish between
pre-trained models (of various sizes), fine-tuned variants of these models, and
the user-facing products that use these models. In particular, user-facing
products typically include additional pre- and post-processing steps.
Additionally, the underlying models may evolve over time. Therefore, one should
not expect the performance of user-facing products to exactly match the results
reported in this report.


7. [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](http://arxiv.org/abs/2305.10601v2), Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan, 17-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Language models are increasingly being deployed for general problem solving
across a wide range of tasks, but are still confined to token-level,
left-to-right decision-making processes during inference. This means they can
fall short in tasks that require exploration, strategic lookahead, or where
initial decisions play a pivotal role. To surmount these challenges, we
introduce a new framework for language model inference, Tree of Thoughts (ToT),
which generalizes over the popular Chain of Thought approach to prompting
language models, and enables exploration over coherent units of text (thoughts)
that serve as intermediate steps toward problem solving. ToT allows LMs to
perform deliberate decision making by considering multiple different reasoning
paths and self-evaluating choices to decide the next course of action, as well
as looking ahead or backtracking when necessary to make global choices. Our
experiments show that ToT significantly enhances language models'
problem-solving abilities on three novel tasks requiring non-trivial planning
or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in
Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of
tasks, our method achieved a success rate of 74%. Code repo with all prompts:
https://github.com/princeton-nlp/tree-of-thought-llm.


7. [LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206v1), Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy, 18-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large language models are trained in two stages: (1) unsupervised pretraining
from raw text, to learn general-purpose representations, and (2) large scale
instruction tuning and reinforcement learning, to better align to end tasks and
user preferences. We measure the relative importance of these two stages by
training LIMA, a 65B parameter LLaMa language model fine-tuned with the
standard supervised loss on only 1,000 carefully curated prompts and responses,
without any reinforcement learning or human preference modeling. LIMA
demonstrates remarkably strong performance, learning to follow specific
response formats from only a handful of examples in the training data,
including complex queries that range from planning trip itineraries to
speculating about alternate history. Moreover, the model tends to generalize
well to unseen tasks that did not appear in the training data. In a controlled
human study, responses from LIMA are either equivalent or strictly preferred to
GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard
and 65% versus DaVinci003, which was trained with human feedback. Taken
together, these results strongly suggest that almost all knowledge in large
language models is learned during pretraining, and only limited instruction
tuning data is necessary to teach models to produce high quality output.


7. [Reasoning Implicit Sentiment with Chain-of-Thought Prompting](http://arxiv.org/abs/2305.11255v4), Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua, 18-05-2023
     ### Categories
     Computation and Language
    ### Abstract
    While sentiment analysis systems try to determine the sentiment polarities of
given targets based on the key opinion expressions in input texts, in implicit
sentiment analysis (ISA) the opinion cues come in an implicit and obscure
manner. Thus detecting implicit sentiment requires the common-sense and
multi-hop reasoning ability to infer the latent intent of opinion. Inspired by
the recent chain-of-thought (CoT) idea, in this work we introduce a Three-hop
Reasoning (THOR) CoT framework to mimic the human-like reasoning process for
ISA. We design a three-step prompting principle for THOR to step-by-step induce
the implicit aspect, opinion, and finally the sentiment polarity. Our
THOR+Flan-T5 (11B) pushes the state-of-the-art (SoTA) by over 6% F1 on
supervised setup. More strikingly, THOR+GPT3 (175B) boosts the SoTA by over 50%
F1 on zero-shot setting. Our code is open at
https://github.com/scofield7419/THOR-ISA.


7. [Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study](http://arxiv.org/abs/2305.13860v1), Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, Yang Liu, 23-05-2023
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    Large Language Models (LLMs), like ChatGPT, have demonstrated vast potential
but also introduce challenges related to content constraints and potential
misuse. Our study investigates three key research questions: (1) the number of
different prompt types that can jailbreak LLMs, (2) the effectiveness of
jailbreak prompts in circumventing LLM constraints, and (3) the resilience of
ChatGPT against these jailbreak prompts. Initially, we develop a classification
model to analyze the distribution of existing prompts, identifying ten distinct
patterns and three categories of jailbreak prompts. Subsequently, we assess the
jailbreak capability of prompts with ChatGPT versions 3.5 and 4.0, utilizing a
dataset of 3,120 jailbreak questions across eight prohibited scenarios.
Finally, we evaluate the resistance of ChatGPT against jailbreak prompts,
finding that the prompts can consistently evade the restrictions in 40 use-case
scenarios. The study underscores the importance of prompt structures in
jailbreaking LLMs and discusses the challenges of robust jailbreak prompt
generation and prevention.


7. [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1), Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer, 23-05-2023
     ### Categories
     Machine Learning
    ### Abstract
    We present QLoRA, an efficient finetuning approach that reduces memory usage
enough to finetune a 65B parameter model on a single 48GB GPU while preserving
full 16-bit finetuning task performance. QLoRA backpropagates gradients through
a frozen, 4-bit quantized pretrained language model into Low Rank
Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all
previous openly released models on the Vicuna benchmark, reaching 99.3% of the
performance level of ChatGPT while only requiring 24 hours of finetuning on a
single GPU. QLoRA introduces a number of innovations to save memory without
sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is
information theoretically optimal for normally distributed weights (b) double
quantization to reduce the average memory footprint by quantizing the
quantization constants, and (c) paged optimziers to manage memory spikes. We
use QLoRA to finetune more than 1,000 models, providing a detailed analysis of
instruction following and chatbot performance across 8 instruction datasets,
multiple model types (LLaMA, T5), and model scales that would be infeasible to
run with regular finetuning (e.g. 33B and 65B parameter models). Our results
show that QLoRA finetuning on a small high-quality dataset leads to
state-of-the-art results, even when using smaller models than the previous
SoTA. We provide a detailed analysis of chatbot performance based on both human
and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable
alternative to human evaluation. Furthermore, we find that current chatbot
benchmarks are not trustworthy to accurately evaluate the performance levels of
chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to
ChatGPT. We release all of our models and code, including CUDA kernels for
4-bit training.


7. [The CoT Collection: Improving Zero-shot and Few-shot Learning of
  Language Models via Chain-of-Thought Fine-Tuning](http://arxiv.org/abs/2305.14045v2), Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, Minjoon Seo, 23-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Language models (LMs) with less than 100B parameters are known to perform
poorly on chain-of-thought (CoT) reasoning in contrast to large LMs when
solving unseen tasks. In this work, we aim to equip smaller LMs with the
step-by-step reasoning capability by instruction tuning with CoT rationales. In
order to achieve this goal, we first introduce a new instruction-tuning dataset
called the CoT Collection, which augments the existing Flan Collection
(including only 9 CoT tasks) with additional 1.84 million rationales across
1,060 tasks. We show that CoT fine-tuning Flan-T5 (3B & 11B) with CoT
Collection enables smaller LMs to have better CoT capabilities on unseen tasks.
On the BIG-Bench-Hard (BBH) benchmark, we report an average improvement of
+4.34% (Flan-T5 3B) and +2.60% (Flan-T5 11B), in terms of zero-shot task
accuracy. Furthermore, we show that instruction tuning with CoT Collection
allows LMs to possess stronger few-shot learning capabilities on 4
domain-specific tasks, resulting in an improvement of +2.24% (Flan-T5 3B) and
+2.37% (Flan-T5 11B), even outperforming ChatGPT utilizing demonstrations until
the max length by a +13.98% margin. Our code, the CoT Collection data, and
model checkpoints are publicly available.


7. [ExpertPrompting: Instructing Large Language Models to be Distinguished
  Experts](http://arxiv.org/abs/2305.14688v1), Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, Zhendong Mao, 24-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    The answering quality of an aligned large language model (LLM) can be
drastically improved if treated with proper crafting of prompts. In this paper,
we propose ExpertPrompting to elicit the potential of LLMs to answer as
distinguished experts. We first utilize In-Context Learning to automatically
synthesize detailed and customized descriptions of the expert identity for each
specific instruction, and then ask LLMs to provide answer conditioned on such
agent background. Based on this augmented prompting strategy, we produce a new
set of instruction-following data using GPT-3.5, and train a competitive
open-source chat assistant called ExpertLLaMA. We employ GPT4-based evaluation
to show that 1) the expert data is of significantly higher quality than vanilla
answers, and 2) ExpertLLaMA outperforms existing open-source opponents and
achieves 96\% of the original ChatGPT's capability. All data and the
ExpertLLaMA model will be made publicly available at
\url{https://github.com/OFA-Sys/ExpertLLaMA}.


7. [Reasoning with Language Model is Planning with World Model](http://arxiv.org/abs/2305.14992v2), Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu, 24-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large language models (LLMs) have shown remarkable reasoning capabilities,
especially when prompted to generate intermediate reasoning steps (e.g.,
Chain-of-Thought, CoT). However, LLMs can still struggle with problems that are
easy for humans, such as generating action plans for executing tasks in a given
environment, or performing complex math, logical, and commonsense reasoning.
The deficiency stems from the key fact that LLMs lack an internal
$\textit{world model}$ to predict the world $\textit{state}$ (e.g., environment
status, intermediate variable values) and simulate long-term outcomes of
actions. This prevents LLMs from performing deliberate planning akin to human
brains, which involves exploring alternative reasoning paths, anticipating
future states and rewards, and iteratively refining existing reasoning steps.
To overcome the limitations, we propose a new LLM reasoning framework,
$\underline{R}$easoning vi$\underline{a}$ $\underline{P}$lanning
$\textbf{(RAP)}$. RAP repurposes the LLM as both a world model and a reasoning
agent, and incorporates a principled planning algorithm (based on Monto Carlo
Tree Search) for strategic exploration in the vast reasoning space. During
reasoning, the LLM (as agent) incrementally builds a reasoning tree under the
guidance of the LLM (as world model) and task-specific rewards, and obtains a
high-reward reasoning path efficiently with a proper balance between
exploration $\textit{vs.}$ exploitation. We apply RAP to a variety of
challenging reasoning problems including plan generation, math reasoning, and
logical inference. Empirical results on these tasks demonstrate the superiority
of RAP over various strong baselines, including CoT and least-to-most prompting
with self-consistency. RAP on LLAMA-33B surpasses CoT on GPT-4 with 33%
relative improvement in a plan generation setting.


7. [SPRING: Studying the Paper and Reasoning to Play Games](http://arxiv.org/abs/2305.15486v3), Yue Wu, Shrimai Prabhumoye, So Yeon Min, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Tom Mitchell, Yuanzhi Li, 24-05-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    Open-world survival games pose significant challenges for AI algorithms due
to their multi-tasking, deep exploration, and goal prioritization requirements.
Despite reinforcement learning (RL) being popular for solving games, its high
sample complexity limits its effectiveness in complex open-world games like
Crafter or Minecraft. We propose a novel approach, SPRING, to read the game's
original academic paper and use the knowledge learned to reason and play the
game through a large language model (LLM). Prompted with the LaTeX source as
game context and a description of the agent's current observation, our SPRING
framework employs a directed acyclic graph (DAG) with game-related questions as
nodes and dependencies as edges. We identify the optimal action to take in the
environment by traversing the DAG and calculating LLM responses for each node
in topological order, with the LLM's answer to final node directly translating
to environment actions. In our experiments, we study the quality of in-context
"reasoning" induced by different forms of prompts under the setting of the
Crafter open-world environment. Our experiments suggest that LLMs, when
prompted with consistent chain-of-thought, have great potential in completing
sophisticated high-level trajectories. Quantitatively, SPRING with GPT-4
outperforms all state-of-the-art RL baselines, trained for 1M steps, without
any training. Finally, we show the potential of games as a test bed for LLMs.


7. [MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of
  Thought Prompting](http://arxiv.org/abs/2305.16896v1), Tatsuro Inaba, Hirokazu Kiyomaru, Fei Cheng, Sadao Kurohashi, 26-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large language models (LLMs) have achieved impressive performance on various
reasoning tasks. To further improve the performance, we propose MultiTool-CoT,
a novel framework that leverages chain-of-thought (CoT) prompting to
incorporate multiple external tools, such as a calculator and a knowledge
retriever, during the reasoning process. We apply MultiTool-CoT to the Task 2
dataset of NumGLUE, which requires both numerical reasoning and domain-specific
knowledge. The experiments show that our method significantly outperforms
strong baselines and achieves state-of-the-art performance.


7. [Tab-CoT: Zero-shot Tabular Chain of Thought](http://arxiv.org/abs/2305.17812v1), Ziqi Jin, Wei Lu, 28-05-2023
     ### Categories
     Computation and Language
    ### Abstract
    The chain-of-though (CoT) prompting methods were successful in various
natural language processing (NLP) tasks thanks to their ability to unveil the
underlying complex reasoning processes. Such reasoning processes typically
exhibit implicitly structured steps. Recent efforts also started investigating
methods to encourage more explicitly structured reasoning procedures to be
captured. In this work, we propose Tab-CoT, a novel tabular-format CoT
prompting method, which allows the complex reasoning process to be explicitly
modelled in a highly structured manner. Despite its simplicity, we show that
our approach is capable of performing reasoning across multiple dimensions
(i.e., both rows and columns). We demonstrate our approach's strong zero-shot
and few-shot capabilities through extensive experiments on a range of reasoning
tasks.


7. [Less Likely Brainstorming: Using Language Models to Generate Alternative
  Hypotheses](http://arxiv.org/abs/2305.19339v1), Liyan Tang, Yifan Peng, Yanshan Wang, Ying Ding, Greg Durrett, Justin F. Rousseau, 30-05-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    A human decision-maker benefits the most from an AI assistant that corrects
for their biases. For problems such as generating interpretation of a radiology
report given findings, a system predicting only highly likely outcomes may be
less useful, where such outcomes are already obvious to the user. To alleviate
biases in human decision-making, it is worth considering a broad differential
diagnosis, going beyond the most likely options. We introduce a new task, "less
likely brainstorming," that asks a model to generate outputs that humans think
are relevant but less likely to happen. We explore the task in two settings: a
brain MRI interpretation generation setting and an everyday commonsense
reasoning setting. We found that a baseline approach of training with less
likely hypotheses as targets generates outputs that humans evaluate as either
likely or irrelevant nearly half of the time; standard MLE training is not
effective. To tackle this problem, we propose a controlled text generation
method that uses a novel contrastive learning strategy to encourage models to
differentiate between generating likely and less likely outputs according to
humans. We compare our method with several state-of-the-art controlled text
generation models via automatic and human evaluations and show that our models'
capability of generating less likely outputs is improved.


7. [Chain-Of-Thought Prompting Under Streaming Batch: A Case Study](http://arxiv.org/abs/2306.00550v1), Yuxin Tang, 01-06-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Recently, Large Language Models (LLMs) have demonstrated remarkable
capabilities. Chain-of-Thought (CoT) has been proposed as a way of assisting
LLMs in performing complex reasoning. However, developing effective prompts can
be a challenging and labor-intensive task. Many studies come out of some way to
automatically construct CoT from test data. Most of them assume that all test
data is visible before testing and only select a small subset to generate
rationales, which is an unrealistic assumption. In this paper, we present a
case study on how to construct and optimize chain-of-thought prompting using
batch data in streaming settings.


7. [ReviewerGPT? An Exploratory Study on Using Large Language Models for
  Paper Reviewing](http://arxiv.org/abs/2306.00622v1), Ryan Liu, Nihar B. Shah, 01-06-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Given the rapid ascent of large language models (LLMs), we study the
question: (How) can large language models help in reviewing of scientific
papers or proposals? We first conduct some pilot studies where we find that (i)
GPT-4 outperforms other LLMs (Bard, Vicuna, Koala, Alpaca, LLaMa, Dolly,
OpenAssistant, StableLM), and (ii) prompting with a specific question (e.g., to
identify errors) outperforms prompting to simply write a review. With these
insights, we study the use of LLMs (specifically, GPT-4) for three tasks:
  1. Identifying errors: We construct 13 short computer science papers each
with a deliberately inserted error, and ask the LLM to check for the
correctness of these papers. We observe that the LLM finds errors in 7 of them,
spanning both mathematical and conceptual errors.
  2. Verifying checklists: We task the LLM to verify 16 closed-ended checklist
questions in the respective sections of 15 NeurIPS 2022 papers. We find that
across 119 {checklist question, paper} pairs, the LLM had an 86.6% accuracy.
  3. Choosing the "better" paper: We generate 10 pairs of abstracts,
deliberately designing each pair in such a way that one abstract was clearly
superior than the other. The LLM, however, struggled to discern these
relatively straightforward distinctions accurately, committing errors in its
evaluations for 6 out of the 10 pairs.
  Based on these experiments, we think that LLMs have a promising use as
reviewing assistants for specific reviewing tasks, but not (yet) for complete
evaluations of papers or proposals.


7. [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora
  with Web Data, and Web Data Only](http://arxiv.org/abs/2306.01116v1), Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay, 01-06-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models are commonly trained on a mixture of filtered web data
and curated high-quality corpora, such as social media conversations, books, or
technical papers. This curation process is believed to be necessary to produce
performant models with broad zero-shot generalization abilities. However, as
larger models requiring pretraining on trillions of tokens are considered, it
is unclear how scalable is curation and whether we will run out of unique
high-quality data soon. At variance with previous beliefs, we show that
properly filtered and deduplicated web data alone can lead to powerful models;
even significantly outperforming models from the state-of-the-art trained on
The Pile. Despite extensive filtering, the high-quality data we extract from
the web is still plentiful, and we are able to obtain five trillion tokens from
CommonCrawl. We publicly release an extract of 600 billion tokens from our
RefinedWeb dataset, and 1.3/7.5B parameters language models trained on it.


7. [When Large Language Model based Agent Meets User Behavior Analysis: A
  Novel User Simulation Paradigm](http://arxiv.org/abs/2306.02552v2), Lei Wang, Jingsen Zhang, Hao Yang, Zhiyuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, Ji-Rong Wen, 05-06-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    User behavior analysis is crucial in human-centered AI applications. In this
field, the collection of sufficient and high-quality user behavior data has
always been a fundamental yet challenging problem. An intuitive idea to address
this problem is automatically simulating the user behaviors. However, due to
the subjective and complex nature of human cognitive processes, reliably
simulating the user behavior is difficult. Recently, large language models
(LLM) have obtained remarkable successes, showing great potential to achieve
human-like intelligence. We argue that these models present significant
opportunities for reliable user simulation, and have the potential to
revolutionize traditional study paradigms in user behavior analysis. In this
paper, we take recommender system as an example to explore the potential of
using LLM for user simulation. Specifically, we regard each user as an
LLM-based autonomous agent, and let different agents freely communicate, behave
and evolve in a virtual simulator called RecAgent. For comprehensively
simulation, we not only consider the behaviors within the recommender system
(\emph{e.g.}, item browsing and clicking), but also accounts for external
influential factors, such as, friend chatting and social advertisement. Our
simulator contains at most 1000 agents, and each agent is composed of a
profiling module, a memory module and an action module, enabling it to behave
consistently, reasonably and reliably. In addition, to more flexibly operate
our simulator, we also design two global functions including real-human playing
and system intervention. To evaluate the effectiveness of our simulator, we
conduct extensive experiments from both agent and system perspectives. In order
to advance this direction, we have released our project at
{https://github.com/RUC-GSAI/YuLan-Rec}.


7. [Mind2Web: Towards a Generalist Agent for the Web](http://arxiv.org/abs/2306.06070v3), Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, Yu Su, 09-06-2023
     ### Categories
     Computation and Language
    ### Abstract
    We introduce Mind2Web, the first dataset for developing and evaluating
generalist agents for the web that can follow language instructions to complete
complex tasks on any website. Existing datasets for web agents either use
simulated websites or only cover a limited set of websites and tasks, thus not
suitable for generalist web agents. With over 2,000 open-ended tasks collected
from 137 websites spanning 31 domains and crowdsourced action sequences for the
tasks, Mind2Web provides three necessary ingredients for building generalist
web agents: 1) diverse domains, websites, and tasks, 2) use of real-world
websites instead of simulated and simplified ones, and 3) a broad spectrum of
user interaction patterns. Based on Mind2Web, we conduct an initial exploration
of using large language models (LLMs) for building generalist web agents. While
the raw HTML of real-world websites are often too large to be fed to LLMs, we
show that first filtering it with a small LM significantly improves the
effectiveness and efficiency of LLMs. Our solution demonstrates a decent level
of performance, even on websites or entire domains the model has never seen
before, but there is still a substantial room to improve towards truly
generalizable agents. We open-source our dataset, model implementation, and
trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further
research on building a generalist agent for the web.


7. [Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context
  Reasoning with Language Models](http://arxiv.org/abs/2306.06891v1), Soochan Lee, Gunhee Kim, 12-06-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Generating intermediate steps, or Chain of Thought (CoT), is an effective way
to significantly improve language models' (LM) multi-step reasoning capability.
However, the CoT lengths can grow rapidly with the problem complexity, easily
exceeding the maximum context size. Instead of increasing the context limit,
which has already been heavily investigated, we explore an orthogonal
direction: making LMs divide a problem into multiple contexts. We propose a new
inference framework, called Recursion of Thought (RoT), which introduces
several special tokens that the models can output to trigger context-related
operations. Extensive experiments with multiple architectures including GPT-3
show that RoT dramatically improves LMs' inference capability to solve
problems, whose solution consists of hundreds of thousands of tokens.


7. [Textbooks Are All You Need](http://arxiv.org/abs/2306.11644v2), Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li, 20-06-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    We introduce phi-1, a new large language model for code, with significantly
smaller size than competing models: phi-1 is a Transformer-based model with
1.3B parameters, trained for 4 days on 8 A100s, using a selection of ``textbook
quality" data from the web (6B tokens) and synthetically generated textbooks
and exercises with GPT-3.5 (1B tokens). Despite this small scale, phi-1 attains
pass@1 accuracy 50.6% on HumanEval and 55.5% on MBPP. It also displays
surprising emergent properties compared to phi-1-base, our model before our
finetuning stage on a dataset of coding exercises, and phi-1-small, a smaller
model with 350M parameters trained with the same pipeline as phi-1 that still
achieves 45% on HumanEval.


7. [Large Language Models Understand and Can be Enhanced by Emotional
  Stimuli](http://arxiv.org/abs/2307.11760v7), Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, Xing Xie, 14-07-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Emotional intelligence significantly impacts our daily behaviors and
interactions. Although Large Language Models (LLMs) are increasingly viewed as
a stride toward artificial general intelligence, exhibiting impressive
performance in numerous tasks, it is still uncertain if LLMs can genuinely
grasp psychological emotional stimuli. Understanding and responding to
emotional cues gives humans a distinct advantage in problem-solving. In this
paper, we take the first step towards exploring the ability of LLMs to
understand emotional stimuli. To this end, we first conduct automatic
experiments on 45 tasks using various LLMs, including Flan-T5-Large, Vicuna,
Llama 2, BLOOM, ChatGPT, and GPT-4. Our tasks span deterministic and generative
applications that represent comprehensive evaluation scenarios. Our automatic
experiments show that LLMs have a grasp of emotional intelligence, and their
performance can be improved with emotional prompts (which we call
"EmotionPrompt" that combines the original prompt with emotional stimuli),
e.g., 8.00% relative performance improvement in Instruction Induction and 115%
in BIG-Bench. In addition to those deterministic tasks that can be
automatically evaluated using existing metrics, we conducted a human study with
106 participants to assess the quality of generative tasks using both vanilla
and emotional prompts. Our human study results demonstrate that EmotionPrompt
significantly boosts the performance of generative tasks (10.9% average
improvement in terms of performance, truthfulness, and responsibility metrics).
We provide an in-depth discussion regarding why EmotionPrompt works for LLMs
and the factors that may influence its performance. We posit that EmotionPrompt
heralds a novel avenue for exploring interdisciplinary knowledge for human-LLMs
interaction.


7. [Llama 2: Open Foundation and Fine-Tuned Chat Models](http://arxiv.org/abs/2307.09288v2), Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom, 18-07-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In this work, we develop and release Llama 2, a collection of pretrained and
fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70
billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for
dialogue use cases. Our models outperform open-source chat models on most
benchmarks we tested, and based on our human evaluations for helpfulness and
safety, may be a suitable substitute for closed-source models. We provide a
detailed description of our approach to fine-tuning and safety improvements of
Llama 2-Chat in order to enable the community to build on our work and
contribute to the responsible development of LLMs.


7. [A Real-World WebAgent with Planning, Long Context Understanding, and
  Program Synthesis](http://arxiv.org/abs/2307.12856v3), Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, Aleksandra Faust, 24-07-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Pre-trained large language models (LLMs) have recently achieved better
generalization and sample efficiency in autonomous web automation. However, the
performance on real-world websites has still suffered from (1) open domainness,
(2) limited context length, and (3) lack of inductive bias on HTML. We
introduce WebAgent, an LLM-driven agent that learns from self-experience to
complete tasks on real websites following natural language instructions.
WebAgent plans ahead by decomposing instructions into canonical
sub-instructions, summarizes long HTML documents into task-relevant snippets,
and acts on websites via Python programs generated from those. We design
WebAgent with Flan-U-PaLM, for grounded code generation, and HTML-T5, new
pre-trained LLMs for long HTML documents using local and global attention
mechanisms and a mixture of long-span denoising objectives, for planning and
summarization. We empirically demonstrate that our modular recipe improves the
success on real websites by over 50%, and that HTML-T5 is the best model to
solve various HTML understanding tasks; achieving 18.7% higher success rate
than the prior method on MiniWoB web automation benchmark, and SoTA performance
on Mind2Web, an offline task planning evaluation.


7. [WebArena: A Realistic Web Environment for Building Autonomous Agents](http://arxiv.org/abs/2307.13854v3), Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig, 25-07-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    With advances in generative AI, there is now potential for autonomous agents
to manage daily tasks via natural language commands. However, current agents
are primarily created and tested in simplified synthetic environments, leading
to a disconnect with real-world scenarios. In this paper, we build an
environment for language-guided agents that is highly realistic and
reproducible. Specifically, we focus on agents that perform tasks on the web,
and create an environment with fully functional websites from four common
domains: e-commerce, social forum discussions, collaborative software
development, and content management. Our environment is enriched with tools
(e.g., a map) and external knowledge bases (e.g., user manuals) to encourage
human-like task-solving. Building upon our environment, we release a set of
benchmark tasks focusing on evaluating the functional correctness of task
completions. The tasks in our benchmark are diverse, long-horizon, and designed
to emulate tasks that humans routinely perform on the internet. We experiment
with several baseline agents, integrating recent techniques such as reasoning
before acting. The results demonstrate that solving complex tasks is
challenging: our best GPT-4-based agent only achieves an end-to-end task
success rate of 14.41%, significantly lower than the human performance of
78.24%. These results highlight the need for further development of robust
agents, that current state-of-the-art large language models are far from
perfect performance in these real-life tasks, and that WebArena can be used to
measure such progress.


7. [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](http://arxiv.org/abs/2308.00352v5), Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, Jürgen Schmidhuber, 01-08-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Remarkable progress has been made on automated problem solving through
societies of agents based on large language models (LLMs). Existing LLM-based
multi-agent systems can already solve simple dialogue tasks. Solutions to more
complex tasks, however, are complicated through logic inconsistencies due to
cascading hallucinations caused by naively chaining LLMs. Here we introduce
MetaGPT, an innovative meta-programming framework incorporating efficient human
workflows into LLM-based multi-agent collaborations. MetaGPT encodes
Standardized Operating Procedures (SOPs) into prompt sequences for more
streamlined workflows, thus allowing agents with human-like domain expertise to
verify intermediate results and reduce errors. MetaGPT utilizes an assembly
line paradigm to assign diverse roles to various agents, efficiently breaking
down complex tasks into subtasks involving many agents working together. On
collaborative software engineering benchmarks, MetaGPT generates more coherent
solutions than previous chat-based multi-agent systems. Our project can be
found at https://github.com/geekan/MetaGPT


7. [Evaluating ChatGPT text-mining of clinical records for obesity
  monitoring](http://arxiv.org/abs/2308.01666v1), Ivo S. Fins, Heather Davies, Sean Farrell, Jose R. Torres, Gina Pinchbeck, Alan D. Radford, Peter-John Noble, 03-08-2023
     ### Categories
     Computation and Language
    ### Abstract
    Background: Veterinary clinical narratives remain a largely untapped resource
for addressing complex diseases. Here we compare the ability of a large
language model (ChatGPT) and a previously developed regular expression (RegexT)
to identify overweight body condition scores (BCS) in veterinary narratives.
Methods: BCS values were extracted from 4,415 anonymised clinical narratives
using either RegexT or by appending the narrative to a prompt sent to ChatGPT
coercing the model to return the BCS information. Data were manually reviewed
for comparison. Results: The precision of RegexT was higher (100%, 95% CI
94.81-100%) than the ChatGPT (89.3%; 95% CI82.75-93.64%). However, the recall
of ChatGPT (100%. 95% CI 96.18-100%) was considerably higher than that of
RegexT (72.6%, 95% CI 63.92-79.94%). Limitations: Subtle prompt engineering is
needed to improve ChatGPT output. Conclusions: Large language models create
diverse opportunities and, whilst complex, present an intuitive interface to
information but require careful implementation to avoid unpredictable errors.


7. [Cumulative Reasoning with Large Language Models](http://arxiv.org/abs/2308.04371v5), Yifan Zhang, Jingqin Yang, Yang Yuan, Andrew Chi-Chih Yao, 08-08-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    While language models are powerful and versatile, they often fail to address
highly complex problems. This is because solving complex problems requires
deliberate thinking, which has been only minimally guided during training. In
this paper, we propose a new method called Cumulative Reasoning (CR), which
employs language models in a cumulative and iterative manner to emulate human
thought processes. By decomposing tasks into smaller components, CR streamlines
the problem-solving process, rendering it both more manageable and effective.
For logical inference tasks, CR consistently outperforms existing methods with
an improvement up to 9.3%, and achieves an accuracy of 98.04% on the curated
FOLIO wiki dataset. In the context of the Game of 24, CR achieves an accuracy
of 98%, which signifies a substantial enhancement of 24% over the previous
state-of-the-art method. Finally, on the MATH dataset, we establish new
state-of-the-art results with 58.0% overall accuracy, surpassing the previous
best approach by a margin of 4.2%, and achieving 43% relative improvement on
the hardest level 5 problems (22.4% to 32.1%). Additionally, we expand the
concept of Cumulative Reasoning to incorporate a Python code environment,
deliberately omitting external aids such as retrieval and web browsing and
focusing solely on the LLM's intrinsic reasoning capabilities within a Python
code environment. Our experiments in this setting yielded impressive results,
with an overall accuracy of 72.2% on the MATH dataset, significantly
outperforming the PAL method with 38.8% relative improvement. Code is available
at https://github.com/iiis-ai/cumulative-reasoning.


7. [You Only Prompt Once: On the Capabilities of Prompt Learning on Large
  Language Models to Tackle Toxic Content](http://arxiv.org/abs/2308.05596v1), Xinlei He, Savvas Zannettou, Yun Shen, Yang Zhang, 10-08-2023
     ### Categories
     Computation and Language
    ### Abstract
    The spread of toxic content online is an important problem that has adverse
effects on user experience online and in our society at large. Motivated by the
importance and impact of the problem, research focuses on developing solutions
to detect toxic content, usually leveraging machine learning (ML) models
trained on human-annotated datasets. While these efforts are important, these
models usually do not generalize well and they can not cope with new trends
(e.g., the emergence of new toxic terms). Currently, we are witnessing a shift
in the approach to tackling societal issues online, particularly leveraging
large language models (LLMs) like GPT-3 or T5 that are trained on vast corpora
and have strong generalizability. In this work, we investigate how we can use
LLMs and prompt learning to tackle the problem of toxic content, particularly
focusing on three tasks; 1) Toxicity Classification, 2) Toxic Span Detection,
and 3) Detoxification. We perform an extensive evaluation over five model
architectures and eight datasets demonstrating that LLMs with prompt learning
can achieve similar or even better performance compared to models trained on
these specific tasks. We find that prompt learning achieves around 10\%
improvement in the toxicity classification task compared to the baselines,
while for the toxic span detection task we find better performance to the best
baseline (0.643 vs. 0.640 in terms of $F_1$-score). Finally, for the
detoxification task, we find that prompt learning can successfully reduce the
average toxicity score (from 0.775 to 0.213) while preserving semantic meaning.


7. [Large Language Models as Optimizers](http://arxiv.org/abs/2309.03409v2), Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen, 07-09-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Optimization is ubiquitous. While derivative-based algorithms have been
powerful tools for various problems, the absence of gradient imposes challenges
on many real-world applications. In this work, we propose Optimization by
PROmpting (OPRO), a simple and effective approach to leverage large language
models (LLMs) as optimizers, where the optimization task is described in
natural language. In each optimization step, the LLM generates new solutions
from the prompt that contains previously generated solutions with their values,
then the new solutions are evaluated and added to the prompt for the next
optimization step. We first showcase OPRO on linear regression and traveling
salesman problems, then move on to prompt optimization where the goal is to
find instructions that maximize the task accuracy. With a variety of LLMs, we
demonstrate that the best prompts optimized by OPRO outperform human-designed
prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. Code at
https://github.com/google-deepmind/opro.


7. [Large Language Models as Optimizers](http://arxiv.org/abs/2309.03409v2), Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen, 07-09-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Optimization is ubiquitous. While derivative-based algorithms have been
powerful tools for various problems, the absence of gradient imposes challenges
on many real-world applications. In this work, we propose Optimization by
PROmpting (OPRO), a simple and effective approach to leverage large language
models (LLMs) as optimizers, where the optimization task is described in
natural language. In each optimization step, the LLM generates new solutions
from the prompt that contains previously generated solutions with their values,
then the new solutions are evaluated and added to the prompt for the next
optimization step. We first showcase OPRO on linear regression and traveling
salesman problems, then move on to prompt optimization where the goal is to
find instructions that maximize the task accuracy. With a variety of LLMs, we
demonstrate that the best prompts optimized by OPRO outperform human-designed
prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. Code at
https://github.com/google-deepmind/opro.


7. [From Sparse to Dense: GPT-4 Summarization with Chain of Density
  Prompting](http://arxiv.org/abs/2309.04269v1), Griffin Adams, Alexander Fabbri, Faisal Ladhak, Eric Lehman, Noémie Elhadad, 08-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Selecting the ``right'' amount of information to include in a summary is a
difficult task. A good summary should be detailed and entity-centric without
being overly dense and hard to follow. To better understand this tradeoff, we
solicit increasingly dense GPT-4 summaries with what we refer to as a ``Chain
of Density'' (CoD) prompt. Specifically, GPT-4 generates an initial
entity-sparse summary before iteratively incorporating missing salient entities
without increasing the length. Summaries generated by CoD are more abstractive,
exhibit more fusion, and have less of a lead bias than GPT-4 summaries
generated by a vanilla prompt. We conduct a human preference study on 100 CNN
DailyMail articles and find that that humans prefer GPT-4 summaries that are
more dense than those generated by a vanilla prompt and almost as dense as
human written summaries. Qualitative analysis supports the notion that there
exists a tradeoff between informativeness and readability. 500 annotated CoD
summaries, as well as an extra 5,000 unannotated summaries, are freely
available on HuggingFace
(https://huggingface.co/datasets/griffin/chain_of_density).


7. [Textbooks Are All You Need II: phi-1.5 technical report](http://arxiv.org/abs/2309.05463v1), Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee, 11-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We continue the investigation into the power of smaller Transformer-based
language models as initiated by \textbf{TinyStories} -- a 10 million parameter
model that can produce coherent English -- and the follow-up work on
\textbf{phi-1}, a 1.3 billion parameter model with Python coding performance
close to the state-of-the-art. The latter work proposed to use existing Large
Language Models (LLMs) to generate ``textbook quality" data as a way to enhance
the learning process compared to traditional web data. We follow the
``Textbooks Are All You Need" approach, focusing this time on common sense
reasoning in natural language, and create a new 1.3 billion parameter model
named \textbf{phi-1.5}, with performance on natural language tasks comparable
to models 5x larger, and surpassing most non-frontier LLMs on more complex
reasoning tasks such as grade-school mathematics and basic coding. More
generally, \textbf{phi-1.5} exhibits many of the traits of much larger LLMs,
both good -- such as the ability to ``think step by step" or perform some
rudimentary in-context learning -- and bad, including hallucinations and the
potential for toxic and biased generations -- encouragingly though, we are
seeing improvement on that front thanks to the absence of web data. We
open-source \textbf{phi-1.5} to promote further research on these urgent
topics.


7. [Query-Dependent Prompt Evaluation and Optimization with Offline Inverse
  RL](http://arxiv.org/abs/2309.06553v3), Hao Sun, Alihan Hüyük, Mihaela van der Schaar, 13-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    In this study, we aim to enhance the arithmetic reasoning ability of Large
Language Models (LLMs) through zero-shot prompt optimization. We identify a
previously overlooked objective of query dependency in such optimization and
elucidate two ensuing challenges that impede the successful and economical
design of prompt optimization techniques. One primary issue is the absence of
an effective method to evaluate prompts during inference when the golden answer
is unavailable. Concurrently, learning via interactions with the LLMs to
navigate the expansive natural language prompting space proves to be
resource-intensive. To address this, we introduce Prompt-OIRL, which harnesses
offline inverse reinforcement learning to draw insights from offline prompting
demonstration data. Such data exists as by-products when diverse prompts are
benchmarked on open-accessible datasets. With Prompt-OIRL, the query-dependent
prompt optimization objective is achieved by first learning an offline reward
model. This model can evaluate any query-prompt pairs without accessing LLMs.
Subsequently, a best-of-N strategy is deployed to recommend the optimal prompt.
Our experimental evaluations across various LLM scales and arithmetic reasoning
datasets underscore both the efficacy and economic viability of the proposed
approach.


7. [Agents: An Open-source Framework for Autonomous Language Agents](http://arxiv.org/abs/2309.07870v3), Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Xiangru Tang, Ningyu Zhang, Huajun Chen, Peng Cui, Mrinmaya Sachan, 14-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Recent advances on large language models (LLMs) enable researchers and
developers to build autonomous language agents that can automatically solve
various tasks and interact with environments, humans, and other agents using
natural language interfaces. We consider language agents as a promising
direction towards artificial general intelligence and release Agents, an
open-source library with the goal of opening up these advances to a wider
non-specialist audience. Agents is carefully engineered to support important
features including planning, memory, tool usage, multi-agent communication, and
fine-grained symbolic control. Agents is user-friendly as it enables
non-specialists to build, customize, test, tune, and deploy state-of-the-art
autonomous language agents without much coding. The library is also
research-friendly as its modularized design makes it easily extensible for
researchers. Agents is available at https://github.com/aiwaves-cn/agents.


7. [Clinical Text Summarization: Adapting Large Language Models Can
  Outperform Human Experts](http://arxiv.org/abs/2309.07430v3), Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari, 14-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Sifting through vast textual data and summarizing key information from
electronic health records (EHR) imposes a substantial burden on how clinicians
allocate their time. Although large language models (LLMs) have shown immense
promise in natural language processing (NLP) tasks, their efficacy on a diverse
range of clinical summarization tasks has not yet been rigorously demonstrated.
In this work, we apply domain adaptation methods to eight LLMs, spanning six
datasets and four distinct clinical summarization tasks: radiology reports,
patient questions, progress notes, and doctor-patient dialogue. Our thorough
quantitative assessment reveals trade-offs between models and adaptation
methods in addition to instances where recent advances in LLMs may not improve
results. Further, in a clinical reader study with ten physicians, we show that
summaries from our best-adapted LLMs are preferable to human summaries in terms
of completeness and correctness. Our ensuing qualitative analysis highlights
challenges faced by both LLMs and human experts. Lastly, we correlate
traditional quantitative NLP metrics with reader study scores to enhance our
understanding of how these metrics align with physician preferences. Our
research marks the first evidence of LLMs outperforming human experts in
clinical text summarization across multiple tasks. This implies that
integrating LLMs into clinical workflows could alleviate documentation burden,
empowering clinicians to focus more on personalized patient care and the
inherently human aspects of medicine.


7. [The Rise and Potential of Large Language Model Based Agents: A Survey](http://arxiv.org/abs/2309.07864v3), Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui, 14-09-2023
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    For a long time, humanity has pursued artificial intelligence (AI) equivalent
to or surpassing the human level, with AI agents considered a promising vehicle
for this pursuit. AI agents are artificial entities that sense their
environment, make decisions, and take actions. Many efforts have been made to
develop intelligent agents, but they mainly focus on advancement in algorithms
or training strategies to enhance specific capabilities or performance on
particular tasks. Actually, what the community lacks is a general and powerful
model to serve as a starting point for designing AI agents that can adapt to
diverse scenarios. Due to the versatile capabilities they demonstrate, large
language models (LLMs) are regarded as potential sparks for Artificial General
Intelligence (AGI), offering hope for building general AI agents. Many
researchers have leveraged LLMs as the foundation to build AI agents and have
achieved significant progress. In this paper, we perform a comprehensive survey
on LLM-based agents. We start by tracing the concept of agents from its
philosophical origins to its development in AI, and explain why LLMs are
suitable foundations for agents. Building upon this, we present a general
framework for LLM-based agents, comprising three main components: brain,
perception, and action, and the framework can be tailored for different
applications. Subsequently, we explore the extensive applications of LLM-based
agents in three aspects: single-agent scenarios, multi-agent scenarios, and
human-agent cooperation. Following this, we delve into agent societies,
exploring the behavior and personality of LLM-based agents, the social
phenomena that emerge from an agent society, and the insights they offer for
human society. Finally, we discuss several key topics and open problems within
the field. A repository for the related papers at
https://github.com/WooooDyy/LLM-Agent-Paper-List.


7. [Connecting Large Language Models with Evolutionary Algorithms Yields
  Powerful Prompt Optimizers](http://arxiv.org/abs/2309.08532v1), Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang, 15-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) excel in various tasks, but they rely on
carefully crafted prompts that often demand substantial human effort. To
automate this process, in this paper, we propose a novel framework for discrete
prompt optimization, called EvoPrompt, which borrows the idea of evolutionary
algorithms (EAs) as they exhibit good performance and fast convergence. To
enable EAs to work on discrete prompts, which are natural language expressions
that need to be coherent and human-readable, we connect LLMs with EAs. This
approach allows us to simultaneously leverage the powerful language processing
capabilities of LLMs and the efficient optimization performance of EAs.
Specifically, abstaining from any gradients or parameters, EvoPrompt starts
from a population of prompts and iteratively generates new prompts with LLMs
based on the evolutionary operators, improving the population based on the
development set. We optimize prompts for both closed- and open-source LLMs
including GPT-3.5 and Alpaca, on 9 datasets spanning language understanding and
generation tasks. EvoPrompt significantly outperforms human-engineered prompts
and existing methods for automatic prompt generation by up to 25% and 14%
respectively. Furthermore, EvoPrompt demonstrates that connecting LLMs with EAs
creates synergies, which could inspire further research on the combination of
LLMs and conventional algorithms.


7. [PDFTriage: Question Answering over Long, Structured Documents](http://arxiv.org/abs/2309.08872v2), Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A. Rossi, Franck Dernoncourt, 16-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large Language Models (LLMs) have issues with document question answering
(QA) in situations where the document is unable to fit in the small context
length of an LLM. To overcome this issue, most existing works focus on
retrieving the relevant context from the document, representing them as plain
text. However, documents such as PDFs, web pages, and presentations are
naturally structured with different pages, tables, sections, and so on.
Representing such structured documents as plain text is incongruous with the
user's mental model of these documents with rich structure. When a system has
to query the document for context, this incongruity is brought to the fore, and
seemingly trivial questions can trip up the QA system. To bridge this
fundamental gap in handling structured documents, we propose an approach called
PDFTriage that enables models to retrieve the context based on either structure
or content. Our experiments demonstrate the effectiveness of the proposed
PDFTriage-augmented models across several classes of questions where existing
retrieval-augmented LLMs fail. To facilitate further research on this
fundamental problem, we release our benchmark dataset consisting of 900+
human-generated questions over 80 structured documents from 10 different
categories of question types for document QA. Our code and datasets will be
released soon on Github.


7. [OWL: A Large Language Model for IT Operations](http://arxiv.org/abs/2309.09298v1), Hongcheng Guo, Jian Yang, Jiaheng Liu, Liqun Yang, Linzheng Chai, Jiaqi Bai, Junran Peng, Xiaorong Hu, Chao Chen, Dongfeng Zhang, Xu Shi, Tieqiao Zheng, Liangfan Zheng, Bo Zhang, Ke Xu, Zhoujun Li, 17-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    With the rapid development of IT operations, it has become increasingly
crucial to efficiently manage and analyze large volumes of data for practical
applications. The techniques of Natural Language Processing (NLP) have shown
remarkable capabilities for various tasks, including named entity recognition,
machine translation and dialogue systems. Recently, Large Language Models
(LLMs) have achieved significant improvements across various NLP downstream
tasks. However, there is a lack of specialized LLMs for IT operations. In this
paper, we introduce the OWL, a large language model trained on our collected
OWL-Instruct dataset with a wide range of IT-related information, where the
mixture-of-adapter strategy is proposed to improve the parameter-efficient
tuning across different domains or tasks. Furthermore, we evaluate the
performance of our OWL on the OWL-Bench established by us and open IT-related
benchmarks. OWL demonstrates superior performance results on IT tasks, which
outperforms existing models by significant margins. Moreover, we hope that the
findings of our work will provide more insights to revolutionize the techniques
of IT operations with specialized LLMs.


7. [Investigating Zero- and Few-shot Generalization in Fact Verification](http://arxiv.org/abs/2309.09444v1), Liangming Pan, Yunxiang Zhang, Min-Yen Kan, 18-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    In this paper, we explore zero- and few-shot generalization for fact
verification (FV), which aims to generalize the FV model trained on
well-resourced domains (e.g., Wikipedia) to low-resourced domains that lack
human annotations. To this end, we first construct a benchmark dataset
collection which contains 11 FV datasets representing 6 domains. We conduct an
empirical analysis of generalization across these FV datasets, finding that
current models generalize poorly. Our analysis reveals that several factors
affect generalization, including dataset size, length of evidence, and the type
of claims. Finally, we show that two directions of work improve generalization:
1) incorporating domain knowledge via pretraining on specialized domains, and
2) automatically generating training data via claim generation.


7. [LLM4Jobs: Unsupervised occupation extraction and standardization
  leveraging Large Language Models](http://arxiv.org/abs/2309.09708v2), Nan Li, Bo Kang, Tijl De Bie, 18-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Automated occupation extraction and standardization from free-text job
postings and resumes are crucial for applications like job recommendation and
labor market policy formation. This paper introduces LLM4Jobs, a novel
unsupervised methodology that taps into the capabilities of large language
models (LLMs) for occupation coding. LLM4Jobs uniquely harnesses both the
natural language understanding and generation capacities of LLMs. Evaluated on
rigorous experimentation on synthetic and real-world datasets, we demonstrate
that LLM4Jobs consistently surpasses unsupervised state-of-the-art benchmarks,
demonstrating its versatility across diverse datasets and granularities. As a
side result of our work, we present both synthetic and real-world datasets,
which may be instrumental for subsequent research in this domain. Overall, this
investigation highlights the promise of contemporary LLMs for the intricate
task of occupation extraction and standardization, laying the foundation for a
robust and adaptable framework relevant to both research and industrial
contexts.


7. [MindAgent: Emergent Gaming Interaction](http://arxiv.org/abs/2309.09971v2), Ran Gong, Qiuyuan Huang, Xiaojian Ma, Hoi Vo, Zane Durante, Yusuke Noda, Zilong Zheng, Song-Chun Zhu, Demetri Terzopoulos, Li Fei-Fei, Jianfeng Gao, 18-09-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) have the capacity of performing complex
scheduling in a multi-agent system and can coordinate these agents into
completing sophisticated tasks that require extensive collaboration. However,
despite the introduction of numerous gaming frameworks, the community has
insufficient benchmarks towards building general multi-agents collaboration
infrastructure that encompass both LLM and human-NPCs collaborations. In this
work, we propose a novel infrastructure - MindAgent - to evaluate planning and
coordination emergent capabilities for gaming interaction. In particular, our
infrastructure leverages existing gaming framework, to i) require understanding
of the coordinator for a multi-agent system, ii) collaborate with human players
via un-finetuned proper instructions, and iii) establish an in-context learning
on few-shot prompt with feedback. Furthermore, we introduce CUISINEWORLD, a new
gaming scenario and related benchmark that dispatch a multi-agent collaboration
efficiency and supervise multiple agents playing the game simultaneously. We
conduct comprehensive evaluations with new auto-metric CoS for calculating the
collaboration efficiency. Finally, our infrastructure can be deployed into
real-world gaming scenarios in a customized VR version of CUISINEWORLD and
adapted in existing broader Minecraft gaming domain. We hope our findings on
LLMs and the new infrastructure for general-purpose scheduling and coordination
can help shed light on how such skills can be obtained by learning from large
language corpora.


7. [PolicyGPT: Automated Analysis of Privacy Policies with Large Language
  Models](http://arxiv.org/abs/2309.10238v1), Chenhao Tang, Zhengliang Liu, Chong Ma, Zihao Wu, Yiwei Li, Wei Liu, Dajiang Zhu, Quanzheng Li, Xiang Li, Tianming Liu, Lei Fan, 19-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Privacy policies serve as the primary conduit through which online service
providers inform users about their data collection and usage procedures.
However, in a bid to be comprehensive and mitigate legal risks, these policy
documents are often quite verbose. In practical use, users tend to click the
Agree button directly rather than reading them carefully. This practice exposes
users to risks of privacy leakage and legal issues. Recently, the advent of
Large Language Models (LLM) such as ChatGPT and GPT-4 has opened new
possibilities for text analysis, especially for lengthy documents like privacy
policies. In this study, we investigate a privacy policy text analysis
framework PolicyGPT based on the LLM. This framework was tested using two
datasets. The first dataset comprises of privacy policies from 115 websites,
which were meticulously annotated by legal experts, categorizing each segment
into one of 10 classes. The second dataset consists of privacy policies from
304 popular mobile applications, with each sentence manually annotated and
classified into one of another 10 categories. Under zero-shot learning
conditions, PolicyGPT demonstrated robust performance. For the first dataset,
it achieved an accuracy rate of 97%, while for the second dataset, it attained
an 87% accuracy rate, surpassing that of the baseline machine learning and
neural network models.


7. [Prompt, Condition, and Generate: Classification of Unsupported Claims
  with In-Context Learning](http://arxiv.org/abs/2309.10359v1), Peter Ebert Christensen, Srishti Yadav, Serge Belongie, 19-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Unsupported and unfalsifiable claims we encounter in our daily lives can
influence our view of the world. Characterizing, summarizing, and -- more
generally -- making sense of such claims, however, can be challenging. In this
work, we focus on fine-grained debate topics and formulate a new task of
distilling, from such claims, a countable set of narratives. We present a
crowdsourced dataset of 12 controversial topics, comprising more than 120k
arguments, claims, and comments from heterogeneous sources, each annotated with
a narrative label. We further investigate how large language models (LLMs) can
be used to synthesise claims using In-Context Learning. We find that generated
claims with supported evidence can be used to improve the performance of
narrative classification models and, additionally, that the same model can
infer the stance and aspect using a few training examples. Such a model can be
useful in applications which rely on narratives , e.g. fact-checking.


7. [Chain-of-Verification Reduces Hallucination in Large Language Models](http://arxiv.org/abs/2309.11495v2), Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston, 20-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Generation of plausible yet incorrect factual information, termed
hallucination, is an unsolved issue in large language models. We study the
ability of language models to deliberate on the responses they give in order to
correct their mistakes. We develop the Chain-of-Verification (CoVe) method
whereby the model first (i) drafts an initial response; then (ii) plans
verification questions to fact-check its draft; (iii) answers those questions
independently so the answers are not biased by other responses; and (iv)
generates its final verified response. In experiments, we show CoVe decreases
hallucinations across a variety of tasks, from list-based questions from
Wikidata, closed book MultiSpanQA and longform text generation.


7. [OpenChat: Advancing Open-source Language Models with Mixed-Quality Data](http://arxiv.org/abs/2309.11235v1), Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, Yang Liu, 20-09-2023
     ### Categories
     Computation and Language
    ### Abstract
    Nowadays, open-source large language models like LLaMA have emerged. Recent
developments have incorporated supervised fine-tuning (SFT) and reinforcement
learning fine-tuning (RLFT) to align these models with human goals. However,
SFT methods treat all training data with mixed quality equally, while RLFT
methods require high-quality pairwise or ranking-based preference data. In this
study, we present a novel framework, named OpenChat, to advance open-source
language models with mixed-quality data. Specifically, we consider the general
SFT training data, consisting of a small amount of expert data mixed with a
large proportion of sub-optimal data, without any preference labels. We propose
the C(onditioned)-RLFT, which regards different data sources as coarse-grained
reward labels and learns a class-conditioned policy to leverage complementary
data quality information. Interestingly, the optimal policy in C-RLFT can be
easily solved through single-stage, RL-free supervised learning, which is
lightweight and avoids costly human preference labeling. Through extensive
experiments on three standard benchmarks, our openchat-13b fine-tuned with
C-RLFT achieves the highest average performance among all 13b open-source
language models. Moreover, we use AGIEval to validate the model generalization
performance, in which only openchat-13b surpasses the base model. Finally, we
conduct a series of analyses to shed light on the effectiveness and robustness
of OpenChat. Our code, data, and models are publicly available at
https://github.com/imoneoi/openchat.


7. [Large Language Model Alignment: A Survey](http://arxiv.org/abs/2309.15025v1), Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong, 26-09-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Recent years have witnessed remarkable progress made in large language models
(LLMs). Such advancements, while garnering significant attention, have
concurrently elicited various concerns. The potential of these models is
undeniably vast; however, they may yield texts that are imprecise, misleading,
or even detrimental. Consequently, it becomes paramount to employ alignment
techniques to ensure these models to exhibit behaviors consistent with human
values.
  This survey endeavors to furnish an extensive exploration of alignment
methodologies designed for LLMs, in conjunction with the extant capability
research in this domain. Adopting the lens of AI alignment, we categorize the
prevailing methods and emergent proposals for the alignment of LLMs into outer
and inner alignment. We also probe into salient issues including the models'
interpretability, and potential vulnerabilities to adversarial attacks. To
assess LLM alignment, we present a wide variety of benchmarks and evaluation
methodologies. After discussing the state of alignment research for LLMs, we
finally cast a vision toward the future, contemplating the promising avenues of
research that lie ahead.
  Our aspiration for this survey extends beyond merely spurring research
interests in this realm. We also envision bridging the gap between the AI
alignment research community and the researchers engrossed in the capability
exploration of LLMs for both capable and safe LLMs.


7. [AutoAgents: A Framework for Automatic Agent Generation](http://arxiv.org/abs/2309.17288v2), Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F. Karlsson, Jie Fu, Yemin Shi, 29-09-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Large language models (LLMs) have enabled remarkable advances in automated
task-solving with multi-agent systems. However, most existing LLM-based
multi-agent approaches rely on predefined agents to handle simple tasks,
limiting the adaptability of multi-agent collaboration to different scenarios.
Therefore, we introduce AutoAgents, an innovative framework that adaptively
generates and coordinates multiple specialized agents to build an AI team
according to different tasks. Specifically, AutoAgents couples the relationship
between tasks and roles by dynamically generating multiple required agents
based on task content and planning solutions for the current task based on the
generated expert agents. Multiple specialized agents collaborate with each
other to efficiently accomplish tasks. Concurrently, an observer role is
incorporated into the framework to reflect on the designated plans and agents'
responses and improve upon them. Our experiments on various benchmarks
demonstrate that AutoAgents generates more coherent and accurate solutions than
the existing multi-agent methods. This underscores the significance of
assigning different roles to different tasks and of team cooperation, offering
new perspectives for tackling complex tasks. The repository of this project is
available at https://github.com/Link-AGI/AutoAgents.


7. [SmartPlay: A Benchmark for LLMs as Intelligent Agents](http://arxiv.org/abs/2310.01557v3), Yue Wu, Xuan Tang, Tom M. Mitchell, Yuanzhi Li, 02-10-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    Recent large language models (LLMs) have demonstrated great potential toward
intelligent agents and next-gen automation, but there currently lacks a
systematic benchmark for evaluating LLMs' abilities as agents. We introduce
SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs
as agents. SmartPlay consists of 6 different games, including
Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique
setting, providing up to 20 evaluation settings and infinite environment
variations. Each game in SmartPlay uniquely challenges a subset of 9 important
capabilities of an intelligent LLM agent, including reasoning with object
dependencies, planning ahead, spatial reasoning, learning from history, and
understanding randomness. The distinction between the set of capabilities each
game test allows us to analyze each capability separately. SmartPlay serves not
only as a rigorous testing ground for evaluating the overall performance of LLM
agents but also as a road-map for identifying gaps in current methodologies. We
release our benchmark at github.com/microsoft/SmartPlay


7. [Can large language models provide useful feedback on research papers? A
  large-scale empirical analysis](http://arxiv.org/abs/2310.01783v1), Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Smith, Yian Yin, Daniel McFarland, James Zou, 03-10-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Expert feedback lays the foundation of rigorous research. However, the rapid
growth of scholarly production and intricate knowledge specialization challenge
the conventional scientific feedback mechanisms. High-quality peer reviews are
increasingly difficult to obtain. Researchers who are more junior or from
under-resourced settings have especially hard times getting timely feedback.
With the breakthrough of large language models (LLM) such as GPT-4, there is
growing interest in using LLMs to generate scientific feedback on research
manuscripts. However, the utility of LLM-generated feedback has not been
systematically studied. To address this gap, we created an automated pipeline
using GPT-4 to provide comments on the full PDFs of scientific papers. We
evaluated the quality of GPT-4's feedback through two large-scale studies. We
first quantitatively compared GPT-4's generated feedback with human peer
reviewer feedback in 15 Nature family journals (3,096 papers in total) and the
ICLR machine learning conference (1,709 papers). The overlap in the points
raised by GPT-4 and by human reviewers (average overlap 30.85% for Nature
journals, 39.23% for ICLR) is comparable to the overlap between two human
reviewers (average overlap 28.58% for Nature journals, 35.25% for ICLR). The
overlap between GPT-4 and human reviewers is larger for the weaker papers. We
then conducted a prospective user study with 308 researchers from 110 US
institutions in the field of AI and computational biology to understand how
researchers perceive feedback generated by our GPT-4 system on their own
papers. Overall, more than half (57.4%) of the users found GPT-4 generated
feedback helpful/very helpful and 82.4% found it more beneficial than feedback
from at least some human reviewers. While our findings show that LLM-generated
feedback can help researchers, we also identify several limitations.


7. [Conversational Health Agents: A Personalized LLM-Powered Agent Framework](http://arxiv.org/abs/2310.02374v3), Mahyar Abbasian, Iman Azimi, Amir M. Rahmani, Ramesh Jain, 03-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    Conversational Health Agents (CHAs) are interactive systems that provide
healthcare services, such as assistance, self-awareness, and diagnosis. Current
CHAs, especially those utilizing Large Language Models (LLMs), primarily focus
on conversation aspects. However, they offer limited agent capabilities
specifically lacking multi-step problem-solving, empathetic conversations, and
multimodal data analysis. Our aim is to overcome these limitations. In this
paper, we propose an LLM-powered framework to empower CHAs to generate a
personalized response for users' healthcare queries. This framework provides
critical thinking, knowledge acquisition, and problem-solving abilities by
integrating healthcare data sources, enabling multilingual and multimodal
conversations, and interacting with various user data analysis tools. We
illustrate the framework's proficiency in handling complex healthcare tasks via
a case study on stress level estimation, showcasing the agent's cognitive and
operational capabilities. Powered by our framework, the CHA can provide
appropriate responses, when the user inquires about their stress level. To
achieve this, it learns to collect photoplethysmogram signals, converts them
into heart rate variability, and interprets them as indicators of stress
levels.


7. [EcoAssistant: Using LLM Assistant More Affordably and Accurately](http://arxiv.org/abs/2310.03046v1), Jieyu Zhang, Ranjay Krishna, Ahmed H. Awadallah, Chi Wang, 03-10-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Today, users ask Large language models (LLMs) as assistants to answer queries
that require external knowledge; they ask about the weather in a specific city,
about stock prices, and even about where specific locations are within their
neighborhood. These queries require the LLM to produce code that invokes
external APIs to answer the user's question, yet LLMs rarely produce correct
code on the first try, requiring iterative code refinement upon execution
results. In addition, using LLM assistants to support high query volumes can be
expensive. In this work, we contribute a framework, EcoAssistant, that enables
LLMs to answer code-driven queries more affordably and accurately. EcoAssistant
contains three components. First, it allows the LLM assistants to converse with
an automatic code executor to iteratively refine code or to produce answers
based on the execution results. Second, we use a hierarchy of LLM assistants,
which attempts to answer the query with weaker, cheaper LLMs before backing off
to stronger, expensive ones. Third, we retrieve solutions from past successful
queries as in-context demonstrations to help subsequent queries. Empirically,
we show that EcoAssistant offers distinct advantages for affordability and
accuracy, surpassing GPT-4 by 10 points of success rate with less than 50% of
GPT-4's cost.


7. [Large Language Models Cannot Self-Correct Reasoning Yet](http://arxiv.org/abs/2310.01798v1), Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou, 03-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) have emerged as a groundbreaking technology with
their unparalleled text generation capabilities across various applications.
Nevertheless, concerns persist regarding the accuracy and appropriateness of
their generated content. A contemporary methodology, self-correction, has been
proposed as a remedy to these issues. Building upon this premise, this paper
critically examines the role and efficacy of self-correction within LLMs,
shedding light on its true potential and limitations. Central to our
investigation is the notion of intrinsic self-correction, whereby an LLM
attempts to correct its initial responses based solely on its inherent
capabilities, without the crutch of external feedback. In the context of
reasoning, our research indicates that LLMs struggle to self-correct their
responses without external feedback, and at times, their performance might even
degrade post self-correction. Drawing from these insights, we offer suggestions
for future research and practical applications in this field.


7. [Large Language Models as Analogical Reasoners](http://arxiv.org/abs/2310.01714v2), Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong Pasupat, Jure Leskovec, Percy Liang, Ed H. Chi, Denny Zhou, 03-10-2023
     ### Categories
     Machine Learning
    ### Abstract
    Chain-of-thought (CoT) prompting for language models demonstrates impressive
performance across reasoning tasks, but typically needs labeled exemplars of
the reasoning process. In this work, we introduce a new prompting approach,
Analogical Prompting, designed to automatically guide the reasoning process of
large language models. Inspired by analogical reasoning, a cognitive process in
which humans draw from relevant past experiences to tackle new problems, our
approach prompts language models to self-generate relevant exemplars or
knowledge in the context, before proceeding to solve the given problem. This
method presents several advantages: it obviates the need for labeling or
retrieving exemplars, offering generality and convenience; it can also tailor
the generated exemplars and knowledge to each problem, offering adaptability.
Experimental results show that our approach outperforms 0-shot CoT and manual
few-shot CoT in a variety of reasoning tasks, including math problem solving in
GSM8K and MATH, code generation in Codeforces, and other reasoning tasks in
BIG-Bench.


7. [How FaR Are Large Language Models From Agents with Theory-of-Mind?](http://arxiv.org/abs/2310.03051v1), Pei Zhou, Aman Madaan, Srividya Pranavi Potharaju, Aditya Gupta, Kevin R. McKee, Ari Holtzman, Jay Pujara, Xiang Ren, Swaroop Mishra, Aida Nematzadeh, Shyam Upadhyay, Manaal Faruqui, 04-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    "Thinking is for Doing." Humans can infer other people's mental states from
observations--an ability called Theory-of-Mind (ToM)--and subsequently act
pragmatically on those inferences. Existing question answering benchmarks such
as ToMi ask models questions to make inferences about beliefs of characters in
a story, but do not test whether models can then use these inferences to guide
their actions. We propose a new evaluation paradigm for large language models
(LLMs): Thinking for Doing (T4D), which requires models to connect inferences
about others' mental states to actions in social scenarios. Experiments on T4D
demonstrate that LLMs such as GPT-4 and PaLM 2 seemingly excel at tracking
characters' beliefs in stories, but they struggle to translate this capability
into strategic action. Our analysis reveals the core challenge for LLMs lies in
identifying the implicit inferences about mental states without being
explicitly asked about as in ToMi, that lead to choosing the correct action in
T4D. To bridge this gap, we introduce a zero-shot prompting framework, Foresee
and Reflect (FaR), which provides a reasoning structure that encourages LLMs to
anticipate future challenges and reason about potential actions. FaR boosts
GPT-4's performance from 50% to 71% on T4D, outperforming other prompting
methods such as Chain-of-Thought and Self-Ask. Moreover, FaR generalizes to
diverse out-of-distribution story structures and scenarios that also require
ToM inferences to choose an action, consistently outperforming other methods
including few-shot in-context learning.


7. [Agent Instructs Large Language Models to be General Zero-Shot Reasoners](http://arxiv.org/abs/2310.03710v1), Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, Chenguang Wang, 05-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    We introduce a method to improve the zero-shot reasoning abilities of large
language models on general language understanding tasks. Specifically, we build
an autonomous agent to instruct the reasoning process of large language models.
We show this approach further unleashes the zero-shot reasoning abilities of
large language models to more tasks. We study the performance of our method on
a wide set of datasets spanning generation, classification, and reasoning. We
show that our method generalizes to most tasks and obtains state-of-the-art
zero-shot performance on 20 of the 29 datasets that we evaluate. For instance,
our method boosts the performance of state-of-the-art large language models by
a large margin, including Vicuna-13b (13.3%), Llama-2-70b-chat (23.2%), and
GPT-3.5 Turbo (17.0%). Compared to zero-shot chain of thought, our improvement
in reasoning is striking, with an average increase of 10.5%. With our method,
Llama-2-70b-chat outperforms zero-shot GPT-3.5 Turbo by 10.2%.


7. [FreshLLMs: Refreshing Large Language Models with Search Engine
  Augmentation](http://arxiv.org/abs/2310.03214v2), Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong, 05-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    Most large language models (LLMs) are trained once and never updated; thus,
they lack the ability to dynamically adapt to our ever-changing world. In this
work, we perform a detailed study of the factuality of LLM-generated text in
the context of answering questions that test current world knowledge.
Specifically, we introduce FreshQA, a novel dynamic QA benchmark encompassing a
diverse range of question and answer types, including questions that require
fast-changing world knowledge as well as questions with false premises that
need to be debunked. We benchmark a diverse array of both closed and
open-source LLMs under a two-mode evaluation procedure that allows us to
measure both correctness and hallucination. Through human evaluations involving
more than 50K judgments, we shed light on limitations of these models and
demonstrate significant room for improvement: for instance, all models
(regardless of model size) struggle on questions that involve fast-changing
knowledge and false premises. Motivated by these results, we present
FreshPrompt, a simple few-shot prompting method that substantially boosts the
performance of an LLM on FreshQA by incorporating relevant and up-to-date
information retrieved from a search engine into the prompt. Our experiments
show that FreshPrompt outperforms both competing search engine-augmented
prompting methods such as Self-Ask (Press et al., 2022) as well as commercial
systems such as Perplexity.AI. Further analysis of FreshPrompt reveals that
both the number of retrieved evidences and their order play a key role in
influencing the correctness of LLM-generated answers. Additionally, instructing
the LLM to generate concise and direct answers helps reduce hallucination
compared to encouraging more verbose answers. To facilitate future work, we
release FreshQA at github.com/freshllms/freshqa and commit to updating it at
regular intervals.


7. [Large Language Models for Software Engineering: Survey and Open Problems](http://arxiv.org/abs/2310.03533v4), Angela Fan, Beliz Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, Jie M. Zhang, 05-10-2023
    ### Abstract
    This paper provides a survey of the emerging area of Large Language Models
(LLMs) for Software Engineering (SE). It also sets out open research challenges
for the application of LLMs to technical problems faced by software engineers.
LLMs' emergent properties bring novelty and creativity with applications right
across the spectrum of Software Engineering activities including coding,
design, requirements, repair, refactoring, performance improvement,
documentation and analytics. However, these very same emergent properties also
pose significant technical challenges; we need techniques that can reliably
weed out incorrect solutions, such as hallucinations. Our survey reveals the
pivotal role that hybrid techniques (traditional SE plus LLMs) have to play in
the development and deployment of reliable, efficient and effective LLM-based
SE.


7. [Compressing Context to Enhance Inference Efficiency of Large Language
  Models](http://arxiv.org/abs/2310.06201v1), Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin, 09-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) achieved remarkable performance across various
tasks. However, they face challenges in managing long documents and extended
conversations, due to significantly increased computational requirements, both
in memory and inference time, and potential context truncation when the input
exceeds the LLM's fixed context length. This paper proposes a method called
Selective Context that enhances the inference efficiency of LLMs by identifying
and pruning redundancy in the input context to make the input more compact. We
test our approach using common data sources requiring long context processing:
arXiv papers, news articles, and long conversations, on tasks of summarisation,
question answering, and response generation. Experimental results show that
Selective Context significantly reduces memory cost and decreases generation
latency while maintaining comparable performance compared to that achieved when
full context is used. Specifically, we achieve a 50\% reduction in context
cost, resulting in a 36\% reduction in inference memory usage and a 32\%
reduction in inference time, while observing only a minor drop of .023 in
BERTscore and .038 in faithfulness on four downstream applications, indicating
that our method strikes a good balance between efficiency and performance.


7. [Take a Step Back: Evoking Reasoning via Abstraction in Large Language
  Models](http://arxiv.org/abs/2310.06117v1), Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, Denny Zhou, 09-10-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    We present Step-Back Prompting, a simple prompting technique that enables
LLMs to do abstractions to derive high-level concepts and first principles from
instances containing specific details. Using the concepts and principles to
guide the reasoning steps, LLMs significantly improve their abilities in
following a correct reasoning path towards the solution. We conduct experiments
of Step-Back Prompting with PaLM-2L models and observe substantial performance
gains on a wide range of challenging reasoning-intensive tasks including STEM,
Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting
improves PaLM-2L performance on MMLU Physics and Chemistry by 7% and 11%,
TimeQA by 27%, and MuSiQue by 7%.


7. [Beyond Memorization: Violating Privacy Via Inference with Large Language
  Models](http://arxiv.org/abs/2310.07298v1), Robin Staab, Mark Vero, Mislav Balunović, Martin Vechev, 11-10-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    Current privacy research on large language models (LLMs) primarily focuses on
the issue of extracting memorized training data. At the same time, models'
inference capabilities have increased drastically. This raises the key question
of whether current LLMs could violate individuals' privacy by inferring
personal attributes from text given at inference time. In this work, we present
the first comprehensive study on the capabilities of pretrained LLMs to infer
personal attributes from text. We construct a dataset consisting of real Reddit
profiles, and show that current LLMs can infer a wide range of personal
attributes (e.g., location, income, sex), achieving up to $85\%$ top-1 and
$95.8\%$ top-3 accuracy at a fraction of the cost ($100\times$) and time
($240\times$) required by humans. As people increasingly interact with
LLM-powered chatbots across all aspects of life, we also explore the emerging
threat of privacy-invasive chatbots trying to extract personal information
through seemingly benign questions. Finally, we show that common mitigations,
i.e., text anonymization and model alignment, are currently ineffective at
protecting user privacy against LLM inference. Our findings highlight that
current LLMs can infer personal data at a previously unattainable scale. In the
absence of working defenses, we advocate for a broader discussion around LLM
privacy implications beyond memorization, striving for a wider privacy
protection.


7. [Exploring the Landscape of Large Language Models In Medical Question
  Answering: Observations and Open Questions](http://arxiv.org/abs/2310.07225v1), Karolina Korgul, Andrew M. Bean, Felix Krones, Robert McCraith, Adam Mahdi, 11-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large Language Models (LLMs) have shown promise in medical question answering
by achieving passing scores in standardised exams and have been suggested as
tools for supporting healthcare workers. Deploying LLMs into such a high-risk
context requires a clear understanding of the limitations of these models. With
the rapid development and release of new LLMs, it is especially valuable to
identify patterns which exist across models and may, therefore, continue to
appear in newer versions. In this paper, we evaluate a wide range of popular
LLMs on their knowledge of medical questions in order to better understand
their properties as a group. From this comparison, we provide preliminary
observations and raise open questions for further research.


7. [Large Language Models Are Zero-Shot Time Series Forecasters](http://arxiv.org/abs/2310.07820v1), Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Gordon Wilson, 11-10-2023
     ### Categories
     Machine Learning
    ### Abstract
    By encoding time series as a string of numerical digits, we can frame time
series forecasting as next-token prediction in text. Developing this approach,
we find that large language models (LLMs) such as GPT-3 and LLaMA-2 can
surprisingly zero-shot extrapolate time series at a level comparable to or
exceeding the performance of purpose-built time series models trained on the
downstream tasks. To facilitate this performance, we propose procedures for
effectively tokenizing time series data and converting discrete distributions
over tokens into highly flexible densities over continuous values. We argue the
success of LLMs for time series stems from their ability to naturally represent
multimodal distributions, in conjunction with biases for simplicity, and
repetition, which align with the salient features in many time series, such as
repeated seasonal trends. We also show how LLMs can naturally handle missing
data without imputation through non-numerical text, accommodate textual side
information, and answer questions to help explain predictions. While we find
that increasing model size generally improves performance on time series, we
show GPT-4 can perform worse than GPT-3 because of how it tokenizes numbers,
and poor uncertainty calibration, which is likely the result of alignment
interventions such as RLHF.


7. [Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4
  on mock CFA Exams](http://arxiv.org/abs/2310.08678v1), Ethan Callanan, Amarachi Mbakwe, Antony Papadimitriou, Yulong Pei, Mathieu Sibue, Xiaodan Zhu, Zhiqiang Ma, Xiaomo Liu, Sameena Shah, 12-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) have demonstrated remarkable performance on a
wide range of Natural Language Processing (NLP) tasks, often matching or even
beating state-of-the-art task-specific models. This study aims at assessing the
financial reasoning capabilities of LLMs. We leverage mock exam questions of
the Chartered Financial Analyst (CFA) Program to conduct a comprehensive
evaluation of ChatGPT and GPT-4 in financial analysis, considering Zero-Shot
(ZS), Chain-of-Thought (CoT), and Few-Shot (FS) scenarios. We present an
in-depth analysis of the models' performance and limitations, and estimate
whether they would have a chance at passing the CFA exams. Finally, we outline
insights into potential strategies and improvements to enhance the
applicability of LLMs in finance. In this perspective, we hope this work paves
the way for future studies to continue enhancing LLMs for financial reasoning
through rigorous evaluation.


7. [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](http://arxiv.org/abs/2310.08659v4), Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao, 12-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Quantization is an indispensable technique for serving Large Language Models
(LLMs) and has recently found its way into LoRA fine-tuning. In this work we
focus on the scenario where quantization and LoRA fine-tuning are applied
together on a pre-trained model. In such cases it is common to observe a
consistent gap in the performance on downstream tasks between full fine-tuning
and quantization plus LoRA fine-tuning approach. In response, we propose LoftQ
(LoRA-Fine-Tuning-aware Quantization), a novel quantization framework that
simultaneously quantizes an LLM and finds a proper low-rank initialization for
LoRA fine-tuning. Such an initialization alleviates the discrepancy between the
quantized and full-precision model and significantly improves generalization in
downstream tasks. We evaluate our method on natural language understanding,
question answering, summarization, and natural language generation tasks.
Experiments show that our method is highly effective and outperforms existing
quantization methods, especially in the challenging 2-bit and 2/4-bit mixed
precision regimes. The code is available on https://github.com/yxli2123/LoftQ.


7. [MiniGPT-v2: large language model as a unified interface for
  vision-language multi-task learning](http://arxiv.org/abs/2310.09478v3), Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny, 14-10-2023
    ### Abstract
    Large language models have shown their remarkable capabilities as a general
interface for various language-related applications. Motivated by this, we
target to build a unified interface for completing many vision-language tasks
including image description, visual question answering, and visual grounding,
among others. The challenge is to use a single model for performing diverse
vision-language tasks effectively with simple multi-modal instructions. Towards
this objective, we introduce MiniGPT-v2, a model that can be treated as a
unified interface for better handling various vision-language tasks. We propose
using unique identifiers for different tasks when training the model. These
identifiers enable our model to better distinguish each task instruction
effortlessly and also improve the model learning efficiency for each task.
After the three-stage training, the experimental results show that MiniGPT-v2
achieves strong performance on many visual question-answering and visual
grounding benchmarks compared to other vision-language generalist models. Our
model and codes are available at https://minigpt-v2.github.io/


7. [Self-RAG: Learning to Retrieve, Generate, and Critique through
  Self-Reflection](http://arxiv.org/abs/2310.11511v1), Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi, 17-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Despite their remarkable capabilities, large language models (LLMs) often
produce responses containing factual inaccuracies due to their sole reliance on
the parametric knowledge they encapsulate. Retrieval-Augmented Generation
(RAG), an ad hoc approach that augments LMs with retrieval of relevant
knowledge, decreases such issues. However, indiscriminately retrieving and
incorporating a fixed number of retrieved passages, regardless of whether
retrieval is necessary, or passages are relevant, diminishes LM versatility or
can lead to unhelpful response generation. We introduce a new framework called
Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's
quality and factuality through retrieval and self-reflection. Our framework
trains a single arbitrary LM that adaptively retrieves passages on-demand, and
generates and reflects on retrieved passages and its own generations using
special tokens, called reflection tokens. Generating reflection tokens makes
the LM controllable during the inference phase, enabling it to tailor its
behavior to diverse task requirements. Experiments show that Self-RAG (7B and
13B parameters) significantly outperforms state-of-the-art LLMs and
retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG
outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA,
reasoning and fact verification tasks, and it shows significant gains in
improving factuality and citation accuracy for long-form generations relative
to these models.


7. [Contrastive Preference Learning: Learning from Human Feedback without RL](http://arxiv.org/abs/2310.13639v2), Joey Hejna, Rafael Rafailov, Harshit Sikchi, Chelsea Finn, Scott Niekum, W. Bradley Knox, Dorsa Sadigh, 20-10-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    Reinforcement Learning from Human Feedback (RLHF) has emerged as a popular
paradigm for aligning models with human intent. Typically RLHF algorithms
operate in two phases: first, use human preferences to learn a reward function
and second, align the model by optimizing the learned reward via reinforcement
learning (RL). This paradigm assumes that human preferences are distributed
according to reward, but recent work suggests that they instead follow the
regret under the user's optimal policy. Thus, learning a reward function from
feedback is not only based on a flawed assumption of human preference, but also
leads to unwieldy optimization challenges that stem from policy gradients or
bootstrapping in the RL phase. Because of these optimization challenges,
contemporary RLHF methods restrict themselves to contextual bandit settings
(e.g., as in large language models) or limit observation dimensionality (e.g.,
state-based robotics). We overcome these limitations by introducing a new
family of algorithms for optimizing behavior from human feedback using the
regret-based model of human preferences. Using the principle of maximum
entropy, we derive Contrastive Preference Learning (CPL), an algorithm for
learning optimal policies from preferences without learning reward functions,
circumventing the need for RL. CPL is fully off-policy, uses only a simple
contrastive objective, and can be applied to arbitrary MDPs. This enables CPL
to elegantly scale to high-dimensional and sequential RLHF problems while being
simpler than prior methods.


7. [The Perils & Promises of Fact-checking with Large Language Models](http://arxiv.org/abs/2310.13549v1), Dorian Quelle, Alexandre Bovet, 20-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    Autonomous fact-checking, using machine learning to verify claims, has grown
vital as misinformation spreads beyond human fact-checking capacity. Large
Language Models (LLMs) like GPT-4 are increasingly trusted to verify
information and write academic papers, lawsuits, and news articles, emphasizing
their role in discerning truth from falsehood and the importance of being able
to verify their outputs. Here, we evaluate the use of LLM agents in
fact-checking by having them phrase queries, retrieve contextual data, and make
decisions. Importantly, in our framework, agents explain their reasoning and
cite the relevant sources from the retrieved context. Our results show the
enhanced prowess of LLMs when equipped with contextual information. GPT-4
outperforms GPT-3, but accuracy varies based on query language and claim
veracity. While LLMs show promise in fact-checking, caution is essential due to
inconsistent accuracy. Our investigation calls for further research, fostering
a deeper comprehension of when agents succeed and when they fail.


7. [ALCUNA: Large Language Models Meet New Knowledge](http://arxiv.org/abs/2310.14820v1), Xunjian Yin, Baizhou Huang, Xiaojun Wan, 23-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    With the rapid development of NLP, large-scale language models (LLMs) excel
in various tasks across multiple domains now. However, existing benchmarks may
not adequately measure these models' capabilities, especially when faced with
new knowledge. In this paper, we address the lack of benchmarks to evaluate
LLMs' ability to handle new knowledge, an important and challenging aspect in
the rapidly evolving world. We propose an approach called KnowGen that
generates new knowledge by altering existing entity attributes and
relationships, resulting in artificial entities that are distinct from
real-world entities. With KnowGen, we introduce a benchmark named ALCUNA to
assess LLMs' abilities in knowledge understanding, differentiation, and
association. We benchmark several LLMs, reveals that their performance in face
of new knowledge is not satisfactory, particularly in reasoning between new and
internal knowledge. We also explore the impact of entity similarity on the
model's understanding of entity knowledge and the influence of contextual
entities. We appeal to the need for caution when using LLMs in new scenarios or
with new knowledge, and hope that our benchmarks can help drive the development
of LLMs in face of new knowledge.


7. [Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model
  System for Answering Medical Questions using Scientific Literature](http://arxiv.org/abs/2310.16146v1), Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, Nigam Shah, 24-10-2023
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    The quickly-expanding nature of published medical literature makes it
challenging for clinicians and researchers to keep up with and summarize
recent, relevant findings in a timely manner. While several closed-source
summarization tools based on large language models (LLMs) now exist, rigorous
and systematic evaluations of their outputs are lacking. Furthermore, there is
a paucity of high-quality datasets and appropriate benchmark tasks with which
to evaluate these tools. We address these issues with four contributions: we
release Clinfo.ai, an open-source WebApp that answers clinical questions based
on dynamically retrieved scientific literature; we specify an information
retrieval and abstractive summarization task to evaluate the performance of
such retrieval-augmented LLM systems; we release a dataset of 200 questions and
corresponding answers derived from published systematic reviews, which we name
PubMed Retrieval and Synthesis (PubMedRS-200); and report benchmark results for
Clinfo.ai and other publicly available OpenQA systems on PubMedRS-200.


7. [NoteChat: A Dataset of Synthetic Doctor-Patient Conversations
  Conditioned on Clinical Notes](http://arxiv.org/abs/2310.15959v2), Junda Wang, Zonghai Yao, Zhichao Yang, Huixue Zhou, Rumeng Li, Xun Wang, Yucheng Xu, Hong Yu, 24-10-2023
     ### Categories
     Computation and Language
    ### Abstract
    We introduce NoteChat, a novel cooperative multi-agent framework leveraging
Large Language Models (LLMs) to generate patient-physician dialogues. NoteChat
embodies the principle that an ensemble of role-specific LLMs, through
structured role-play and strategic prompting, can perform their assigned roles
more effectively. The synergy among these role-playing LLMs results in a
cohesive and efficient dialogue generation. Evaluation on MTS-dialogue, a
benchmark dataset for patient-physician dialogues-note pairs, shows that models
trained with the augmented synthetic patient-physician dialogues by NoteChat
outperforms other state-of-the-art models for generating clinical notes. Our
comprehensive automatic and human evaluation demonstrates that NoteChat
substantially surpasses state-of-the-art models like ChatGPT and GPT-4 up to
22.78% by domain experts in generating superior synthetic patient-physician
dialogues based on clinical notes. NoteChat has the potential to engage
patients directly and help clinical documentation, a leading cause of physician
burnout.


7. [Zephyr: Direct Distillation of LM Alignment](http://arxiv.org/abs/2310.16944v1), Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, Thomas Wolf, 25-10-2023
     ### Categories
     Machine Learning, Computation and Language
    ### Abstract
    We aim to produce a smaller language model that is aligned to user intent.
Previous research has shown that applying distilled supervised fine-tuning
(dSFT) on larger models significantly improves task accuracy; however, these
models are unaligned, i.e. they do not respond well to natural prompts. To
distill this property, we experiment with the use of preference data from AI
Feedback (AIF). Starting from a dataset of outputs ranked by a teacher model,
we apply distilled direct preference optimization (dDPO) to learn a chat model
with significantly improved intent alignment. The approach requires only a few
hours of training without any additional sampling during fine-tuning. The final
result, Zephyr-7B, sets the state-of-the-art on chat benchmarks for 7B
parameter models, and requires no human annotation. In particular, results on
MT-Bench show that Zephyr-7B surpasses Llama2-Chat-70B, the best open-access
RLHF-based model. Code, models, data, and tutorials for the system are
available at https://github.com/huggingface/alignment-handbook.


7. [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](http://arxiv.org/abs/2310.17631v1), Lianghui Zhu, Xinggang Wang, Xinlong Wang, 26-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Evaluating Large Language Models (LLMs) in open-ended scenarios is
challenging because existing benchmarks and metrics can not measure them
comprehensively. To address this problem, we propose to fine-tune LLMs as
scalable judges (JudgeLM) to evaluate LLMs efficiently and effectively in
open-ended benchmarks. We first propose a comprehensive, large-scale,
high-quality dataset containing task seeds, LLMs-generated answers, and
GPT-4-generated judgments for fine-tuning high-performance judges, as well as a
new benchmark for evaluating the judges. We train JudgeLM at different scales
from 7B, 13B, to 33B parameters, and conduct a systematic analysis of its
capabilities and behaviors. We then analyze the key biases in fine-tuning LLM
as a judge and consider them as position bias, knowledge bias, and format bias.
To address these issues, JudgeLM introduces a bag of techniques including swap
augmentation, reference support, and reference drop, which clearly enhance the
judge's performance. JudgeLM obtains the state-of-the-art judge performance on
both the existing PandaLM benchmark and our proposed new benchmark. Our JudgeLM
is efficient and the JudgeLM-7B only needs 3 minutes to judge 5K samples with 8
A100 GPUs. JudgeLM obtains high agreement with the teacher judge, achieving an
agreement exceeding 90% that even surpasses human-to-human agreement. JudgeLM
also demonstrates extended capabilities in being judges of the single answer,
multimodal models, multiple answers, and multi-turn chat.


7. [Large Language Models as Evolutionary Optimizers](http://arxiv.org/abs/2310.19046v2), Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang, Yew-Soon Ong, 29-10-2023
    ### Abstract
    Evolutionary algorithms (EAs) have achieved remarkable success in tackling
complex combinatorial optimization problems. However, EAs often demand
carefully-designed operators with the aid of domain expertise to achieve
satisfactory performance. In this work, we present the first study on large
language models (LLMs) as evolutionary combinatorial optimizers. The main
advantage is that it requires minimal domain knowledge and human efforts, as
well as no additional training of the model. This approach is referred to as
LLM-driven EA (LMEA). Specifically, in each generation of the evolutionary
search, LMEA instructs the LLM to select parent solutions from current
population, and perform crossover and mutation to generate offspring solutions.
Then, LMEA evaluates these new solutions and include them into the population
for the next generation. LMEA is equipped with a self-adaptation mechanism that
controls the temperature of the LLM. This enables it to balance between
exploration and exploitation and prevents the search from getting stuck in
local optima. We investigate the power of LMEA on the classical traveling
salesman problems (TSPs) widely used in combinatorial optimization research.
Notably, the results show that LMEA performs competitively to traditional
heuristics in finding high-quality solutions on TSP instances with up to 20
nodes. Additionally, we also study the effectiveness of LLM-driven
crossover/mutation and the self-adaptation mechanism in evolutionary search. In
summary, our results reveal the great potentials of LLMs as evolutionary
optimizers for solving combinatorial problems. We hope our research shall
inspire future explorations on LLM-driven EAs for complex optimization
challenges.


7. [TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language
  Modeling Likewise](http://arxiv.org/abs/2310.19019v2), Nan He, Hanyu Lai, Chenyang Zhao, Zirui Cheng, Junting Pan, Ruoyu Qin, Ruofan Lu, Rui Lu, Yunchen Zhang, Gangming Zhao, Zhaohui Hou, Zhiyuan Huang, Shaoqing Lu, Ding Liang, Mingjie Zhan, 29-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) exhibit impressive reasoning and data
augmentation capabilities in various NLP tasks. However, what about small
models? In this work, we propose TeacherLM-7.1B, capable of annotating relevant
fundamentals, chain of thought, and common mistakes for most NLP samples, which
makes annotation more than just an answer, thus allowing other models to learn
"why" instead of just "what". The TeacherLM-7.1B model achieved a zero-shot
score of 52.3 on MMLU, surpassing most models with over 100B parameters. Even
more remarkable is its data augmentation ability. Based on TeacherLM-7.1B, we
augmented 58 NLP datasets and taught various student models with different
parameters from OPT and BLOOM series in a multi-task setting. The experimental
results indicate that the data augmentation provided by TeacherLM has brought
significant benefits. We will release the TeacherLM series of models and
augmented datasets as open-source.


7. [EHRTutor: Enhancing Patient Understanding of Discharge Instructions](http://arxiv.org/abs/2310.19212v1), Zihao Zhang, Zonghai Yao, Huixue Zhou, Feiyun ouyang, Hong Yu, 30-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models have shown success as a tutor in education in various
fields. Educating patients about their clinical visits plays a pivotal role in
patients' adherence to their treatment plans post-discharge. This paper
presents EHRTutor, an innovative multi-component framework leveraging the Large
Language Model (LLM) for patient education through conversational
question-answering. EHRTutor first formulates questions pertaining to the
electronic health record discharge instructions. It then educates the patient
through conversation by administering each question as a test. Finally, it
generates a summary at the end of the conversation. Evaluation results using
LLMs and domain experts have shown a clear preference for EHRTutor over the
baseline. Moreover, EHRTutor also offers a framework for generating synthetic
patient education dialogues that can be used for future in-house system
training.


7. [Evaluating Large Language Models: A Comprehensive Survey](http://arxiv.org/abs/2310.19736v3), Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi,  Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, Deyi Xiong, 30-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) have demonstrated remarkable capabilities across
a broad spectrum of tasks. They have attracted significant attention and been
deployed in numerous downstream applications. Nevertheless, akin to a
double-edged sword, LLMs also present potential risks. They could suffer from
private data leaks or yield inappropriate, harmful, or misleading content.
Additionally, the rapid progress of LLMs raises concerns about the potential
emergence of superintelligent systems without adequate safeguards. To
effectively capitalize on LLM capacities as well as ensure their safe and
beneficial development, it is critical to conduct a rigorous and comprehensive
evaluation of LLMs.
  This survey endeavors to offer a panoramic perspective on the evaluation of
LLMs. We categorize the evaluation of LLMs into three major groups: knowledge
and capability evaluation, alignment evaluation and safety evaluation. In
addition to the comprehensive review on the evaluation methodologies and
benchmarks on these three aspects, we collate a compendium of evaluations
pertaining to LLMs' performance in specialized domains, and discuss the
construction of comprehensive evaluation platforms that cover LLM evaluations
on capabilities, alignment, safety, and applicability.
  We hope that this comprehensive overview will stimulate further research
interests in the evaluation of LLMs, with the ultimate goal of making
evaluation serve as a cornerstone in guiding the responsible development of
LLMs. We envision that this will channel their evolution into a direction that
maximizes societal benefit while minimizing potential risks. A curated list of
related papers has been publicly available at
https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers.


7. [Learning From Mistakes Makes LLM Better Reasoner](http://arxiv.org/abs/2310.20689v2), Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen, 31-10-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) recently exhibited remarkable reasoning
capabilities on solving math problems. To further improve this capability, this
work proposes Learning from Mistakes (LeMa), akin to human learning processes.
Consider a human student who failed to solve a math problem, he will learn from
what mistake he has made and how to correct it. Mimicking this error-driven
learning process, LeMa fine-tunes LLMs on mistake-correction data pairs
generated by GPT-4. Specifically, we first collect inaccurate reasoning paths
from various LLMs and then employ GPT-4 as a "corrector" to (1) identify the
mistake step, (2) explain the reason for the mistake, and (3) correct the
mistake and generate the final answer. Experimental results demonstrate the
effectiveness of LeMa: across five backbone LLMs and two mathematical reasoning
tasks, LeMa consistently improves the performance compared with fine-tuning on
CoT data alone. Impressively, LeMa can also benefit specialized LLMs such as
WizardMath and MetaMath, achieving 85.4% pass@1 accuracy on GSM8K and 27.1% on
MATH. This surpasses the SOTA performance achieved by non-execution open-source
models on these challenging tasks. Our code, data and models will be publicly
available at https://github.com/microsoft/LEMA.


7. [TopicGPT: A Prompt-based Topic Modeling Framework](http://arxiv.org/abs/2311.01449v1), Chau Minh Pham, Alexander Hoyle, Simeng Sun, Mohit Iyyer, 02-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Topic modeling is a well-established technique for exploring text corpora.
Conventional topic models (e.g., LDA) represent topics as bags of words that
often require "reading the tea leaves" to interpret; additionally, they offer
users minimal semantic control over topics. To tackle these issues, we
introduce TopicGPT, a prompt-based framework that uses large language models
(LLMs) to uncover latent topics within a provided text collection. TopicGPT
produces topics that align better with human categorizations compared to
competing methods: for example, it achieves a harmonic mean purity of 0.74
against human-annotated Wikipedia topics compared to 0.64 for the strongest
baseline. Its topics are also more interpretable, dispensing with ambiguous
bags of words in favor of topics with natural language labels and associated
free-form descriptions. Moreover, the framework is highly adaptable, allowing
users to specify constraints and modify topics without the need for model
retraining. TopicGPT can be further extended to hierarchical topical modeling,
enabling users to explore topics at various levels of granularity. By
streamlining access to high-quality and interpretable topics, TopicGPT
represents a compelling, human-centered approach to topic modeling.


7. [Large Language Models Illuminate a Progressive Pathway to Artificial
  Healthcare Assistant: A Review](http://arxiv.org/abs/2311.01918v1), Mingze Yuan, Peng Bao, Jiajia Yuan, Yunhao Shen, Zifan Chen, Yi Xie, Jie Zhao, Yang Chen, Li Zhang, Lin Shen, Bin Dong, 03-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    With the rapid development of artificial intelligence, large language models
(LLMs) have shown promising capabilities in mimicking human-level language
comprehension and reasoning. This has sparked significant interest in applying
LLMs to enhance various aspects of healthcare, ranging from medical education
to clinical decision support. However, medicine involves multifaceted data
modalities and nuanced reasoning skills, presenting challenges for integrating
LLMs. This paper provides a comprehensive review on the applications and
implications of LLMs in medicine. It begins by examining the fundamental
applications of general-purpose and specialized LLMs, demonstrating their
utilities in knowledge retrieval, research support, clinical workflow
automation, and diagnostic assistance. Recognizing the inherent multimodality
of medicine, the review then focuses on multimodal LLMs, investigating their
ability to process diverse data types like medical imaging and EHRs to augment
diagnostic accuracy. To address LLMs' limitations regarding personalization and
complex clinical reasoning, the paper explores the emerging development of
LLM-powered autonomous agents for healthcare. Furthermore, it summarizes the
evaluation methodologies for assessing LLMs' reliability and safety in medical
contexts. Overall, this review offers an extensive analysis on the
transformative potential of LLMs in modern medicine. It also highlights the
pivotal need for continuous optimizations and ethical oversight before these
models can be effectively integrated into clinical practice. Visit
https://github.com/mingze-yuan/Awesome-LLM-Healthcare for an accompanying
GitHub repository containing latest papers.


7. [Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs](http://arxiv.org/abs/2311.02262v1), Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao, 03-11-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    In human-written articles, we often leverage the subtleties of text style,
such as bold and italics, to guide the attention of readers. These textual
emphases are vital for the readers to grasp the conveyed information. When
interacting with large language models (LLMs), we have a similar need -
steering the model to pay closer attention to user-specified information, e.g.,
an instruction. Existing methods, however, are constrained to process plain
text and do not support such a mechanism. This motivates us to introduce PASTA
- Post-hoc Attention STeering Approach, a method that allows LLMs to read text
with user-specified emphasis marks. To this end, PASTA identifies a small
subset of attention heads and applies precise attention reweighting on them,
directing the model attention to user-specified parts. Like prompting, PASTA is
applied at inference time and does not require changing any model parameters.
Experiments demonstrate that PASTA can substantially enhance an LLM's ability
to follow user instructions or integrate new knowledge from user inputs,
leading to a significant performance improvement on a variety of tasks, e.g.,
an average accuracy improvement of 22% for LLAMA-7B. Our code is publicly
available at https://github.com/QingruZhang/PASTA .


7. [Can LLMs Follow Simple Rules?](http://arxiv.org/abs/2311.04235v1), Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, David Wagner, 06-11-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    As Large Language Models (LLMs) are deployed with increasing real-world
responsibilities, it is important to be able to specify and constrain the
behavior of these systems in a reliable manner. Model developers may wish to
set explicit rules for the model, such as "do not generate abusive content",
but these may be circumvented by jailbreaking techniques. Evaluating how well
LLMs follow developer-provided rules in the face of adversarial inputs
typically requires manual review, which slows down monitoring and methods
development. To address this issue, we propose Rule-following Language
Evaluation Scenarios (RuLES), a programmatic framework for measuring
rule-following ability in LLMs. RuLES consists of 15 simple text scenarios in
which the model is instructed to obey a set of rules in natural language while
interacting with the human user. Each scenario has a concise evaluation program
to determine whether the model has broken any rules in a conversation. Through
manual exploration of model behavior in our scenarios, we identify 6 categories
of attack strategies and collect two suites of test cases: one consisting of
unique conversations from manual testing and one that systematically implements
strategies from the 6 categories. Across various popular proprietary and open
models such as GPT-4 and Llama 2, we find that all models are susceptible to a
wide variety of adversarial hand-crafted user inputs, though GPT-4 is the
best-performing model. Additionally, we evaluate open models under
gradient-based attacks and find significant vulnerabilities. We propose RuLES
as a challenging new setting for research into exploring and defending against
both manual and automatic attacks on LLMs.


7. [Language Models are Super Mario: Absorbing Abilities from Homologous
  Models as a Free Lunch](http://arxiv.org/abs/2311.03099v1), Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li, 06-11-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    In this paper, we uncover that Language Models (LMs), either encoder- or
decoder-based, can obtain new capabilities by assimilating the parameters of
homologous models without retraining or GPUs. Typically, new abilities of LMs
can be imparted by Supervised Fine-Tuning (SFT), reflected in the disparity
between fine-tuned and pre-trained parameters (i.e., delta parameters). We
initially observe that by introducing a novel operation called DARE (Drop And
REscale), most delta parameters can be directly set to zeros without affecting
the capabilities of SFT LMs and larger models can tolerate a higher proportion
of discarded parameters. Based on this observation, we further sparsify delta
parameters of multiple SFT homologous models with DARE and subsequently merge
them into a single model by parameter averaging. We conduct experiments on
eight datasets from the GLUE benchmark with BERT and RoBERTa. We also merge
WizardLM, WizardMath, and Code Alpaca based on Llama 2. Experimental results
show that: (1) The delta parameter value ranges for SFT models are typically
small, often within 0.005, and DARE can eliminate 99% of them effortlessly.
However, once the models are continuously pre-trained, the value ranges can
grow to around 0.03, making DARE impractical. We have also tried to remove
fine-tuned instead of delta parameters and find that a 10% reduction can lead
to drastically decreased performance (even to 0). This highlights that SFT
merely stimulates the abilities via delta parameters rather than injecting new
abilities into LMs; (2) DARE can merge multiple task-specific LMs into one LM
with diverse abilities. For instance, the merger of WizardLM and WizardMath
improves the GSM8K zero-shot accuracy of WizardLM from 2.2 to 66.3, retaining
its instruction-following ability while surpassing WizardMath's original 64.2
performance. Codes are available at https://github.com/yule-BUAA/MergeLM.


7. [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](http://arxiv.org/abs/2311.03285v2), Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica, 06-11-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    The "pretrain-then-finetune" paradigm is commonly adopted in the deployment
of large language models. Low-Rank Adaptation (LoRA), a parameter-efficient
fine-tuning method, is often employed to adapt a base model to a multitude of
tasks, resulting in a substantial collection of LoRA adapters derived from one
base model. We observe that this paradigm presents significant opportunities
for batched inference during serving. To capitalize on these opportunities, we
present S-LoRA, a system designed for the scalable serving of many LoRA
adapters. S-LoRA stores all adapters in the main memory and fetches the
adapters used by the currently running queries to the GPU memory. To
efficiently use the GPU memory and reduce fragmentation, S-LoRA proposes
Unified Paging. Unified Paging uses a unified memory pool to manage dynamic
adapter weights with different ranks and KV cache tensors with varying sequence
lengths. Additionally, S-LoRA employs a novel tensor parallelism strategy and
highly optimized custom CUDA kernels for heterogeneous batching of LoRA
computation. Collectively, these features enable S-LoRA to serve thousands of
LoRA adapters on a single GPU or across multiple GPUs with a small overhead.
Compared to state-of-the-art libraries such as HuggingFace PEFT and vLLM (with
naive support of LoRA serving), S-LoRA can improve the throughput by up to 4
times and increase the number of served adapters by several orders of
magnitude. As a result, S-LoRA enables scalable serving of many task-specific
fine-tuned models and offers the potential for large-scale customized
fine-tuning services. The code is available at https://github.com/S-LoRA/S-LoRA


7. [Rephrase and Respond: Let Large Language Models Ask Better Questions for
  Themselves](http://arxiv.org/abs/2311.04205v1), Yihe Deng, Weitong Zhang, Zixiang Chen, Quanquan Gu, 07-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Misunderstandings arise not only in interpersonal communication but also
between humans and Large Language Models (LLMs). Such discrepancies can make
LLMs interpret seemingly unambiguous questions in unexpected ways, yielding
incorrect responses. While it is widely acknowledged that the quality of a
prompt, such as a question, significantly impacts the quality of the response
provided by LLMs, a systematic method for crafting questions that LLMs can
better comprehend is still underdeveloped. In this paper, we present a method
named `Rephrase and Respond' (RaR), which allows LLMs to rephrase and expand
questions posed by humans and provide responses in a single prompt. This
approach serves as a simple yet effective prompting method for improving
performance. We also introduce a two-step variant of RaR, where a rephrasing
LLM first rephrases the question and then passes the original and rephrased
questions together to a different responding LLM. This facilitates the
effective utilization of rephrased questions generated by one LLM with another.
Our experiments demonstrate that our methods significantly improve the
performance of different models across a wide range to tasks. We further
provide a comprehensive comparison between RaR and the popular Chain-of-Thought
(CoT) methods, both theoretically and empirically. We show that RaR is
complementary to CoT and can be combined with CoT to achieve even better
performance. Our work not only contributes to enhancing LLM performance
efficiently and effectively but also sheds light on a fair evaluation of LLM
capabilities. Data and codes are available at
https://github.com/uclaml/Rephrase-and-Respond.


7. [ADaPT: As-Needed Decomposition and Planning with Language Models](http://arxiv.org/abs/2311.05772v1), Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, Tushar Khot, 08-11-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Large Language Models (LLMs) are increasingly being used for interactive
decision-making tasks requiring planning and adapting to the environment.
Recent works employ LLMs-as-agents in broadly two ways: iteratively determining
the next action (iterative executors) or generating plans and executing
sub-tasks using LLMs (plan-and-execute). However, these methods struggle with
task complexity, as the inability to execute any sub-task may lead to task
failure. To address these shortcomings, we introduce As-Needed Decomposition
and Planning for complex Tasks (ADaPT), an approach that explicitly plans and
decomposes complex sub-tasks as-needed, i.e., when the LLM is unable to execute
them. ADaPT recursively decomposes sub-tasks to adapt to both task complexity
and LLM capability. Our results demonstrate that ADaPT substantially
outperforms established strong baselines, achieving success rates up to 28.3%
higher in ALFWorld, 27% in WebShop, and 33% in TextCraft -- a novel
compositional dataset that we introduce. Through extensive analysis, we
illustrate the importance of multilevel decomposition and establish that ADaPT
dynamically adjusts to the capabilities of the executor LLM as well as to task
complexity.


7. [A Survey of Large Language Models in Medicine: Principles, Applications,
  and Challenges](http://arxiv.org/abs/2311.05112v2), Hongjian Zhou, Fenglin Liu, Boyang Gu, Xinyu Zou, Jinfa Huang, Jinge Wu, Yiru Li, Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Xian Wu, Yefeng Zheng, Lei Clifton, Zheng Li, Jiebo Luo, David A. Clifton, 09-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs), such as ChatGPT, have received substantial
attention due to their impressive human language understanding and generation
capabilities. Therefore, the application of LLMs in medicine to assist
physicians and patient care emerges as a promising research direction in both
artificial intelligence and clinical medicine. To reflect this trend, this
survey provides a comprehensive overview of the principles, applications, and
challenges faced by LLMs in medicine. Specifically, we aim to address the
following questions: 1) How can medical LLMs be built? 2) What are the
downstream performances of medical LLMs? 3) How can medical LLMs be utilized in
real-world clinical practice? 4) What challenges arise from the use of medical
LLMs? and 5) How can we better construct and utilize medical LLMs? As a result,
this survey aims to provide insights into the opportunities and challenges of
LLMs in medicine and serve as a valuable resource for constructing practical
and effective medical LLMs. A regularly updated list of practical guides on
medical LLMs can be found at
https://github.com/AI-in-Health/MedLLMsPracticalGuide.


7. [A Survey on Hallucination in Large Language Models: Principles,
  Taxonomy, Challenges, and Open Questions](http://arxiv.org/abs/2311.05232v1), Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu, 09-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    The emergence of large language models (LLMs) has marked a significant
breakthrough in natural language processing (NLP), leading to remarkable
advancements in text understanding and generation. Nevertheless, alongside
these strides, LLMs exhibit a critical tendency to produce hallucinations,
resulting in content that is inconsistent with real-world facts or user inputs.
This phenomenon poses substantial challenges to their practical deployment and
raises concerns over the reliability of LLMs in real-world scenarios, which
attracts increasing attention to detect and mitigate these hallucinations. In
this survey, we aim to provide a thorough and in-depth overview of recent
advances in the field of LLM hallucinations. We begin with an innovative
taxonomy of LLM hallucinations, then delve into the factors contributing to
hallucinations. Subsequently, we present a comprehensive overview of
hallucination detection methods and benchmarks. Additionally, representative
approaches designed to mitigate hallucinations are introduced accordingly.
Finally, we analyze the challenges that highlight the current limitations and
formulate open questions, aiming to delineate pathways for future research on
hallucinations in LLMs.


7. [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](http://arxiv.org/abs/2311.05437v1), Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, Chunyuan Li, 09-11-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    LLaVA-Plus is a general-purpose multimodal assistant that expands the
capabilities of large multimodal models. It maintains a skill repository of
pre-trained vision and vision-language models and can activate relevant tools
based on users' inputs to fulfill real-world tasks. LLaVA-Plus is trained on
multimodal instruction-following data to acquire the ability to use tools,
covering visual understanding, generation, external knowledge retrieval, and
compositions. Empirical results show that LLaVA-Plus outperforms LLaVA in
existing capabilities and exhibits new ones. It is distinct in that the image
query is directly grounded and actively engaged throughout the entire human-AI
interaction sessions, significantly improving tool use performance and enabling
new scenarios.


7. [Technical Report: Large Language Models can Strategically Deceive their
  Users when Put Under Pressure](http://arxiv.org/abs/2311.07590v2), Jérémy Scheurer, Mikita Balesni, Marius Hobbhahn, 09-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    We demonstrate a situation in which Large Language Models, trained to be
helpful, harmless, and honest, can display misaligned behavior and
strategically deceive their users about this behavior without being instructed
to do so. Concretely, we deploy GPT-4 as an agent in a realistic, simulated
environment, where it assumes the role of an autonomous stock trading agent.
Within this environment, the model obtains an insider tip about a lucrative
stock trade and acts upon it despite knowing that insider trading is
disapproved of by company management. When reporting to its manager, the model
consistently hides the genuine reasons behind its trading decision. We perform
a brief investigation of how this behavior varies under changes to the setting,
such as removing model access to a reasoning scratchpad, attempting to prevent
the misaligned behavior by changing system instructions, changing the amount of
pressure the model is under, varying the perceived risk of getting caught, and
making other simple changes to the environment. To our knowledge, this is the
first demonstration of Large Language Models trained to be helpful, harmless,
and honest, strategically deceiving their users in a realistic situation
without direct instructions or training for deception.


7. [Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations](http://arxiv.org/abs/2311.05584v1), Joey Hong, Sergey Levine, Anca Dragan, 09-11-2023
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Large language models (LLMs) have emerged as powerful and general solutions
to many natural language tasks. However, many of the most important
applications of language generation are interactive, where an agent has to talk
to a person to reach a desired outcome. For example, a teacher might try to
understand their student's current comprehension level to tailor their
instruction accordingly, and a travel agent might ask questions of their
customer to understand their preferences in order to recommend activities they
might enjoy. LLMs trained with supervised fine-tuning or "single-step" RL, as
with standard RLHF, might struggle which tasks that require such goal-directed
behavior, since they are not trained to optimize for overall conversational
outcomes after multiple turns of interaction. In this work, we explore a new
method for adapting LLMs with RL for such goal-directed dialogue. Our key
insight is that, though LLMs might not effectively solve goal-directed dialogue
tasks out of the box, they can provide useful data for solving such tasks by
simulating suboptimal but human-like behaviors. Given a textual description of
a goal-directed dialogue task, we leverage LLMs to sample diverse synthetic
rollouts of hypothetical in-domain human-human interactions. Our algorithm then
utilizes this dataset with offline reinforcement learning to train an
interactive conversational agent that can optimize goal-directed objectives
over multiple turns. In effect, the LLM produces examples of possible
interactions, and RL then processes these examples to learn to perform more
optimal interactions. Empirically, we show that our proposed approach achieves
state-of-the-art performance in various goal-directed dialogue tasks that
include teaching and preference elicitation.


7. [MEGAVERSE: Benchmarking Large Language Models Across Languages,
  Modalities, Models and Tasks](http://arxiv.org/abs/2311.07463v1), Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, Sunayana Sitaram, 13-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Recently, there has been a rapid advancement in research on Large Language
Models (LLMs), resulting in significant progress in several Natural Language
Processing (NLP) tasks. Consequently, there has been a surge in LLM evaluation
research to comprehend the models' capabilities and limitations. However, much
of this research has been confined to the English language, leaving LLM
building and evaluation for non-English languages relatively unexplored. There
has been an introduction of several new LLMs, necessitating their evaluation on
non-English languages. This study aims to expand our MEGA benchmarking suite by
including six new datasets to form the MEGAVERSE benchmark. The benchmark
comprises 22 datasets covering 81 languages, including low-resource African
languages. We evaluate several state-of-the-art LLMs like GPT-3.5-Turbo, GPT4,
PaLM2, and Llama2 on the MEGAVERSE datasets. Additionally, we include two
multimodal datasets in the benchmark and assess the performance of the
LLaVa-v1.5 model. Our experiments suggest that GPT4 and PaLM2 outperform the
Llama models on various tasks, notably on low-resource languages, with GPT4
outperforming PaLM2 on more datasets than vice versa. However, issues such as
data contamination must be addressed to obtain an accurate assessment of LLM
performance on non-English languages.


7. [The Impact of Large Language Models on Scientific Discovery: a
  Preliminary Study using GPT-4](http://arxiv.org/abs/2311.07361v2), Microsoft Research AI4Science, Microsoft Azure Quantum, 13-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In recent years, groundbreaking advancements in natural language processing
have culminated in the emergence of powerful large language models (LLMs),
which have showcased remarkable capabilities across a vast array of domains,
including the understanding, generation, and translation of natural language,
and even tasks that extend beyond language processing. In this report, we delve
into the performance of LLMs within the context of scientific discovery,
focusing on GPT-4, the state-of-the-art language model. Our investigation spans
a diverse range of scientific areas encompassing drug discovery, biology,
computational chemistry (density functional theory (DFT) and molecular dynamics
(MD)), materials design, and partial differential equations (PDE). Evaluating
GPT-4 on scientific tasks is crucial for uncovering its potential across
various research domains, validating its domain-specific expertise,
accelerating scientific progress, optimizing resource allocation, guiding
future model development, and fostering interdisciplinary research. Our
exploration methodology primarily consists of expert-driven case assessments,
which offer qualitative insights into the model's comprehension of intricate
scientific concepts and relationships, and occasionally benchmark testing,
which quantitatively evaluates the model's capacity to solve well-defined
domain-specific problems. Our preliminary exploration indicates that GPT-4
exhibits promising potential for a variety of scientific applications,
demonstrating its aptitude for handling complex problem-solving and knowledge
integration tasks. Broadly speaking, we evaluate GPT-4's knowledge base,
scientific understanding, scientific numerical calculation abilities, and
various scientific prediction capabilities.


7. [Fast Chain-of-Thought: A Glance of Future from Parallel Decoding Leads
  to Answers Faster](http://arxiv.org/abs/2311.08263v1), Hongxuan Zhang, Zhining Liu, Jiaqi Zheng, Chenyi Zhuang, Jinjie Gu, Guihai Chen, 14-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    In this work, we propose FastCoT, a model-agnostic framework based on
parallel decoding without any further training of an auxiliary model or
modification to the LLM itself. FastCoT uses a size-varying context window
whose size changes with position to conduct parallel decoding and
auto-regressive decoding simultaneously, thus fully utilizing GPU computation
resources. In FastCoT, the parallel decoding part provides the LLM with a quick
glance of the future composed of approximate tokens, which could lead to faster
answers compared to regular autoregressive decoding used by causal
transformers. We also provide an implementation of parallel decoding within
LLM, which supports KV-cache generation and batch processing. Through extensive
experiments, we demonstrate that FastCoT saves inference time by nearly 20%
with only a negligible performance drop compared to the regular approach.
Additionally, we show that the context window size exhibits considerable
robustness for different tasks.


7. [The ART of LLM Refinement: Ask, Refine, and Trust](http://arxiv.org/abs/2311.07961v1), Kumar Shridhar, Koustuv Sinha, Andrew Cohen, Tianlu Wang, Ping Yu, Ram Pasunuru, Mrinmaya Sachan, Jason Weston, Asli Celikyilmaz, 14-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    In recent years, Large Language Models (LLMs) have demonstrated remarkable
generative abilities, but can they judge the quality of their own generations?
A popular concept, referred to as self-refinement, postulates that LLMs can
detect and correct the errors in their generations when asked to do so.
However, recent empirical evidence points in the opposite direction, suggesting
that LLMs often struggle to accurately identify errors when reasoning is
involved. To address this, we propose a reasoning with refinement objective
called ART: Ask, Refine, and Trust, which asks necessary questions to decide
when an LLM should refine its output, and either affirm or withhold trust in
its refinement by ranking the refinement and the initial prediction. On two
multistep reasoning tasks of mathematical word problems (GSM8K) and question
answering (StrategyQA), ART achieves a performance gain of +5 points over
self-refinement baselines, while using a much smaller model as the decision
maker. We also demonstrate the benefit of using smaller models to make
refinement decisions as a cost-effective alternative to fine-tuning a larger
model.


7. [Unifying the Perspectives of NLP and Software Engineering: A Survey on
  Language Models for Code](http://arxiv.org/abs/2311.07989v3), Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, Rui Wang, 14-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In this work we systematically review the recent advancements in code
processing with language models, covering 50+ models, 30+ evaluation tasks,
170+ datasets, and 700 related works. We break down code processing models into
general language models represented by the GPT family and specialized models
that are specifically pretrained on code, often with tailored objectives. We
discuss the relations and differences between these models, and highlight the
historical transition of code modeling from statistical models and RNNs to
pretrained Transformers and LLMs, which is exactly the same course that had
been taken by NLP. We also discuss code-specific features such as AST, CFG, and
unit tests, along with their application in training code language models, and
identify key challenges and potential future directions in this domain. We keep
the survey open and updated on GitHub at
https://github.com/codefuse-ai/Awesome-Code-LLM.


7. [Contrastive Chain-of-Thought Prompting](http://arxiv.org/abs/2311.09277v1), Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing, 15-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Despite the success of chain of thought in enhancing language model
reasoning, the underlying process remains less well understood. Although
logically sound reasoning appears inherently crucial for chain of thought,
prior studies surprisingly reveal minimal impact when using invalid
demonstrations instead. Furthermore, the conventional chain of thought does not
inform language models on what mistakes to avoid, which potentially leads to
more errors. Hence, inspired by how humans can learn from both positive and
negative examples, we propose contrastive chain of thought to enhance language
model reasoning. Compared to the conventional chain of thought, our approach
provides both valid and invalid reasoning demonstrations, to guide the model to
reason step-by-step while reducing reasoning mistakes. To improve
generalization, we introduce an automatic method to construct contrastive
demonstrations. Our experiments on reasoning benchmarks demonstrate that
contrastive chain of thought can serve as a general enhancement of
chain-of-thought prompting.


7. [Towards Verifiable Text Generation with Symbolic References](http://arxiv.org/abs/2311.09188v1), Lucas Torroba Hennigen, Shannon Shen, Aniruddha Nrusimha, Bernhard Gapp, David Sontag, Yoon Kim, 15-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large language models (LLMs) have demonstrated an impressive ability to
synthesize plausible and fluent text. However they remain vulnerable to
hallucinations, and thus their outputs generally require manual human
verification for high-stakes applications, which can be time-consuming and
difficult. This paper proposes symbolically grounded generation (SymGen) as a
simple approach for enabling easier validation of an LLM's output. SymGen
prompts an LLM to interleave its regular output text with explicit symbolic
references to fields present in some conditioning data (e.g., a table in JSON
format). The references can be used to display the provenance of different
spans of text in the generation, reducing the effort required for manual
verification. Across data-to-text and question answering experiments, we find
that LLMs are able to directly output text that makes use of symbolic
references while maintaining fluency and accuracy.


7. [MedAgents: Large Language Models as Collaborators for Zero-shot Medical
  Reasoning](http://arxiv.org/abs/2311.10537v1), Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, Mark Gerstein, 16-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs), despite their remarkable progress across
various general domains, encounter significant barriers in medicine and
healthcare. This field faces unique challenges such as domain-specific
terminologies and the reasoning over specialized knowledge. To address these
obstinate issues, we propose a novel Multi-disciplinary Collaboration (MC)
framework for the medical domain that leverages role-playing LLM-based agents
who participate in a collaborative multi-round discussion, thereby enhancing
LLM proficiency and reasoning capabilities. This training-free and
interpretable framework encompasses five critical steps: gathering domain
experts, proposing individual analyses, summarising these analyses into a
report, iterating over discussions until a consensus is reached, and ultimately
making a decision. Our work particularly focuses on the zero-shot scenario, our
results on nine data sets (MedQA, MedMCQA, PubMedQA, and six subtasks from
MMLU) establish that our proposed MC framework excels at mining and harnessing
the medical expertise in LLMs, as well as extending its reasoning abilities.
Based on these outcomes, we further conduct a human evaluation to pinpoint and
categorize common errors within our method, as well as ablation studies aimed
at understanding the impact of various factors on overall performance. Our code
can be found at \url{https://github.com/gersteinlab/MedAgents}.


7. [R-Tuning: Teaching Large Language Models to Refuse Unknown Questions](http://arxiv.org/abs/2311.09677v1), Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, Tong Zhang, 16-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs) have revolutionized numerous domains with their
impressive performance but still face their challenges. A predominant issue is
the propensity for these models to generate non-existent facts, a concern
termed hallucination. Our research is motivated by the observation that
previous instruction tuning methods force the model to complete a sentence no
matter whether the model knows the knowledge or not. When the question is out
of the parametric knowledge, it will try to make up something and fail to
indicate when it lacks knowledge. In this paper, we present a new approach
called Refusal-Aware Instruction Tuning (R-Tuning). This approach is formalized
by first identifying the knowledge gap between parametric knowledge and the
instruction tuning data. Then, we construct the refusal-aware data based on the
knowledge intersection, to tune LLMs to refrain from responding to questions
beyond its parametric knowledge. Experimental results demonstrate this new
instruction tuning approach effectively improves a model's ability to answer
known questions and refrain from answering unknown questions. Furthermore, when
tested on out-of-domain datasets, the refusal ability was found to be a
meta-skill that could be generalized to other tasks. Further analysis
surprisingly finds that learning the uncertainty during training displays a
better ability to estimate uncertainty than uncertainty-based testing. Our code
will be released at https://github.com/shizhediao/R-Tuning.


7. [Generalized products and Lorentzian length spaces](http://arxiv.org/abs/2311.10691v1), Elefterios Soultanis, 17-11-2023
    ### Abstract
    We construct a Lorentzian length space with an orthogonal splitting on a
product $I\times X$ of an interval and a metric space, and use this framework
to consider the relationship between metric and causal geometry, as well as
synthetic time-like Ricci curvature bounds.
  The generalized Lorentzian product naturally has a Lorentzian length
structure but can fail the push-up condition in general. We recover the push-up
property under a log-Lipschitz condition on the time variable and establish
sufficient conditions for global hyperbolicity. Moreover we formulate time-like
Ricci curvature bounds without push-up and regularity assumptions, and obtain a
partial rigidity of the splitting under a strong energy condition.


7. [Testing Language Model Agents Safely in the Wild](http://arxiv.org/abs/2311.10538v3), Silen Naihin, David Atkinson, Marc Green, Merwane Hamadi, Craig Swift, Douglas Schonholtz, Adam Tauman Kalai, David Bau, 17-11-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    A prerequisite for safe autonomy-in-the-wild is safe testing-in-the-wild. Yet
real-world autonomous tests face several unique safety challenges, both due to
the possibility of causing harm during a test, as well as the risk of
encountering new unsafe agent behavior through interactions with real-world and
potentially malicious actors. We propose a framework for conducting safe
autonomous agent tests on the open internet: agent actions are audited by a
context-sensitive monitor that enforces a stringent safety boundary to stop an
unsafe test, with suspect behavior ranked and logged to be examined by humans.
We design a basic safety monitor (AgentMonitor) that is flexible enough to
monitor existing LLM agents, and, using an adversarial simulated agent, we
measure its ability to identify and stop unsafe situations. Then we apply the
AgentMonitor on a battery of real-world tests of AutoGPT, and we identify
several limitations and challenges that will face the creation of safe
in-the-wild tests as autonomous agents grow more capable.


7. [Orca 2: Teaching Small Language Models How to Reason](http://arxiv.org/abs/2311.11045v2), Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah, 18-11-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Orca 1 learns from rich signals, such as explanation traces, allowing it to
outperform conventional instruction-tuned models on benchmarks like BigBench
Hard and AGIEval. In Orca 2, we continue exploring how improved training
signals can enhance smaller LMs' reasoning abilities. Research on training
small LMs has often relied on imitation learning to replicate the output of
more capable models. We contend that excessive emphasis on imitation may
restrict the potential of smaller models. We seek to teach small LMs to employ
different solution strategies for different tasks, potentially different from
the one used by the larger model. For example, while larger models might
provide a direct answer to a complex task, smaller models may not have the same
capacity. In Orca 2, we teach the model various reasoning techniques
(step-by-step, recall then generate, recall-reason-generate, direct answer,
etc.). More crucially, we aim to help the model learn to determine the most
effective solution strategy for each task. We evaluate Orca 2 using a
comprehensive set of 15 diverse benchmarks (corresponding to approximately 100
tasks and over 36,000 unique prompts). Orca 2 significantly surpasses models of
similar size and attains performance levels similar or better to those of
models 5-10x larger, as assessed on complex tasks that test advanced reasoning
abilities in zero-shot settings. make Orca 2 weights publicly available at
aka.ms/orca-lm to support research on the development, evaluation, and
alignment of smaller LMs


7. [Igniting Language Intelligence: The Hitchhiker's Guide From
  Chain-of-Thought Reasoning to Language Agents](http://arxiv.org/abs/2311.11797v1), Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, Hai Zhao, 20-11-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) have dramatically enhanced the field of language
intelligence, as demonstrably evidenced by their formidable empirical
performance across a spectrum of complex reasoning tasks. Additionally,
theoretical proofs have illuminated their emergent reasoning capabilities,
providing a compelling showcase of their advanced cognitive abilities in
linguistic contexts. Critical to their remarkable efficacy in handling complex
reasoning tasks, LLMs leverage the intriguing chain-of-thought (CoT) reasoning
techniques, obliging them to formulate intermediate steps en route to deriving
an answer. The CoT reasoning approach has not only exhibited proficiency in
amplifying reasoning performance but also in enhancing interpretability,
controllability, and flexibility. In light of these merits, recent research
endeavors have extended CoT reasoning methodologies to nurture the development
of autonomous language agents, which adeptly adhere to language instructions
and execute actions within varied environments. This survey paper orchestrates
a thorough discourse, penetrating vital research dimensions, encompassing: (i)
the foundational mechanics of CoT techniques, with a focus on elucidating the
circumstances and justification behind its efficacy; (ii) the paradigm shift in
CoT; and (iii) the burgeoning of language agents fortified by CoT approaches.
Prospective research avenues envelop explorations into generalization,
efficiency, customization, scaling, and safety. This paper caters to a wide
audience, including beginners seeking comprehensive knowledge of CoT reasoning
and language agents, as well as experienced researchers interested in
foundational mechanics and engaging in cutting-edge discussions on these
topics. A repository for the related papers is available at
https://github.com/Zoeyyao27/CoT-Igniting-Agent.


7. [From Classification to Clinical Insights: Towards Analyzing and
  Reasoning About Mobile and Behavioral Health Data With Large Language Models](http://arxiv.org/abs/2311.13063v2), Zachary Englhardt, Chengqian Ma, Margaret E. Morris, Xuhai "Orson" Xu, Chun-Cheng Chang, Lianhui Qin, Daniel McDuff, Xin Liu, Shwetak Patel, Vikram Iyer, 21-11-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Passively collected behavioral health data from ubiquitous sensors holds
significant promise to provide mental health professionals insights from
patient's daily lives; however, developing analysis tools to use this data in
clinical practice requires addressing challenges of generalization across
devices and weak or ambiguous correlations between the measured signals and an
individual's mental health. To address these challenges, we take a novel
approach that leverages large language models (LLMs) to synthesize clinically
useful insights from multi-sensor data. We develop chain of thought prompting
methods that use LLMs to generate reasoning about how trends in data such as
step count and sleep relate to conditions like depression and anxiety. We first
demonstrate binary depression classification with LLMs achieving accuracies of
61.1% which exceed the state of the art. While it is not robust for clinical
use, this leads us to our key finding: even more impactful and valued than
classification is a new human-AI collaboration approach in which clinician
experts interactively query these tools and combine their domain expertise and
context about the patient with AI generated reasoning to support clinical
decision-making. We find models like GPT-4 correctly reference numerical data
75% of the time, and clinician participants express strong interest in using
this approach to interpret self-tracking data.


7. [Algorithm Evolution Using Large Language Model](http://arxiv.org/abs/2311.15249v1), Fei Liu, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang, 26-11-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    Optimization can be found in many real-life applications. Designing an
effective algorithm for a specific optimization problem typically requires a
tedious amount of effort from human experts with domain knowledge and algorithm
design skills. In this paper, we propose a novel approach called Algorithm
Evolution using Large Language Model (AEL). It utilizes a large language model
(LLM) to automatically generate optimization algorithms via an evolutionary
framework. AEL does algorithm-level evolution without model training. Human
effort and requirements for domain knowledge can be significantly reduced. We
take constructive methods for the salesman traveling problem as a test example,
we show that the constructive algorithm obtained by AEL outperforms simple
hand-crafted and LLM-generated heuristics. Compared with other domain deep
learning model-based algorithms, these methods exhibit excellent scalability
across different problem sizes. AEL is also very different from previous
attempts that utilize LLMs as search operators in algorithms.


7. [ChatGPT's One-year Anniversary: Are Open-Source Large Language Models
  Catching up?](http://arxiv.org/abs/2311.16989v4), Hailin Chen, Fangkai Jiao, Xingxuan Li, Chengwei Qin, Mathieu Ravaut, Ruochen Zhao, Caiming Xiong, Shafiq Joty, 28-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Upon its release in late 2022, ChatGPT has brought a seismic shift in the
entire landscape of AI, both in research and commerce. Through
instruction-tuning a large language model (LLM) with supervised fine-tuning and
reinforcement learning from human feedback, it showed that a model could answer
human questions and follow instructions on a broad panel of tasks. Following
this success, interests in LLMs have intensified, with new LLMs flourishing at
frequent interval across academia and industry, including many start-ups
focused on LLMs. While closed-source LLMs (e.g., OpenAI's GPT, Anthropic's
Claude) generally outperform their open-source counterparts, the progress on
the latter has been rapid with claims of achieving parity or even better on
certain tasks. This has crucial implications not only on research but also on
business. In this work, on the first anniversary of ChatGPT, we provide an
exhaustive overview of this success, surveying all tasks where an open-source
LLM has claimed to be on par or better than ChatGPT.


7. [ChatGPT's One-year Anniversary: Are Open-Source Large Language Models
  Catching up?](http://arxiv.org/abs/2311.16989v4), Hailin Chen, Fangkai Jiao, Xingxuan Li, Chengwei Qin, Mathieu Ravaut, Ruochen Zhao, Caiming Xiong, Shafiq Joty, 28-11-2023
     ### Categories
     Computation and Language
    ### Abstract
    Upon its release in late 2022, ChatGPT has brought a seismic shift in the
entire landscape of AI, both in research and commerce. Through
instruction-tuning a large language model (LLM) with supervised fine-tuning and
reinforcement learning from human feedback, it showed that a model could answer
human questions and follow instructions on a broad panel of tasks. Following
this success, interests in LLMs have intensified, with new LLMs flourishing at
frequent interval across academia and industry, including many start-ups
focused on LLMs. While closed-source LLMs (e.g., OpenAI's GPT, Anthropic's
Claude) generally outperform their open-source counterparts, the progress on
the latter has been rapid with claims of achieving parity or even better on
certain tasks. This has crucial implications not only on research but also on
business. In this work, on the first anniversary of ChatGPT, we provide an
exhaustive overview of this success, surveying all tasks where an open-source
LLM has claimed to be on par or better than ChatGPT.


7. [A collection of principles for guiding and evaluating large language
  models](http://arxiv.org/abs/2312.10059v1), Konstantin Hebenstreit, Robert Praas, Matthias Samwald, 04-12-2023
    ### Abstract
    Large language models (LLMs) demonstrate outstanding capabilities, but
challenges remain regarding their ability to solve complex reasoning tasks, as
well as their transparency, robustness, truthfulness, and ethical alignment. In
this preliminary study, we compile a set of core principles for steering and
evaluating the reasoning of LLMs by curating literature from several relevant
strands of work: structured reasoning in LLMs, self-evaluation/self-reflection,
explainability, AI system safety/security, guidelines for human critical
thinking, and ethical/regulatory guidelines for AI. We identify and curate a
list of 220 principles from literature, and derive a set of 37 core principles
organized into seven categories: assumptions and perspectives, reasoning,
information and evidence, robustness and security, ethics, utility, and
implications. We conduct a small-scale expert survey, eliciting the subjective
importance experts assign to different principles and lay out avenues for
future work beyond our preliminary results. We envision that the development of
a shared model of principles can serve multiple purposes: monitoring and
steering models at inference time, improving model behavior during training,
and guiding human evaluation of model reasoning.


7. [Data Management For Large Language Models: A Survey](http://arxiv.org/abs/2312.01700v2), Zige Wang, Wanjun Zhong, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Lifeng Shang, Xin Jiang, Qun Liu, 04-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Data plays a fundamental role in the training of Large Language Models
(LLMs). Effective data management, particularly in the formulation of a
well-suited training dataset, holds significance for enhancing model
performance and improving training efficiency during pretraining and supervised
fine-tuning phases. Despite the considerable importance of data management, the
current research community still falls short in providing a systematic analysis
of the rationale behind management strategy selection, its consequential
effects, methodologies for evaluating curated datasets, and the ongoing pursuit
of improved strategies. Consequently, the exploration of data management has
attracted more and more attention among the research community. This survey
provides a comprehensive overview of current research in data management within
both the pretraining and supervised fine-tuning stages of LLMs, covering
various noteworthy aspects of data management strategy design: data quantity,
data quality, domain/task composition, etc. Looking toward the future, we
extrapolate existing challenges and outline promising directions for
development in this field. Therefore, this survey serves as a guiding resource
for practitioners aspiring to construct powerful LLMs through effective data
management practices. The collection of the latest papers is available at
https://github.com/ZigeW/data_management_LLM.


7. [Creative Agents: Empowering Agents with Imagination for Creative Tasks](http://arxiv.org/abs/2312.02519v1), Chi Zhang, Penglin Cai, Yuhui Fu, Haoqi Yuan, Zongqing Lu, 05-12-2023
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    We study building embodied agents for open-ended creative tasks. While
existing methods build instruction-following agents that can perform diverse
open-ended tasks, none of them demonstrates creativity -- the ability to give
novel and diverse task solutions implicit in the language instructions. This
limitation comes from their inability to convert abstract language instructions
into concrete task goals in the environment and perform long-horizon planning
for such complicated goals. Given the observation that humans perform creative
tasks with the help of imagination, we propose a class of solutions for
creative agents, where the controller is enhanced with an imaginator that
generates detailed imaginations of task outcomes conditioned on language
instructions. We introduce several approaches to implementing the components of
creative agents. We implement the imaginator with either a large language model
for textual imagination or a diffusion model for visual imagination. The
controller can either be a behavior-cloning policy learned from data or a
pre-trained foundation model generating executable codes in the environment. We
benchmark creative tasks with the challenging open-world game Minecraft, where
the agents are asked to create diverse buildings given free-form language
instructions. In addition, we propose novel evaluation metrics for open-ended
creative tasks utilizing GPT-4V, which holds many advantages over existing
metrics. We perform a detailed experimental analysis of creative agents,
showing that creative agents are the first AI agents accomplishing diverse
building creation in the survival mode of Minecraft. Our benchmark and models
are open-source for future research on creative agents
(https://github.com/PKU-RL/Creative-Agents).


7. [Foundation Models for Weather and Climate Data Understanding: A
  Comprehensive Survey](http://arxiv.org/abs/2312.03014v1), Shengchao Chen, Guodong Long, Jing Jiang, Dikai Liu, Chengqi Zhang, 05-12-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    As artificial intelligence (AI) continues to rapidly evolve, the realm of
Earth and atmospheric sciences is increasingly adopting data-driven models,
powered by progressive developments in deep learning (DL). Specifically, DL
techniques are extensively utilized to decode the chaotic and nonlinear aspects
of Earth systems, and to address climate challenges via understanding weather
and climate data. Cutting-edge performance on specific tasks within narrower
spatio-temporal scales has been achieved recently through DL. The rise of large
models, specifically large language models (LLMs), has enabled fine-tuning
processes that yield remarkable outcomes across various downstream tasks,
thereby propelling the advancement of general AI. However, we are still
navigating the initial stages of crafting general AI for weather and climate.
In this survey, we offer an exhaustive, timely overview of state-of-the-art AI
methodologies specifically engineered for weather and climate data, with a
special focus on time series and text data. Our primary coverage encompasses
four critical aspects: types of weather and climate data, principal model
architectures, model scopes and applications, and datasets for weather and
climate. Furthermore, in relation to the creation and application of foundation
models for weather and climate data understanding, we delve into the field's
prevailing challenges, offer crucial insights, and propose detailed avenues for
future research. This comprehensive approach equips practitioners with the
requisite knowledge to make substantial progress in this domain. Our survey
encapsulates the most recent breakthroughs in research on large, data-driven
models for weather and climate data understanding, emphasizing robust
foundations, current advancements, practical applications, crucial resources,
and prospective research opportunities.


7. [An LLM Compiler for Parallel Function Calling](http://arxiv.org/abs/2312.04511v1), Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami, 07-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    Large Language Models (LLMs) have shown remarkable results on various complex
reasoning benchmarks. The reasoning capabilities of LLMs enable them to execute
function calls, using user-provided functions to overcome their inherent
limitations, such as knowledge cutoffs, poor arithmetic skills, or lack of
access to private data. This development has expanded LLMs' scope to include
multi-function calling, where LLMs are equipped with a variety of functions and
select the proper functions based on the context. Multi-function calling
abilities of LLMs have catalyzed LLM-based software development, allowing them
to tackle more complex problems. However, current methods for multi-function
calling often require sequential reasoning and acting for each function which
can result in high latency, cost, and sometimes inaccurate behavior. To address
this, we introduce LLMCompiler, which executes functions in parallel to
efficiently orchestrate multi-function calling. Drawing from the principles of
classical compilers, LLMCompiler streamlines parallel function calling with
three components: (i) an LLM Planner, formulating execution strategies and
dependencies; (ii) a Task Fetching Unit, dispatching function calling tasks;
and (iii) an Executor, executing these tasks in parallel. LLMCompiler
automatically computes an optimized orchestration for the function calls and
can be used with open-source models such as LLaMA-2. We have benchmarked
LLMCompiler on a range of tasks including cases with non-trivial
inter-dependency between function calls, as well as cases that require dynamic
replanning based on intermediate results. We observe consistent latency speedup
of up to 3.7x, cost savings of up to 6.7x, and accuracy improvement of up to
~9% as compared to ReAct. Additionally, LLMCompiler achieves up to 1.35x
latency gain over OpenAI's recent parallel function calling, while achieving
similar accuracy.


7. [Are We Testing or Being Tested? Exploring the Practical Applications of
  Large Language Models in Software Testing](http://arxiv.org/abs/2312.04860v1), Robson Santos, Italo Santos, Cleyton Magalhaes, Ronnie de Souza Santos, 08-12-2023
    ### Abstract
    A Large Language Model (LLM) represents a cutting-edge artificial
intelligence model that generates coherent content, including grammatically
precise sentences, human-like paragraphs, and syntactically accurate code
snippets. LLMs can play a pivotal role in software development, including
software testing. LLMs go beyond traditional roles such as requirement analysis
and documentation and can support test case generation, making them valuable
tools that significantly enhance testing practices within the field. Hence, we
explore the practical application of LLMs in software testing within an
industrial setting, focusing on their current use by professional testers. In
this context, rather than relying on existing data, we conducted a
cross-sectional survey and collected data within real working contexts,
specifically, engaging with practitioners in industrial settings. We applied
quantitative and qualitative techniques to analyze and synthesize our collected
data. Our findings demonstrate that LLMs effectively enhance testing documents
and significantly assist testing professionals in programming tasks like
debugging and test case automation. LLMs can support individuals engaged in
manual testing who need to code. However, it is crucial to emphasize that, at
this early stage, software testing professionals should use LLMs with caution
while well-defined methods and guidelines are being built for the secure
adoption of these tools.


7. [KwaiAgents: Generalized Information-seeking Agent System with Large
  Language Models](http://arxiv.org/abs/2312.04889v3), Haojie Pan, Zepeng Zhai, Hao Yuan, Yaojia Lv, Ruiji Fu, Ming Liu, Zhongyuan Wang, Bing Qin, 08-12-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Driven by curiosity, humans have continually sought to explore and understand
the world around them, leading to the invention of various tools to satiate
this inquisitiveness. Despite not having the capacity to process and memorize
vast amounts of information in their brains, humans excel in critical thinking,
planning, reflection, and harnessing available tools to interact with and
interpret the world, enabling them to find answers efficiently. The recent
advancements in large language models (LLMs) suggest that machines might also
possess the aforementioned human-like capabilities, allowing them to exhibit
powerful abilities even with a constrained parameter count. In this paper, we
introduce KwaiAgents, a generalized information-seeking agent system based on
LLMs. Within KwaiAgents, we propose an agent system that employs LLMs as its
cognitive core, which is capable of understanding a user's query, behavior
guidelines, and referencing external documents. The agent can also update and
retrieve information from its internal memory, plan and execute actions using a
time-aware search-browse toolkit, and ultimately provide a comprehensive
response. We further investigate the system's performance when powered by LLMs
less advanced than GPT-4, and introduce the Meta-Agent Tuning (MAT) framework,
designed to ensure even an open-sourced 7B or 13B model performs well among
many agent systems. We exploit both benchmark and human evaluations to
systematically validate these capabilities. Extensive experiments show the
superiority of our agent system compared to other autonomous agents and
highlight the enhanced generalized agent-abilities of our fine-tuned LLMs.


7. [Large-scale Training of Foundation Models for Wearable Biosignals](http://arxiv.org/abs/2312.05409v1), Salar Abbaspourazad, Oussama Elachqar, Andrew C. Miller, Saba Emrani, Udhyakumar Nallasamy, Ian Shapiro, 08-12-2023
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    Tracking biosignals is crucial for monitoring wellness and preempting the
development of severe medical conditions. Today, wearable devices can
conveniently record various biosignals, creating the opportunity to monitor
health status without disruption to one's daily routine. Despite widespread use
of wearable devices and existing digital biomarkers, the absence of curated
data with annotated medical labels hinders the development of new biomarkers to
measure common health conditions. In fact, medical datasets are usually small
in comparison to other domains, which is an obstacle for developing neural
network models for biosignals. To address this challenge, we have employed
self-supervised learning using the unlabeled sensor data collected under
informed consent from the large longitudinal Apple Heart and Movement Study
(AHMS) to train foundation models for two common biosignals:
photoplethysmography (PPG) and electrocardiogram (ECG) recorded on Apple Watch.
We curated PPG and ECG datasets from AHMS that include data from ~141K
participants spanning ~3 years. Our self-supervised learning framework includes
participant level positive pair selection, stochastic augmentation module and a
regularized contrastive loss optimized with momentum training, and generalizes
well to both PPG and ECG modalities. We show that the pre-trained foundation
models readily encode information regarding participants' demographics and
health conditions. To the best of our knowledge, this is the first study that
builds foundation models using large-scale PPG and ECG data collected via
wearable consumer devices $\unicode{x2013}$ prior works have commonly used
smaller-size datasets collected in clinical and experimental settings. We
believe PPG and ECG foundation models can enhance future wearable devices by
reducing the reliance on labeled data and hold the potential to help the users
improve their health.


7. [LLM360: Towards Fully Transparent Open-Source LLMs](http://arxiv.org/abs/2312.06550v1), Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, Eric P. Xing, 11-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    The recent surge in open-source Large Language Models (LLMs), such as LLaMA,
Falcon, and Mistral, provides diverse options for AI practitioners and
researchers. However, most LLMs have only released partial artifacts, such as
the final model weights or inference code, and technical reports increasingly
limit their scope to high-level design choices and surface statistics. These
choices hinder progress in the field by degrading transparency into the
training of LLMs and forcing teams to rediscover many details in the training
process. We present LLM360, an initiative to fully open-source LLMs, which
advocates for all training code and data, model checkpoints, and intermediate
results to be made available to the community. The goal of LLM360 is to support
open and collaborative AI research by making the end-to-end LLM training
process transparent and reproducible by everyone. As a first step of LLM360, we
release two 7B parameter LLMs pre-trained from scratch, Amber and CrystalCoder,
including their training code, data, intermediate checkpoints, and analyses (at
https://www.llm360.ai). We are committed to continually pushing the boundaries
of LLMs through this open-source effort. More large-scale and stronger models
are underway and will be released in the future.


7. ["I Want It That Way": Enabling Interactive Decision Support Using Large
  Language Models and Constraint Programming](http://arxiv.org/abs/2312.06908v1), Connor Lawless, Jakob Schoeffer, Lindy Le, Kael Rowan, Shilad Sen, Cristina St. Hill, Jina Suh, Bahar Sarrafzadeh, 12-12-2023
    ### Abstract
    A critical factor in the success of decision support systems is the accurate
modeling of user preferences. Psychology research has demonstrated that users
often develop their preferences during the elicitation process, highlighting
the pivotal role of system-user interaction in developing personalized systems.
This paper introduces a novel approach, combining Large Language Models (LLMs)
with Constraint Programming to facilitate interactive decision support. We
study this hybrid framework through the lens of meeting scheduling, a
time-consuming daily activity faced by a multitude of information workers. We
conduct three studies to evaluate the novel framework, including a diary study
(n=64) to characterize contextual scheduling preferences, a quantitative
evaluation of the system's performance, and a user study (n=10) with a
prototype system. Our work highlights the potential for a hybrid LLM and
optimization approach for iterative preference elicitation and design
considerations for building systems that support human-system collaborative
decision-making processes.


7. [Alignment for Honesty](http://arxiv.org/abs/2312.07000v1), Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, Pengfei Liu, 12-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Recent research has made significant strides in applying alignment techniques
to enhance the helpfulness and harmlessness of large language models (LLMs) in
accordance with human intentions. In this paper, we argue for the importance of
alignment for honesty, ensuring that LLMs proactively refuse to answer
questions when they lack knowledge, while still not being overly conservative.
However, a pivotal aspect of alignment for honesty involves discerning the
limits of an LLM's knowledge, which is far from straightforward. This challenge
demands comprehensive solutions in terms of metric development, benchmark
creation, and training methodologies. In this paper, we address these
challenges by first establishing a precise problem definition and defining
``honesty'' inspired by the Analects of Confucius. This serves as a cornerstone
for developing metrics that effectively measure an LLM's honesty by quantifying
its progress post-alignment. Furthermore, we introduce a flexible training
framework which is further instantiated by several efficient fine-tuning
techniques that emphasize honesty without sacrificing performance on other
tasks. Our extensive experiments reveal that these aligned models show a marked
increase in honesty, as indicated by our proposed metrics. We open-source a
wealth of resources to facilitate future research at
https://github.com/GAIR-NLP/alignment-for-honesty, including honesty-aligned
models, training and evaluation datasets for honesty alignment, concept
glossary, as well as all relevant source code.


7. [Efficient Few-Shot Clinical Task Adaptation with Large Language Models](http://arxiv.org/abs/2312.07125v1), Kaipeng Zheng, Weiran Huang, Lichao Sun, 12-12-2023
    ### Abstract
    Few-shot learning has been studied to adapt models to tasks with very few
samples. It holds profound significance, particularly in clinical tasks, due to
the high annotation cost of medical images. Several works have explored
few-shot learning on medical images, yet they still require a large number of
medical images for pre-training models to gain domain-specific priors. Vision
foundation models recently have achieved remarkable success in natural images.
Hence, adapting rapidly advancing vision foundation models from natural images
to few-shot clinical tasks holds great promise. MedFMC has recently organized a
challenge to shed more light on this topic at NeurIPS 2023. In this work, we
present our challenge solution. We observe that a simple variant of fine-tuning
with partial freezing shows remarkable performance. Empirical evidence
demonstrates that this approach could outperform various common fine-tuning
methods under limited sample sizes. Additionally, we explore enhanced
utilization of semantic supervision to boost performance. We propose a novel
approach that contextualizes labels via large language models (LLMs). Our
findings reveal that the context generated by LLMs significantly enhances the
discrimination of semantic embeddings for similar categories, resulting in a
notable performance improvement of 3%-5% in 1-shot settings compared to
commonly employed one-hot labels and other semantic supervision methods. Our
solution secures the 1st place in the MedFMC challenge.


7. [LLM in a flash: Efficient Large Language Model Inference with Limited
  Memory](http://arxiv.org/abs/2312.11514v2), Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar, 12-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large language models (LLMs) are central to modern natural language
processing, delivering exceptional performance in various tasks. However, their
substantial computational and memory requirements present challenges,
especially for devices with limited DRAM capacity. This paper tackles the
challenge of efficiently running LLMs that exceed the available DRAM capacity
by storing the model parameters in flash memory, but bringing them on demand to
DRAM. Our method involves constructing an inference cost model that takes into
account the characteristics of flash memory, guiding us to optimize in two
critical areas: reducing the volume of data transferred from flash and reading
data in larger, more contiguous chunks. Within this hardware-informed
framework, we introduce two principal techniques. First, "windowing"
strategically reduces data transfer by reusing previously activated neurons,
and second, "row-column bundling", tailored to the sequential data access
strengths of flash memory, increases the size of data chunks read from flash
memory. These methods collectively enable running models up to twice the size
of the available DRAM, with a 4-5x and 20-25x increase in inference speed
compared to naive loading approaches in CPU and GPU, respectively. Our
integration of sparsity awareness, context-adaptive loading, and a
hardware-oriented design paves the way for effective inference of LLMs on
devices with limited memory.


7. [LLMEval: A Preliminary Study on How to Evaluate Large Language Models](http://arxiv.org/abs/2312.07398v2), Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, Xuanjing Huang, 12-12-2023
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    Recently, the evaluation of Large Language Models has emerged as a popular
area of research. The three crucial questions for LLM evaluation are ``what,
where, and how to evaluate''. However, the existing research mainly focuses on
the first two questions, which are basically what tasks to give the LLM during
testing and what kind of knowledge it should deal with. As for the third
question, which is about what standards to use, the types of evaluators, how to
score, and how to rank, there hasn't been much discussion. In this paper, we
analyze evaluation methods by comparing various criteria with both manual and
automatic evaluation, utilizing onsite, crowd-sourcing, public annotators and
GPT-4, with different scoring methods and ranking systems. We propose a new
dataset, LLMEval and conduct evaluations on 20 LLMs. A total of 2,186
individuals participated, leading to the generation of 243,337 manual
annotations and 57,511 automatic evaluation results. We perform comparisons and
analyses of different settings and conduct 10 conclusions that can provide some
insights for evaluating LLM in the future. The dataset and the results are
publicly available at https://github.com/llmeval .


7. [SM70: A Large Language Model for Medical Devices](http://arxiv.org/abs/2312.06974v1), Anubhav Bhatti, Surajsinh Parmar, San Lee, 12-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We are introducing SM70, a 70 billion-parameter Large Language Model that is
specifically designed for SpassMed's medical devices under the brand name
'JEE1' (pronounced as G1 and means 'Life'). This large language model provides
more accurate and safe responses to medical-domain questions. To fine-tune
SM70, we used around 800K data entries from the publicly available dataset
MedAlpaca. The Llama2 70B open-sourced model served as the foundation for SM70,
and we employed the QLoRA technique for fine-tuning. The evaluation is
conducted across three benchmark datasets - MEDQA - USMLE, PUBMEDQA, and USMLE
- each representing a unique aspect of medical knowledge and reasoning. The
performance of SM70 is contrasted with other notable LLMs, including Llama2
70B, Clinical Camel 70 (CC70), GPT 3.5, GPT 4, and Med-Palm, to provide a
comparative understanding of its capabilities within the medical domain. Our
results indicate that SM70 outperforms several established models in these
datasets, showcasing its proficiency in handling a range of medical queries,
from fact-based questions derived from PubMed abstracts to complex clinical
decision-making scenarios. The robust performance of SM70, particularly in the
USMLE and PUBMEDQA datasets, suggests its potential as an effective tool in
clinical decision support and medical information retrieval. Despite its
promising results, the paper also acknowledges the areas where SM70 lags behind
the most advanced model, GPT 4, thereby highlighting the need for further
development, especially in tasks demanding extensive medical knowledge and
intricate reasoning.


7. [Distributed Inference and Fine-tuning of Large Language Models Over The
  Internet](http://arxiv.org/abs/2312.08361v1), Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, Colin Raffel, 13-12-2023
     ### Categories
     Machine Learning
    ### Abstract
    Large language models (LLMs) are useful in many NLP tasks and become more
capable with size, with the best open-source models having over 50 billion
parameters. However, using these 50B+ models requires high-end hardware, making
them inaccessible to most researchers. In this work, we investigate methods for
cost-efficient inference and fine-tuning of LLMs, comparing local and
distributed strategies. We observe that a large enough model (50B+) can run
efficiently even on geodistributed devices in a consumer-grade network. This
could allow running LLM efficiently by pooling together idle compute resources
of multiple research groups and volunteers. We address two open problems: (1)
how to perform inference and fine-tuning reliably if any device can disconnect
abruptly and (2) how to partition LLMs between devices with uneven hardware,
joining and leaving at will. In order to do that, we develop special
fault-tolerant inference algorithms and load-balancing protocols that
automatically assign devices to maximize the total system throughput. We
showcase these algorithms in Petals - a decentralized system that runs Llama 2
(70B) and BLOOM (176B) over the Internet up to 10x faster than offloading for
interactive generation. We evaluate the performance of our system in simulated
conditions and a real-world setup spanning two continents.


7. [PromptBench: A Unified Library for Evaluation of Large Language Models](http://arxiv.org/abs/2312.07910v2), Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie, 13-12-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    The evaluation of large language models (LLMs) is crucial to assess their
performance and mitigate potential security risks. In this paper, we introduce
PromptBench, a unified library to evaluate LLMs. It consists of several key
components that are easily used and extended by researchers: prompt
construction, prompt engineering, dataset and model loading, adversarial prompt
attack, dynamic evaluation protocols, and analysis tools. PromptBench is
designed to be an open, general, and flexible codebase for research purposes
that can facilitate original study in creating new benchmarks, deploying
downstream applications, and designing new evaluation protocols. The code is
available at: https://github.com/microsoft/promptbench and will be continuously
supported.


7. [CogAgent: A Visual Language Model for GUI Agents](http://arxiv.org/abs/2312.08914v2), Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, Jie Tang, 14-12-2023
    ### Abstract
    People are spending an enormous amount of time on digital devices through
graphical user interfaces (GUIs), e.g., computer or smartphone screens. Large
language models (LLMs) such as ChatGPT can assist people in tasks like writing
emails, but struggle to understand and interact with GUIs, thus limiting their
potential to increase automation levels. In this paper, we introduce CogAgent,
an 18-billion-parameter visual language model (VLM) specializing in GUI
understanding and navigation. By utilizing both low-resolution and
high-resolution image encoders, CogAgent supports input at a resolution of
1120*1120, enabling it to recognize tiny page elements and text. As a
generalist visual language model, CogAgent achieves the state of the art on
five text-rich and four general VQA benchmarks, including VQAv2, OK-VQA,
Text-VQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. CogAgent, using
only screenshots as input, outperforms LLM-based methods that consume extracted
HTML text on both PC and Android GUI navigation tasks -- Mind2Web and AITW,
advancing the state of the art. The model and codes are available at
https://github.com/THUDM/CogVLM .


7. [TigerBot: An Open Multilingual Multitask LLM](http://arxiv.org/abs/2312.08688v2), Ye Chen, Wei Cai, Liangmin Wu, Xiaowei Li, Zhanxuan Xin, Cong Fu, 14-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We release and introduce the TigerBot family of large language models (LLMs),
consisting of base and chat models, sized from 7, 13, 70 and 180 billion
parameters. We develop our models embarking from Llama-2 and BLOOM, and push
the boundary further in data, training algorithm, infrastructure, and
application tools. Our models yield meaningful performance gain over SOTA
open-source models, e.g., Llama-2, specifically 6% gain in English and 20% gain
in Chinese. TigerBot model family also achieves leading performance in major
academic and industrial benchmarks and leaderboards. We believe that TigerBot
represents just a snapshot of lightning-fast progression in LLM open-source
community. Therefore, we are thrilled to give back by publicly releasing our
models and reporting our approach behind, with additional emphases on building
SOTA LLMs in a democratized way and making LLMs of use in real-world
applications.


7. [Catwalk: A Unified Language Model Evaluation Framework for Many Datasets](http://arxiv.org/abs/2312.10253v1), Dirk Groeneveld, Anas Awadalla, Iz Beltagy, Akshita Bhagia, Ian Magnusson, Hao Peng, Oyvind Tafjord, Pete Walsh, Kyle Richardson, Jesse Dodge, 15-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    The success of large language models has shifted the evaluation paradigms in
natural language processing (NLP). The community's interest has drifted towards
comparing NLP models across many tasks, domains, and datasets, often at an
extreme scale. This imposes new engineering challenges: efforts in constructing
datasets and models have been fragmented, and their formats and interfaces are
incompatible. As a result, it often takes extensive (re)implementation efforts
to make fair and controlled comparisons at scale.
  Catwalk aims to address these issues. Catwalk provides a unified interface to
a broad range of existing NLP datasets and models, ranging from both canonical
supervised training and fine-tuning, to more modern paradigms like in-context
learning. Its carefully-designed abstractions allow for easy extensions to many
others. Catwalk substantially lowers the barriers to conducting controlled
experiments at scale. For example, we finetuned and evaluated over 64 models on
over 86 datasets with a single command, without writing any code. Maintained by
the AllenNLP team at the Allen Institute for Artificial Intelligence (AI2),
Catwalk is an ongoing open-source effort: https://github.com/allenai/catwalk.


7. [Extending Context Window of Large Language Models via Semantic
  Compression](http://arxiv.org/abs/2312.09571v1), Weizhi Fei, Xueyan Niu, Pingyi Zhou, Lu Hou, Bo Bai, Lei Deng, Wei Han, 15-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    Transformer-based Large Language Models (LLMs) often impose limitations on
the length of the text input to ensure the generation of fluent and relevant
responses. This constraint restricts their applicability in scenarios involving
long texts. We propose a novel semantic compression method that enables
generalization to texts that are 6-8 times longer, without incurring
significant computational costs or requiring fine-tuning. Our proposed
framework draws inspiration from source coding in information theory and
employs a pre-trained model to reduce the semantic redundancy of long inputs
before passing them to the LLMs for downstream tasks. Experimental results
demonstrate that our method effectively extends the context window of LLMs
across a range of tasks including question answering, summarization, few-shot
learning, and information retrieval. Furthermore, the proposed semantic
compression method exhibits consistent fluency in text generation while
reducing the associated computational overhead.


7. [Faithful Persona-based Conversational Dataset Generation with Large
  Language Models](http://arxiv.org/abs/2312.10007v1), Pegah Jandaghi, XiangHai Sheng, Xinyi Bai, Jay Pujara, Hakim Sidahmed, 15-12-2023
     ### Categories
     Computation and Language, Machine Learning
    ### Abstract
    High-quality conversational datasets are essential for developing AI models
that can communicate with users. One way to foster deeper interactions between
a chatbot and its user is through personas, aspects of the user's character
that provide insights into their personality, motivations, and behaviors.
Training Natural Language Processing (NLP) models on a diverse and
comprehensive persona-based dataset can lead to conversational models that
create a deeper connection with the user, and maintain their engagement. In
this paper, we leverage the power of Large Language Models (LLMs) to create a
large, high-quality conversational dataset from a seed dataset. We propose a
Generator-Critic architecture framework to expand the initial dataset, while
improving the quality of its conversations. The Generator is an LLM prompted to
output conversations. The Critic consists of a mixture of expert LLMs that
control the quality of the generated conversations. These experts select the
best generated conversations, which we then use to improve the Generator. We
release Synthetic-Persona-Chat, consisting of 20k conversations seeded from
Persona-Chat. We evaluate the quality of Synthetic-Persona-Chat and our
generation framework on different dimensions through extensive experiments, and
observe that the losing rate of Synthetic-Persona-Chat against Persona-Chat
during Turing test decreases from 17.2% to 8.8% over three iterations.


7. [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](http://arxiv.org/abs/2312.12456v1), Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen, 16-12-2023
     ### Categories
     Machine Learning
    ### Abstract
    This paper introduces PowerInfer, a high-speed Large Language Model (LLM)
inference engine on a personal computer (PC) equipped with a single
consumer-grade GPU. The key underlying the design of PowerInfer is exploiting
the high locality inherent in LLM inference, characterized by a power-law
distribution in neuron activation. This distribution indicates that a small
subset of neurons, termed hot neurons, are consistently activated across
inputs, while the majority, cold neurons, vary based on specific inputs.
PowerInfer exploits such an insight to design a GPU-CPU hybrid inference
engine: hot-activated neurons are preloaded onto the GPU for fast access, while
cold-activated neurons are computed on the CPU, thus significantly reducing GPU
memory demands and CPU-GPU data transfers. PowerInfer further integrates
adaptive predictors and neuron-aware sparse operators, optimizing the
efficiency of neuron activation and computational sparsity. Evaluation shows
that PowerInfer attains an average token generation rate of 13.20 tokens/s,
with a peak of 29.08 tokens/s, across various LLMs (including OPT-175B) on a
single NVIDIA RTX 4090 GPU, only 18% lower than that achieved by a top-tier
server-grade A100 GPU. This significantly outperforms llama.cpp by up to 11.69x
while retaining model accuracy.


7. [A Survey of Reasoning with Foundation Models](http://arxiv.org/abs/2312.11562v4), Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, Yue Wu, Wenhai Wang, Junsong Chen, Zhangyue Yin, Xiaozhe Ren, Jie Fu, Junxian He, Wu Yuan, Qi Liu, Xihui Liu, Yu Li, Hao Dong, Yu Cheng, Ming Zhang, Pheng Ann Heng, Jifeng Dai, Ping Luo, Jingdong Wang, Ji-Rong Wen, Xipeng Qiu, Yike Guo, Hui Xiong, Qun Liu, Zhenguo Li, 17-12-2023
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Reasoning, a crucial ability for complex problem-solving, plays a pivotal
role in various real-world settings such as negotiation, medical diagnosis, and
criminal investigation. It serves as a fundamental methodology in the field of
Artificial General Intelligence (AGI). With the ongoing development of
foundation models, there is a growing interest in exploring their abilities in
reasoning tasks. In this paper, we introduce seminal foundation models proposed
or adaptable for reasoning, highlighting the latest advancements in various
reasoning tasks, methods, and benchmarks. We then delve into the potential
future directions behind the emergence of reasoning abilities within foundation
models. We also discuss the relevance of multimodal learning, autonomous
agents, and super alignment in the context of reasoning. By discussing these
future research directions, we hope to inspire researchers in their exploration
of this field, stimulate further advancements in reasoning with foundation
models, and contribute to the development of AGI.


7. [A Comprehensive Survey of Attack Techniques, Implementation, and
  Mitigation Strategies in Large Language Models](http://arxiv.org/abs/2312.10982v1), Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan, 18-12-2023
    ### Abstract
    Ensuring the security of large language models (LLMs) is an ongoing challenge
despite their widespread popularity. Developers work to enhance LLMs security,
but vulnerabilities persist, even in advanced versions like GPT-4. Attackers
exploit these weaknesses, highlighting the need for proactive cybersecurity
measures in AI model development. This article explores two attack categories:
attacks on models themselves and attacks on model applications. The former
requires expertise, access to model data, and significant implementation time,
while the latter is more accessible to attackers and has seen increased
attention. Our study reviews over 100 recent research works, providing an
in-depth analysis of each attack type. We identify the latest attack methods
and explore various approaches to carry them out. We thoroughly investigate
mitigation techniques, assessing their effectiveness and limitations.
Furthermore, we summarize future defenses against these attacks. We also
examine real-world techniques, including reported and our implemented attacks
on LLMs, to consolidate our findings. Our research highlights the urgency of
addressing security concerns and aims to enhance the understanding of LLM
attacks, contributing to robust defense development in this evolving domain.


7. [An In-depth Look at Gemini's Language Abilities](http://arxiv.org/abs/2312.11444v2), Syeda Nahida Akter, Zichun Yu, Aashiq Muhamed, Tianyue Ou, Alex Bäuerle, Ángel Alexander Cabrera, Krish Dholakia, Chenyan Xiong, Graham Neubig, 18-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    The recently released Google Gemini class of models are the first to
comprehensively report results that rival the OpenAI GPT series across a wide
variety of tasks. In this paper, we do an in-depth exploration of Gemini's
language abilities, making two contributions. First, we provide a third-party,
objective comparison of the abilities of the OpenAI GPT and Google Gemini
models with reproducible code and fully transparent results. Second, we take a
closer look at the results, identifying areas where one of the two model
classes excels. We perform this analysis over 10 datasets testing a variety of
language abilities, including reasoning, answering knowledge-based questions,
solving math problems, translating between languages, generating code, and
acting as instruction-following agents. From this analysis, we find that Gemini
Pro achieves accuracy that is close but slightly inferior to the corresponding
GPT 3.5 Turbo on all tasks that we benchmarked. We further provide explanations
for some of this under-performance, including failures in mathematical
reasoning with many digits, sensitivity to multiple-choice answer ordering,
aggressive content filtering, and others. We also identify areas where Gemini
demonstrates comparably high performance, including generation into non-English
languages, and handling longer and more complex reasoning chains. Code and data
for reproduction can be found at https://github.com/neulab/gemini-benchmark


7. [Gemini: A Family of Highly Capable Multimodal Models](http://arxiv.org/abs/2312.11805v1),  Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Martin Chadwick, Gaurav Singh Tomar, Xavier Garcia, Evan Senter, Emanuel Taropa, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Yujing Zhang, Ravi Addanki, Antoine Miech, Annie Louis, Laurent El Shafey, Denis Teplyashin, Geoff Brown, Elliot Catt, Nithya Attaluri, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming-Wei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozińska, Vitaly Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Patel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, Hanzhao Lin, James Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo-yiin Chang, Paul Komarek, Ross McIlroy, Mario Lučić, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yong Cheng, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Konstantinos Aisopos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çağlar Ünlü, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, YaGuang Li, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Moufarek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Rakićević, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Gamaleldin Elsayed, Ed Chi, Mahdis Mahdieh, Ian Tenney, Nan Hua, Ivan Petrychenko, Patrick Kane, Dylan Scandinaro, Rishub Jain, Jonathan Uesato, Romina Datta, Adam Sadovsky, Oskar Bunyan, Dominik Rabiej, Shimu Wu, John Zhang, Gautam Vasudevan, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Betty Chan, Pam G Rabinovitch, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Sahitya Potluri, Jane Park, Elnaz Davoodi, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Chris Gorgolewski, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Paul Suganthan, Evan Palmer, Geoffrey Irving, Edward Loper, Manaal Faruqui, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Michael Fink, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybiński, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marin Georgiev, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Minnie Lui, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Lam Nguyen Thiet, Daniel Andor, Pedro Valenzuela, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Sarmishta Velury, Sebastian Krause, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Tejasi Latkar, Mingyang Zhang, Quoc Le, Elena Allica Abellan, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Sid Lall, Ken Franko, Egor Filonov, Anna Bulanova, Rémi Leblond, Vikas Yadav, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Hao Zhou, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Jeremiah Liu, Mark Omernick, Colton Bishop, Chintu Kumar, Rachel Sterneck, Ryan Foley, Rohan Jain, Swaroop Mishra, Jiawei Xia, Taylor Bos, Geoffrey Cideron, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Petru Gurita, Hila Noga, Premal Shah, Daniel J. Mankowitz, Alex Polozov, Nate Kushman, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Anhad Mohananey, Matthieu Geist, Sidharth Mudgal, Sertan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christopher Yew, Quan Yuan, Sumit Bagri, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Aliaksei Severyn, Jonathan Lai, Kathy Wu, Heng-Tze Cheng, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Mark Geller, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Andrei Sozanschi, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Abhimanyu Goyal, Diane Wu, Denese Owusu-Afriyie, Cosmo Du, Chloe Thornton, Jordi Pont-Tuset, Pradyumna Narayana, Jing Li, Sabaer Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Tao Zhu, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Dustin Tran, Yeqing Li, Nir Levine, Ariel Stolovich, Norbert Kalb, Rebeca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Balaji Lakshminarayanan, Charlie Deck, Shyam Upadhyay, Hyo Lee, Mike Dusenberry, Zonglin Li, Xuezhi Wang, Kyle Levin, Raphael Hoffmann, Dan Holtmann-Rice, Olivier Bachem, Summer Yue, Sho Arora, Eric Malmi, Daniil Mirylenka, Qijun Tan, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Steven Zheng, Francesco Pongetti, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Ragha Kotikalapudi, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Chenkai Kuang, Vinod Koverkathu, Christopher A. Choquette-Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Pei Sun, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Ishita Dasgupta, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Yuan Liu, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Plucińska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Ivo Penchev, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Adam Kurzrok, Lynette Webb, Sahil Dua, Dong Li, Preethi Lahoti, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Taylan Bilal, Evgenii Eltyshev, Daniel Balle, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Adams Yu, Christof Angermueller, Xiaowei Li, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Kevin Brooks, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Komal Jalan, Dinghua Li, Ginger Perng, Blake Hechtman, Parker Schuh, Milad Nasr, Mia Chen, Kieran Milan, Vladimir Mikulik, Trevor Strohman, Juliana Franco, Tim Green, Demis Hassabis, Koray Kavukcuoglu, Jeffrey Dean, Oriol Vinyals, 19-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    This report introduces a new family of multimodal models, Gemini, that
exhibit remarkable capabilities across image, audio, video, and text
understanding. The Gemini family consists of Ultra, Pro, and Nano sizes,
suitable for applications ranging from complex reasoning tasks to on-device
memory-constrained use-cases. Evaluation on a broad range of benchmarks shows
that our most-capable Gemini Ultra model advances the state of the art in 30 of
32 of these benchmarks - notably being the first model to achieve human-expert
performance on the well-studied exam benchmark MMLU, and improving the state of
the art in every one of the 20 multimodal benchmarks we examined. We believe
that the new capabilities of Gemini models in cross-modal reasoning and
language understanding will enable a wide variety of use cases and we discuss
our approach toward deploying them responsibly to users.


7. [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models:
  A Critical Review and Assessment](http://arxiv.org/abs/2312.12148v1), Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang, 19-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    With the continuous growth in the number of parameters of transformer-based
pretrained language models (PLMs), particularly the emergence of large language
models (LLMs) with billions of parameters, many natural language processing
(NLP) tasks have demonstrated remarkable success. However, the enormous size
and computational demands of these models pose significant challenges for
adapting them to specific downstream tasks, especially in environments with
limited computational resources. Parameter Efficient Fine-Tuning (PEFT) offers
an effective solution by reducing the number of fine-tuning parameters and
memory usage while achieving comparable performance to full fine-tuning. The
demands for fine-tuning PLMs, especially LLMs, have led to a surge in the
development of PEFT methods, as depicted in Fig. 1. In this paper, we present a
comprehensive and systematic review of PEFT methods for PLMs. We summarize
these PEFT methods, discuss their applications, and outline future directions.
Furthermore, we conduct experiments using several representative PEFT methods
to better understand their effectiveness in parameter efficiency and memory
efficiency. By offering insights into the latest advancements and practical
applications, this survey serves as an invaluable resource for researchers and
practitioners seeking to navigate the challenges and opportunities presented by
PEFT in the context of PLMs.


7. [Generative Multimodal Models are In-Context Learners](http://arxiv.org/abs/2312.13286v1), Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, Xinlong Wang, 20-12-2023
    ### Abstract
    The human ability to easily solve multimodal tasks in context (i.e., with
only a few demonstrations or simple instructions), is what current multimodal
systems have largely struggled to imitate. In this work, we demonstrate that
the task-agnostic in-context learning capabilities of large multimodal models
can be significantly enhanced by effective scaling-up. We introduce Emu2, a
generative multimodal model with 37 billion parameters, trained on large-scale
multimodal sequences with a unified autoregressive objective. Emu2 exhibits
strong multimodal in-context learning abilities, even emerging to solve tasks
that require on-the-fly reasoning, such as visual prompting and object-grounded
generation. The model sets a new record on multiple multimodal understanding
tasks in few-shot settings. When instruction-tuned to follow specific
instructions, Emu2 further achieves new state-of-the-art on challenging tasks
such as question answering benchmarks for large multimodal models and
open-ended subject-driven generation. These achievements demonstrate that Emu2
can serve as a base model and general-purpose interface for a wide range of
multimodal tasks. Code and models are publicly available to facilitate future
research.


7. [Mini-GPTs: Efficient Large Language Models through Contextual Pruning](http://arxiv.org/abs/2312.12682v1), Tim Valicenti, Justice Vidal, Ritik Patnaik, 20-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In AI research, the optimization of Large Language Models (LLMs) remains a
significant challenge, crucial for advancing the field's practical applications
and sustainability. Building upon the foundational work of Professor Song Han's
lab at MIT, this paper introduces a novel approach in developing Mini-GPTs via
contextual pruning. Our methodology strategically prunes the computational
architecture of traditional LLMs, like Phi-1.5, focusing on retaining core
functionalities while drastically reducing model sizes. We employ the technique
across diverse and complex datasets, including US law, Medical Q&A, Skyrim
dialogue, English-Taiwanese translation, and Economics articles. The results
underscore the efficiency and effectiveness of contextual pruning, not merely
as a theoretical concept but as a practical tool in developing domain-specific,
resource-efficient LLMs. Contextual pruning is a promising method for building
domain-specific LLMs, and this research is a building block towards future
development with more hardware compute, refined fine-tuning, and quantization.


7. [Time is Encoded in the Weights of Finetuned Language Models](http://arxiv.org/abs/2312.13401v2), Kai Nylund, Suchin Gururangan, Noah A. Smith, 20-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    We present time vectors, a simple tool to customize language models to new
time periods. Time vectors are created by finetuning a language model on data
from a single time (e.g., a year or month), and then subtracting the weights of
the original pretrained model. This vector specifies a direction in weight
space that, as our experiments show, improves performance on text from that
time period. Time vectors specialized to adjacent time periods appear to be
positioned closer together in a manifold. Using this structure, we interpolate
between time vectors to induce new models that perform better on intervening
and future time periods, without any additional training. We demonstrate the
consistency of our findings across different tasks, domains, model sizes, and
time scales. Our results suggest that time is encoded in the weight space of
finetuned models.


7. [AppAgent: Multimodal Agents as Smartphone Users](http://arxiv.org/abs/2312.13771v2), Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, Gang Yu, 21-12-2023
    ### Abstract
    Recent advancements in large language models (LLMs) have led to the creation
of intelligent agents capable of performing complex tasks. This paper
introduces a novel LLM-based multimodal agent framework designed to operate
smartphone applications. Our framework enables the agent to operate smartphone
applications through a simplified action space, mimicking human-like
interactions such as tapping and swiping. This novel approach bypasses the need
for system back-end access, thereby broadening its applicability across diverse
apps. Central to our agent's functionality is its innovative learning method.
The agent learns to navigate and use new apps either through autonomous
exploration or by observing human demonstrations. This process generates a
knowledge base that the agent refers to for executing complex tasks across
different applications. To demonstrate the practicality of our agent, we
conducted extensive testing over 50 tasks in 10 different applications,
including social media, email, maps, shopping, and sophisticated image editing
tools. The results affirm our agent's proficiency in handling a diverse array
of high-level tasks.


7. [Exploring the intersection of Generative AI and Software Development](http://arxiv.org/abs/2312.14262v1), Filipe Calegario, Vanilson Burégio, Francisco Erivaldo, Daniel Moraes Costa Andrade, Kailane Felix, Nathalia Barbosa, Pedro Lucas da Silva Lucena, César França, 21-12-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    In the ever-evolving landscape of Artificial Intelligence (AI), the synergy
between generative AI and Software Engineering emerges as a transformative
frontier. This whitepaper delves into the unexplored realm, elucidating how
generative AI techniques can revolutionize software development. Spanning from
project management to support and updates, we meticulously map the demands of
each development stage and unveil the potential of generative AI in addressing
them. Techniques such as zero-shot prompting, self-consistency, and multimodal
chain-of-thought are explored, showcasing their unique capabilities in
enhancing generative AI models. The significance of vector embeddings, context,
plugins, tools, and code assistants is underscored, emphasizing their role in
capturing semantic information and amplifying generative AI capabilities.
Looking ahead, this intersection promises to elevate productivity, improve code
quality, and streamline the software development process. This whitepaper
serves as a guide for stakeholders, urging discussions and experiments in the
application of generative AI in Software Engineering, fostering innovation and
collaboration for a qualitative leap in the efficiency and effectiveness of
software development.


7. [LARP: Language-Agent Role Play for Open-World Games](http://arxiv.org/abs/2312.17653v1), Ming Yan, Ruihao Li, Hao Zhang, Hao Wang, Zhilan Yang, Ji Yan, 24-12-2023
     ### Categories
     Artificial Intelligence
    ### Abstract
    Language agents have shown impressive problem-solving skills within defined
settings and brief timelines. Yet, with the ever-evolving complexities of
open-world simulations, there's a pressing need for agents that can flexibly
adapt to complex environments and consistently maintain a long-term memory to
ensure coherent actions. To bridge the gap between language agents and
open-world games, we introduce Language Agent for Role-Playing (LARP), which
includes a cognitive architecture that encompasses memory processing and a
decision-making assistant, an environment interaction module with a
feedback-driven learnable action space, and a postprocessing method that
promotes the alignment of various personalities. The LARP framework refines
interactions between users and agents, predefined with unique backgrounds and
personalities, ultimately enhancing the gaming experience in open-world
contexts. Furthermore, it highlights the diverse uses of language models in a
range of areas such as entertainment, education, and various simulation
scenarios. The project page is released at https://miao-ai-lab.github.io/LARP/.


7. [Principled Instructions Are All You Need for Questioning LLaMA-1/2,
  GPT-3.5/4](http://arxiv.org/abs/2312.16171v1), Sondos Mahmoud Bsharat, Aidar Myrzakhan, Zhiqiang Shen, 26-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    This paper introduces 26 guiding principles designed to streamline the
process of querying and prompting large language models. Our goal is to
simplify the underlying concepts of formulating questions for various scales of
large language models, examining their abilities, and enhancing user
comprehension on the behaviors of different scales of large language models
when feeding into different prompts. Extensive experiments are conducted on
LLaMA-1/2 (7B, 13B and 70B), GPT-3.5/4 to verify the effectiveness of the
proposed principles on instructions and prompts design. We hope that this work
provides a better guide for researchers working on the prompting of large
language models. Project page is available at
https://github.com/VILA-Lab/ATLAS.


7. [Supervised Knowledge Makes Large Language Models Better In-context
  Learners](http://arxiv.org/abs/2312.15918v1), Linyi Yang, Shuibai Zhang, Zhuohao Yu, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, Yue Zhang, 26-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large Language Models (LLMs) exhibit emerging in-context learning abilities
through prompt engineering. The recent progress in large-scale generative
models has further expanded their use in real-world language applications.
However, the critical challenge of improving the generalizability and
factuality of LLMs in natural language understanding and question answering
remains under-explored. While previous in-context learning research has focused
on enhancing models to adhere to users' specific instructions and quality
expectations, and to avoid undesired outputs, little to no work has explored
the use of task-Specific fine-tuned Language Models (SLMs) to improve LLMs'
in-context learning during the inference stage. Our primary contribution is the
establishment of a simple yet effective framework that enhances the reliability
of LLMs as it: 1) generalizes out-of-distribution data, 2) elucidates how LLMs
benefit from discriminative models, and 3) minimizes hallucinations in
generative tasks. Using our proposed plug-in method, enhanced versions of Llama
2 and ChatGPT surpass their original versions regarding generalizability and
factuality. We offer a comprehensive suite of resources, including 16 curated
datasets, prompts, model checkpoints, and LLM outputs across 9 distinct tasks.
Our empirical analysis sheds light on the advantages of incorporating
discriminative models into LLMs and highlights the potential of our methodology
in fostering more reliable LLMs.


7. [Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil
  Cognitive Depth in LLMs](http://arxiv.org/abs/2312.17080v1), Zhongshen Zeng, Pengguang Chen, Haiyun Jiang, Jiaya Jia, 28-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    In this work, we introduce a novel evaluation paradigm for Large Language
Models, one that challenges them to engage in meta-reasoning. This approach
addresses critical shortcomings in existing math problem-solving benchmarks,
traditionally used to evaluate the cognitive capabilities of agents. Our
paradigm shifts the focus from result-oriented assessments, which often
overlook the reasoning process, to a more holistic evaluation that effectively
differentiates the cognitive capabilities among models. For example, in our
benchmark, GPT-4 demonstrates a performance ten times more accurate than
GPT3-5. The significance of this new paradigm lies in its ability to reveal
potential cognitive deficiencies in LLMs that current benchmarks, such as
GSM8K, fail to uncover due to their saturation and lack of effective
differentiation among varying reasoning abilities. Our comprehensive analysis
includes several state-of-the-art math models from both open-source and
closed-source communities, uncovering fundamental deficiencies in their
training and evaluation approaches. This paper not only advocates for a
paradigm shift in the assessment of LLMs but also contributes to the ongoing
discourse on the trajectory towards Artificial General Intelligence (AGI). By
promoting the adoption of meta-reasoning evaluation methods similar to ours, we
aim to facilitate a more accurate assessment of the true cognitive abilities of
LLMs.


7. [Experiential Co-Learning of Software-Developing Agents](http://arxiv.org/abs/2312.17025v2), Chen Qian, Yufan Dang, Jiahao Li, Wei Liu, Weize Chen, Cheng Yang, Zhiyuan Liu, Maosong Sun, 28-12-2023
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Recent advancements in large language models (LLMs) have brought significant
changes to various domains, especially through LLM-driven autonomous agents.
These agents are now capable of collaborating seamlessly, splitting tasks and
enhancing accuracy, thus minimizing the need for human involvement. However,
these agents often approach a diverse range of tasks in isolation, without
benefiting from past experiences. This isolation can lead to repeated mistakes
and inefficient trials in task solving. To this end, this paper introduces
Experiential Co-Learning, a novel framework in which instructor and assistant
agents gather shortcut-oriented experiences from their historical trajectories
and use these past experiences for mutual reasoning. This paradigm, enriched
with previous experiences, equips agents to more effectively address unseen
tasks.


7. [MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile
  Devices](http://arxiv.org/abs/2312.16886v2), Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen, 28-12-2023
    ### Abstract
    We present MobileVLM, a competent multimodal vision language model (MMVLM)
targeted to run on mobile devices. It is an amalgamation of a myriad of
architectural designs and techniques that are mobile-oriented, which comprises
a set of language models at the scale of 1.4B and 2.7B parameters, trained from
scratch, a multimodal vision model that is pre-trained in the CLIP fashion,
cross-modality interaction via an efficient projector. We evaluate MobileVLM on
several typical VLM benchmarks. Our models demonstrate on par performance
compared with a few much larger models. More importantly, we measure the
inference speed on both a Qualcomm Snapdragon 888 CPU and an NVIDIA Jeston Orin
GPU, and we obtain state-of-the-art performance of 21.5 tokens and 65.3 tokens
per second, respectively. Our code will be made available at:
https://github.com/Meituan-AutoML/MobileVLM.


7. [Large Language Models for Generative Information Extraction: A Survey](http://arxiv.org/abs/2312.17617v1), Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Enhong Chen, 29-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    Information extraction (IE) aims to extract structural knowledge (such as
entities, relations, and events) from plain natural language texts. Recently,
generative Large Language Models (LLMs) have demonstrated remarkable
capabilities in text understanding and generation, allowing for generalization
across various domains and tasks. As a result, numerous works have been
proposed to harness abilities of LLMs and offer viable solutions for IE tasks
based on a generative paradigm. To conduct a comprehensive systematic review
and exploration of LLM efforts for IE tasks, in this study, we survey the most
recent advancements in this field. We first present an extensive overview by
categorizing these works in terms of various IE subtasks and learning
paradigms, then we empirically analyze the most advanced methods and discover
the emerging trend of IE tasks with LLMs. Based on thorough review conducted,
we identify several insights in technique and promising research directions
that deserve further exploration in future studies. We maintain a public
repository and consistently update related resources at:
\url{https://github.com/quqxui/Awesome-LLM4IE-Papers}.


7. [Pushing Boundaries: Exploring Zero Shot Object Classification with Large
  Multimodal Models](http://arxiv.org/abs/2401.00127v1), Ashhadul Islam, Md. Rafiul Biswas, Wajdi Zaghouani, Samir Brahim Belhaouari, Zubair Shah, 30-12-2023
    ### Abstract
    $ $The synergy of language and vision models has given rise to Large Language
and Vision Assistant models (LLVAs), designed to engage users in rich
conversational experiences intertwined with image-based queries. These
comprehensive multimodal models seamlessly integrate vision encoders with Large
Language Models (LLMs), expanding their applications in general-purpose
language and visual comprehension. The advent of Large Multimodal Models (LMMs)
heralds a new era in Artificial Intelligence (AI) assistance, extending the
horizons of AI utilization. This paper takes a unique perspective on LMMs,
exploring their efficacy in performing image classification tasks using
tailored prompts designed for specific datasets. We also investigate the LLVAs
zero-shot learning capabilities. Our study includes a benchmarking analysis
across four diverse datasets: MNIST, Cats Vs. Dogs, Hymnoptera (Ants Vs. Bees),
and an unconventional dataset comprising Pox Vs. Non-Pox skin images. The
results of our experiments demonstrate the model's remarkable performance,
achieving classification accuracies of 85\%, 100\%, 77\%, and 79\% for the
respective datasets without any fine-tuning. To bolster our analysis, we assess
the model's performance post fine-tuning for specific tasks. In one instance,
fine-tuning is conducted over a dataset comprising images of faces of children
with and without autism. Prior to fine-tuning, the model demonstrated a test
accuracy of 55\%, which significantly improved to 83\% post fine-tuning. These
results, coupled with our prior findings, underscore the transformative
potential of LLVAs and their versatile applications in real-world scenarios.


7. [DocLLM: A layout-aware generative language model for multimodal document
  understanding](http://arxiv.org/abs/2401.00908v1), Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, Xiaomo Liu, 31-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    Enterprise documents such as forms, invoices, receipts, reports, contracts,
and other similar records, often carry rich semantics at the intersection of
textual and spatial modalities. The visual cues offered by their complex
layouts play a crucial role in comprehending these documents effectively. In
this paper, we present DocLLM, a lightweight extension to traditional large
language models (LLMs) for reasoning over visual documents, taking into account
both textual semantics and spatial layout. Our model differs from existing
multimodal LLMs by avoiding expensive image encoders and focuses exclusively on
bounding box information to incorporate the spatial layout structure.
Specifically, the cross-alignment between text and spatial modalities is
captured by decomposing the attention mechanism in classical transformers to a
set of disentangled matrices. Furthermore, we devise a pre-training objective
that learns to infill text segments. This approach allows us to address
irregular layouts and heterogeneous content frequently encountered in visual
documents. The pre-trained model is fine-tuned using a large-scale instruction
dataset, covering four core document intelligence tasks. We demonstrate that
our solution outperforms SotA LLMs on 14 out of 16 datasets across all tasks,
and generalizes well to 4 out of 5 previously unseen datasets.


7. [Improving Text Embeddings with Large Language Models](http://arxiv.org/abs/2401.00368v1), Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei, 31-12-2023
     ### Categories
     Computation and Language
    ### Abstract
    In this paper, we introduce a novel and simple method for obtaining
high-quality text embeddings using only synthetic data and less than 1k
training steps. Unlike existing methods that often depend on multi-stage
intermediate pre-training with billions of weakly-supervised text pairs,
followed by fine-tuning with a few labeled datasets, our method does not
require building complex training pipelines or relying on manually collected
datasets that are often constrained by task diversity and language coverage. We
leverage proprietary LLMs to generate diverse synthetic data for hundreds of
thousands of text embedding tasks across nearly 100 languages. We then
fine-tune open-source decoder-only LLMs on the synthetic data using standard
contrastive loss. Experiments demonstrate that our method achieves strong
performance on highly competitive text embedding benchmarks without using any
labeled data. Furthermore, when fine-tuned with a mixture of synthetic and
labeled data, our model sets new state-of-the-art results on the BEIR and MTEB
benchmarks.


7. [Opening A Pandora's Box: Things You Should Know in the Era of Custom
  GPTs](http://arxiv.org/abs/2401.00905v1), Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang, 31-12-2023
    ### Abstract
    The emergence of large language models (LLMs) has significantly accelerated
the development of a wide range of applications across various fields. There is
a growing trend in the construction of specialized platforms based on LLMs,
such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide
various functionalities like web browsing and code execution, they also
introduce significant security threats. In this paper, we conduct a
comprehensive analysis of the security and privacy issues arising from the
custom GPT platform. Our systematic examination categorizes potential attack
scenarios into three threat models based on the role of the malicious actor,
and identifies critical data exchange channels in custom GPTs. Utilizing the
STRIDE threat modeling framework, we identify 26 potential attack vectors, with
19 being partially or fully validated in real-world settings. Our findings
emphasize the urgent need for robust security and privacy measures in the
custom GPT ecosystem, especially in light of the forthcoming launch of the
official GPT store by OpenAI.
## 2024 (33 papers)



8. [A Computational Framework for Behavioral Assessment of LLM Therapists](http://arxiv.org/abs/2401.00820v1), Yu Ying Chiu, Ashish Sharma, Inna Wanyin Lin, Tim Althoff, 01-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    The emergence of ChatGPT and other large language models (LLMs) has greatly
increased interest in utilizing LLMs as therapists to support individuals
struggling with mental health challenges. However, due to the lack of
systematic studies, our understanding of how LLM therapists behave, i.e., ways
in which they respond to clients, is significantly limited. Understanding their
behavior across a wide range of clients and situations is crucial to accurately
assess their capabilities and limitations in the high-risk setting of mental
health, where undesirable behaviors can lead to severe consequences. In this
paper, we propose BOLT, a novel computational framework to study the
conversational behavior of LLMs when employed as therapists. We develop an
in-context learning method to quantitatively measure the behavior of LLMs based
on 13 different psychotherapy techniques including reflections, questions,
solutions, normalizing, and psychoeducation. Subsequently, we compare the
behavior of LLM therapists against that of high- and low-quality human therapy,
and study how their behavior can be modulated to better reflect behaviors
observed in high-quality therapy. Our analysis of GPT and Llama-variants
reveals that these LLMs often resemble behaviors more commonly exhibited in
low-quality therapy rather than high-quality therapy, such as offering a higher
degree of problem-solving advice when clients share emotions, which is against
typical recommendations. At the same time, unlike low-quality therapy, LLMs
reflect significantly more upon clients' needs and strengths. Our analysis
framework suggests that despite the ability of LLMs to generate anecdotal
examples that appear similar to human therapists, LLM therapists are currently
not fully consistent with high-quality care, and thus require additional
research to ensure quality care.


8. [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large
  Language Models](http://arxiv.org/abs/2401.00625v2), Guangji Bai, Zheng Chai, Chen Ling, Shiyu Wang, Jiaying Lu, Nan Zhang, Tingwei Shi, Ziyang Yu, Mengdan Zhu, Yifei Zhang, Carl Yang, Yue Cheng, Liang Zhao, 01-01-2024
     ### Categories
     Machine Learning
    ### Abstract
    The burgeoning field of Large Language Models (LLMs), exemplified by
sophisticated models like OpenAI's ChatGPT, represents a significant
advancement in artificial intelligence. These models, however, bring forth
substantial challenges in the high consumption of computational, memory,
energy, and financial resources, especially in environments with limited
resource capabilities. This survey aims to systematically address these
challenges by reviewing a broad spectrum of techniques designed to enhance the
resource efficiency of LLMs. We categorize methods based on their optimization
focus: computational, memory, energy, financial, and network resources and
their applicability across various stages of an LLM's lifecycle, including
architecture design, pretraining, finetuning, and system design. Additionally,
the survey introduces a nuanced categorization of resource efficiency
techniques by their specific resource types, which uncovers the intricate
relationships and mappings between various resources and corresponding
optimization techniques. A standardized set of evaluation metrics and datasets
is also presented to facilitate consistent and fair comparisons across
different models and techniques. By offering a comprehensive overview of the
current sota and identifying open research avenues, this survey serves as a
foundational reference for researchers and practitioners, aiding them in
developing more sustainable and efficient LLMs in a rapidly evolving landscape.


8. [General-purpose foundation models for increased autonomy in
  robot-assisted surgery](http://arxiv.org/abs/2401.00678v1), Samuel Schmidgall, Ji Woong Kim, Alan Kuntz, Ahmed Ezzat Ghazi, Axel Krieger, 01-01-2024
     ### Categories
     Machine Learning
    ### Abstract
    The dominant paradigm for end-to-end robot learning focuses on optimizing
task-specific objectives that solve a single robotic problem such as picking up
an object or reaching a target position. However, recent work on high-capacity
models in robotics has shown promise toward being trained on large collections
of diverse and task-agnostic datasets of video demonstrations. These models
have shown impressive levels of generalization to unseen circumstances,
especially as the amount of data and the model complexity scale. Surgical robot
systems that learn from data have struggled to advance as quickly as other
fields of robot learning for a few reasons: (1) there is a lack of existing
large-scale open-source data to train models, (2) it is challenging to model
the soft-body deformations that these robots work with during surgery because
simulation cannot match the physical and visual complexity of biological
tissue, and (3) surgical robots risk harming patients when tested in clinical
trials and require more extensive safety measures. This perspective article
aims to provide a path toward increasing robot autonomy in robot-assisted
surgery through the development of a multi-modal, multi-task,
vision-language-action model for surgical robots. Ultimately, we argue that
surgical robots are uniquely positioned to benefit from general-purpose models
and provide three guiding actions toward increased autonomy in robot-assisted
surgery.


8. [If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code
  Empowers Large Language Models to Serve as Intelligent Agents](http://arxiv.org/abs/2401.00812v2), Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R. Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, Heng Ji, Chengxiang Zhai, 01-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    The prominent large language models (LLMs) of today differ from past language
models not only in size, but also in the fact that they are trained on a
combination of natural language and formal language (code). As a medium between
humans and computers, code translates high-level goals into executable steps,
featuring standard syntax, logical consistency, abstraction, and modularity. In
this survey, we present an overview of the various benefits of integrating code
into LLMs' training data. Specifically, beyond enhancing LLMs in code
generation, we observe that these unique properties of code help (i) unlock the
reasoning ability of LLMs, enabling their applications to a range of more
complex natural language tasks; (ii) steer LLMs to produce structured and
precise intermediate steps, which can then be connected to external execution
ends through function calls; and (iii) take advantage of code compilation and
execution environment, which also provides diverse feedback for model
improvement. In addition, we trace how these profound capabilities of LLMs,
brought by code, have led to their emergence as intelligent agents (IAs) in
situations where the ability to understand instructions, decompose goals, plan
and execute actions, and refine from feedback are crucial to their success on
downstream tasks. Finally, we present several key challenges and future
directions of empowering LLMs with code.


8. [The Earth is Flat? Unveiling Factual Errors in Large Language Models](http://arxiv.org/abs/2401.00761v1), Wenxuan Wang, Juluan Shi, Zhaopeng Tu, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, Michael R. Lyu, 01-01-2024
     ### Categories
     Artificial Intelligence, Computation and Language
    ### Abstract
    Large Language Models (LLMs) like ChatGPT are foundational in various
applications due to their extensive knowledge from pre-training and
fine-tuning. Despite this, they are prone to generating factual and commonsense
errors, raising concerns in critical areas like healthcare, journalism, and
education to mislead users. Current methods for evaluating LLMs' veracity are
limited by test data leakage or the need for extensive human labor, hindering
efficient and accurate error detection. To tackle this problem, we introduce a
novel, automatic testing framework, FactChecker, aimed at uncovering factual
inaccuracies in LLMs. This framework involves three main steps: First, it
constructs a factual knowledge graph by retrieving fact triplets from a
large-scale knowledge database. Then, leveraging the knowledge graph,
FactChecker employs a rule-based approach to generates three types of questions
(Yes-No, Multiple-Choice, and WH questions) that involve single-hop and
multi-hop relations, along with correct answers. Lastly, it assesses the LLMs'
responses for accuracy using tailored matching strategies for each question
type. Our extensive tests on six prominent LLMs, including text-davinci-002,
text-davinci-003, ChatGPT~(gpt-3.5-turbo, gpt-4), Vicuna, and LLaMA-2, reveal
that FactChecker can trigger factual errors in up to 45\% of questions in these
models. Moreover, we demonstrate that FactChecker's test cases can improve
LLMs' factual accuracy through in-context learning and fine-tuning (e.g.,
llama-2-13b-chat's accuracy increase from 35.3\% to 68.5\%). We are making all
code, data, and results available for future research endeavors.


8. [A Comprehensive Study of Knowledge Editing for Large Language Models](http://arxiv.org/abs/2401.01286v3), Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen, 02-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    Large Language Models (LLMs) have shown extraordinary capabilities in
understanding and generating text that closely mirrors human communication.
However, a primary limitation lies in the significant computational demands
during training, arising from their extensive parameterization. This challenge
is further intensified by the dynamic nature of the world, necessitating
frequent updates to LLMs to correct outdated information or integrate new
knowledge, thereby ensuring their continued relevance. Note that many
applications demand continual model adjustments post-training to address
deficiencies or undesirable behaviors. There is an increasing interest in
efficient, lightweight methods for on-the-fly model modifications. To this end,
recent years have seen a burgeoning in the techniques of knowledge editing for
LLMs, which aim to efficiently modify LLMs' behaviors within specific domains
while preserving overall performance across various inputs. In this paper, we
first define the knowledge editing problem and then provide a comprehensive
review of cutting-edge approaches. Drawing inspiration from educational and
cognitive research theories, we propose a unified categorization criterion that
classifies knowledge editing methods into three groups: resorting to external
knowledge, merging knowledge into the model, and editing intrinsic knowledge.
Furthermore, we introduce a new benchmark, KnowEdit, for a comprehensive
empirical evaluation of representative knowledge editing approaches.
Additionally, we provide an in-depth analysis of knowledge location, which can
give a deeper understanding of the knowledge structures inherent within LLMs.
Finally, we discuss several potential applications of knowledge editing,
outlining its broad and impactful implications.


8. [A Comprehensive Survey of Hallucination Mitigation Techniques in Large
  Language Models](http://arxiv.org/abs/2401.01313v3), S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, Amitava Das, 02-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    As Large Language Models (LLMs) continue to advance in their ability to write
human-like text, a key challenge remains around their tendency to hallucinate
generating content that appears factual but is ungrounded. This issue of
hallucination is arguably the biggest hindrance to safely deploying these
powerful LLMs into real-world production systems that impact people's lives.
The journey toward widespread adoption of LLMs in practical settings heavily
relies on addressing and mitigating hallucinations. Unlike traditional AI
systems focused on limited tasks, LLMs have been exposed to vast amounts of
online text data during training. While this allows them to display impressive
language fluency, it also means they are capable of extrapolating information
from the biases in training data, misinterpreting ambiguous prompts, or
modifying the information to align superficially with the input. This becomes
hugely alarming when we rely on language generation capabilities for sensitive
applications, such as summarizing medical records, financial analysis reports,
etc. This paper presents a comprehensive survey of over 32 techniques developed
to mitigate hallucination in LLMs. Notable among these are Retrieval Augmented
Generation (Lewis et al, 2021), Knowledge Retrieval (Varshney et al,2023),
CoNLI (Lei et al, 2023), and CoVe (Dhuliawala et al, 2023). Furthermore, we
introduce a detailed taxonomy categorizing these methods based on various
parameters, such as dataset utilization, common tasks, feedback mechanisms, and
retriever types. This classification helps distinguish the diverse approaches
specifically designed to tackle hallucination issues in LLMs. Additionally, we
analyze the challenges and limitations inherent in these techniques, providing
a solid foundation for future research in addressing hallucinations and related
phenomena within the realm of LLMs.


8. [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](http://arxiv.org/abs/2401.01325v1), Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, Xia Hu, 02-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    This work elicits LLMs' inherent ability to handle long contexts without
fine-tuning. The limited length of the training sequence during training may
limit the application of Large Language Models (LLMs) on long input sequences
for inference. In this work, we argue that existing LLMs themselves have
inherent capabilities for handling long contexts. Based on this argument, we
suggest extending LLMs' context window by themselves to fully utilize the
inherent ability.We propose Self-Extend to stimulate LLMs' long context
handling potential. The basic idea is to construct bi-level attention
information: the group level and the neighbor level. The two levels are
computed by the original model's self-attention, which means the proposed does
not require any training. With only four lines of code modification, the
proposed method can effortlessly extend existing LLMs' context window without
any fine-tuning. We conduct comprehensive experiments and the results show that
the proposed method can effectively extend existing LLMs' context window's
length.


8. [LLaMA Beyond English: An Empirical Study on Language Capability Transfer](http://arxiv.org/abs/2401.01055v2), Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, Xuanjing Huang, 02-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In recent times, substantial advancements have been witnessed in large
language models (LLMs), exemplified by ChatGPT, showcasing remarkable
proficiency across a range of complex tasks. However, many mainstream LLMs
(e.g. LLaMA) are pretrained on English-dominant corpus, which limits their
performance in other non-English languages. In this paper, we focus on how to
effectively transfer the capabilities of language generation and following
instructions to a non-English language. To answer this question, we conduct an
extensive empirical investigation based on LLaMA, accumulating over 1440 GPU
hours. We analyze the impact of key factors such as vocabulary extension,
further pretraining, and instruction tuning on transfer. To accurately assess
the model's level of knowledge, we employ four widely used standardized testing
benchmarks: C-Eval, MMLU, AGI-Eval, and GAOKAO-Bench. Furthermore, a
comprehensive evaluation of the model's response quality is conducted,
considering aspects such as accuracy, fluency, informativeness, logical
coherence, and harmlessness, based on LLM-Eval, a benchmarks consisting
instruction tasks from 17 diverse categories. Our evaluation results
demonstrate that comparable performance to state-of-the-art transfer models can
be achieved with less than 1% of the pretraining data, both in terms of
knowledge alignment and response quality. Furthermore, the experimental
outcomes across the thirteen low-resource languages also exhibit similar
trends. We anticipate that the conclusions revealed by the experiments will aid
the community in developing non-English LLMs.


8. [Enhancing the medical foundation model with multi-scale and
  cross-modality feature learning](http://arxiv.org/abs/2401.01583v1), Weijian Huang, Cheng Li, Hong-Yu Zhou, Jiarun Liu, Hao Yang, Yong Liang, Shanshan Wang, 03-01-2024
    ### Abstract
    The development of multi-modal medical foundation models has attracted
significant attention in the field of medicine and healthcare due to their
promising prospects in various clinical applications. One area of focus in this
research direction is the extractions of features at different scales. While
previous studies have explored feature learning at individual scales,
investigation on integrating the diverse scales and modalities of information
is lacking, which may hinder the potential for mutual reinforcement among these
features. This paper aims to bridge this gap by proposing a method that
effectively exploits multi-scale and cross-modality information to enhance the
performance of medical foundation models. The proposed method simultaneously
exploit features at the local, instance, modality and global aspects,
facilitating comprehensive representation learning within the models. We
evaluate the effectiveness of the proposed method on six open-source datasets
across different clinical tasks, demonstrating its ability to enhance the
performance of medical foundation models.


8. [Exploring the Frontiers of LLMs in Psychological Applications: A
  Comprehensive Review](http://arxiv.org/abs/2401.01519v2), Luoma Ke, Song Tong, Peng Cheng, Kaiping Peng, 03-01-2024
     ### Categories
     Machine Learning, Artificial Intelligence
    ### Abstract
    This paper explores the frontiers of large language models (LLMs) in
psychology applications. Psychology has undergone several theoretical changes,
and the current use of Artificial Intelligence (AI) and Machine Learning,
particularly LLMs, promises to open up new research directions. We provide a
detailed exploration of how LLMs like ChatGPT are transforming psychological
research. It discusses the impact of LLMs across various branches of
psychology, including cognitive and behavioral, clinical and counseling,
educational and developmental, and social and cultural psychology, highlighting
their potential to simulate aspects of human cognition and behavior. The paper
delves into the capabilities of these models to emulate human-like text
generation, offering innovative tools for literature review, hypothesis
generation, experimental design, experimental subjects, data analysis, academic
writing, and peer review in psychology. While LLMs are essential in advancing
research methodologies in psychology, the paper also cautions about their
technical and ethical challenges. There are issues like data privacy, the
ethical implications of using LLMs in psychological research, and the need for
a deeper understanding of these models' limitations. Researchers should
responsibly use LLMs in psychological studies, adhering to ethical standards
and considering the potential consequences of deploying these technologies in
sensitive areas. Overall, the article provides a comprehensive overview of the
current state of LLMs in psychology, exploring potential benefits and
challenges. It serves as a call to action for researchers to leverage LLMs'
advantages responsibly while addressing associated risks.


8. [Few-shot Adaptation of Multi-modal Foundation Models: A Survey](http://arxiv.org/abs/2401.01736v2), Fan Liu, Tianshu Zhang, Wenwen Dai, Wenwen Cai, Xiaocong Zhou, Delong Chen, 03-01-2024
    ### Abstract
    Multi-modal (vision-language) models, such as CLIP, are replacing traditional
supervised pre-training models (e.g., ImageNet-based pre-training) as the new
generation of visual foundation models. These models with robust and aligned
semantic representations learned from billions of internet image-text pairs and
can be applied to various downstream tasks in a zero-shot manner. However, in
some fine-grained domains like medical imaging and remote sensing, the
performance of multi-modal foundation models often leaves much to be desired.
Consequently, many researchers have begun to explore few-shot adaptation
methods for these models, gradually deriving three main technical approaches:
1) prompt-based methods, 2) adapter-based methods, and 3) external
knowledge-based methods. Nevertheless, this rapidly developing field has
produced numerous results without a comprehensive survey to systematically
organize the research progress. Therefore, in this survey, we introduce and
analyze the research advancements in few-shot adaptation methods for
multi-modal models, summarizing commonly used datasets and experimental setups,
and comparing the results of different methods. In addition, due to the lack of
reliable theoretical support for existing methods, we derive the few-shot
adaptation generalization error bound for multi-modal models. The theorem
reveals that the generalization error of multi-modal foundation models is
constrained by three factors: domain gap, model capacity, and sample size.
Based on this, we propose three possible solutions from the following aspects:
1) adaptive domain generalization, 2) adaptive model selection, and 3) adaptive
knowledge utilization.


8. [Large Language Models Relearn Removed Concepts](http://arxiv.org/abs/2401.01814v1), Michelle Lo, Shay B. Cohen, Fazl Barez, 03-01-2024
     ### Categories
     Artificial Intelligence
    ### Abstract
    Advances in model editing through neuron pruning hold promise for removing
undesirable concepts from large language models. However, it remains unclear
whether models have the capacity to reacquire pruned concepts after editing. To
investigate this, we evaluate concept relearning in models by tracking concept
saliency and similarity in pruned neurons during retraining. Our findings
reveal that models can quickly regain performance post-pruning by relocating
advanced concepts to earlier layers and reallocating pruned concepts to primed
neurons with similar semantics. This demonstrates that models exhibit
polysemantic capacities and can blend old and new concepts in individual
neurons. While neuron pruning provides interpretability into model concepts,
our results highlight the challenges of permanent concept removal for improved
model \textit{safety}. Monitoring concept reemergence and developing techniques
to mitigate relearning of unsafe concepts will be important directions for more
robust model editing. Overall, our work strongly demonstrates the resilience
and fluidity of concept representations in LLMs post concept removal.


8. [Correctness Comparison of ChatGPT-4, Bard, Claude-2, and Copilot for
  Spatial Tasks](http://arxiv.org/abs/2401.02404v2), Hartwig H. Hochmair, Levente Juhasz, Takoda Kemp, 04-01-2024
    ### Abstract
    Generative AI including large language models (LLMs) have recently gained
significant interest in the geo-science community through its versatile
task-solving capabilities including coding, spatial computations, generation of
sample data, time-series forecasting, toponym recognition, or image
classification. So far, the assessment of LLMs for spatial tasks has primarily
focused on ChatGPT, arguably the most prominent AI chatbot, whereas other
chatbots received less attention. To narrow this research gap, this study
evaluates the correctness of responses for a set of 54 spatial tasks assigned
to four prominent chatbots, i.e., ChatGPT-4, Bard, Claude-2, and Copilot.
Overall, the chatbots performed well on spatial literacy, GIS theory, and
interpretation of programming code and given functions, but revealed weaknesses
in mapping, code generation, and code translation. ChatGPT-4 outperformed other
chatbots across most task categories.


8. [LLM Augmented LLMs: Expanding Capabilities through Composition](http://arxiv.org/abs/2401.02412v1), Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, Partha Talukdar, 04-01-2024
     ### Categories
     Machine Learning, Artificial Intelligence, Computation and Language
    ### Abstract
    Foundational models with billions of parameters which have been trained on
large corpora of data have demonstrated non-trivial skills in a variety of
domains. However, due to their monolithic structure, it is challenging and
expensive to augment them or impart new skills. On the other hand, due to their
adaptation abilities, several new instances of these models are being trained
towards new domains and tasks. In this work, we study the problem of efficient
and practical composition of existing foundation models with more specific
models to enable newer capabilities. To this end, we propose CALM --
Composition to Augment Language Models -- which introduces cross-attention
between models to compose their representations and enable new capabilities.
Salient features of CALM are: (i) Scales up LLMs on new tasks by 're-using'
existing LLMs along with a few additional parameters and data, (ii) Existing
model weights are kept intact, and hence preserves existing capabilities, and
(iii) Applies to diverse domains and settings. We illustrate that augmenting
PaLM2-S with a smaller model trained on low-resource languages results in an
absolute improvement of up to 13\% on tasks like translation into English and
arithmetic reasoning for low-resource languages. Similarly, when PaLM2-S is
augmented with a code-specific model, we see a relative improvement of 40\%
over the base model for code generation and explanation tasks -- on-par with
fully fine-tuned counterparts.


8. [LLaMA Pro: Progressive LLaMA with Block Expansion](http://arxiv.org/abs/2401.02415v1), Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, Ying Shan, 04-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    Humans generally acquire new skills without compromising the old; however,
the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to
CodeLLaMA. To this end, we propose a new post-pretraining method for LLMs with
an expansion of Transformer blocks. We tune the expanded blocks using only new
corpus, efficiently and effectively improving the model's knowledge without
catastrophic forgetting. In this paper, we experiment on the corpus of code and
math, yielding LLaMA Pro-8.3B, a versatile foundation model initialized from
LLaMA2-7B, excelling in general tasks, programming, and mathematics. LLaMA Pro
and its instruction-following counterpart (LLaMA Pro-Instruct) achieve advanced
performance among various benchmarks, demonstrating superiority over existing
open models in the LLaMA family and the immense potential of reasoning and
addressing diverse tasks as an intelligent agent. Our findings provide valuable
insights into integrating natural and programming languages, laying a solid
foundation for developing advanced language agents that operate effectively in
various environments.


8. [LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model](http://arxiv.org/abs/2401.02330v2), Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, Jian Tang, 04-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    In this paper, we introduce LLaVA-$\phi$ (LLaVA-Phi), an efficient
multi-modal assistant that harnesses the power of the recently advanced small
language model, Phi-2, to facilitate multi-modal dialogues. LLaVA-Phi marks a
notable advancement in the realm of compact multi-modal models. It demonstrates
that even smaller language models, with as few as 2.7B parameters, can
effectively engage in intricate dialogues that integrate both textual and
visual elements, provided they are trained with high-quality corpora. Our model
delivers commendable performance on publicly available benchmarks that
encompass visual comprehension, reasoning, and knowledge-based perception.
Beyond its remarkable performance in multi-modal dialogue tasks, our model
opens new avenues for applications in time-sensitive environments and systems
that require real-time interaction, such as embodied agents. It highlights the
potential of smaller language models to achieve sophisticated levels of
understanding and interaction, while maintaining greater resource
efficiency.The project is available at {https://github.com/zhuyiche/llava-phi}.


8. [TinyLlama: An Open-Source Small Language Model](http://arxiv.org/abs/2401.02385v1), Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu, 04-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    We present TinyLlama, a compact 1.1B language model pretrained on around 1
trillion tokens for approximately 3 epochs. Building on the architecture and
tokenizer of Llama 2, TinyLlama leverages various advances contributed by the
open-source community (e.g., FlashAttention), achieving better computational
efficiency. Despite its relatively small size, TinyLlama demonstrates
remarkable performance in a series of downstream tasks. It significantly
outperforms existing open-source language models with comparable sizes. Our
model checkpoints and code are publicly available on GitHub at
https://github.com/jzhang38/TinyLlama.


8. [Understanding LLMs: A Comprehensive Overview from Training to Inference](http://arxiv.org/abs/2401.02038v2), Yiheng Liu, Hao He, Tianle Han, Xu Zhang, Mengyuan Liu, Jiaming Tian, Yutong Zhang, Jiaqi Wang, Xiaohui Gao, Tianyang Zhong, Yi Pan, Shaochen Xu, Zihao Wu, Zhengliang Liu, Xin Zhang, Shu Zhang, Xintao Hu, Tuo Zhang, Ning Qiang, Tianming Liu, Bao Ge, 04-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    The introduction of ChatGPT has led to a significant increase in the
utilization of Large Language Models (LLMs) for addressing downstream tasks.
There's an increasing focus on cost-efficient training and deployment within
this context. Low-cost training and deployment of LLMs represent the future
development trend. This paper reviews the evolution of large language model
training techniques and inference deployment technologies aligned with this
emerging trend. The discussion on training includes various aspects, including
data preprocessing, training architecture, pre-training tasks, parallel
training, and relevant content related to model fine-tuning. On the inference
side, the paper covers topics such as model compression, parallel computation,
memory scheduling, and structural optimization. It also explores LLMs'
utilization and provides insights into their future development.


8. [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](http://arxiv.org/abs/2401.02954v1),  DeepSeek-AI,  :, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, Yuheng Zou, 05-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence, Machine Learning
    ### Abstract
    The rapid development of open-source large language models (LLMs) has been
truly remarkable. However, the scaling law described in previous literature
presents varying conclusions, which casts a dark cloud over scaling LLMs. We
delve into the study of scaling laws and present our distinctive findings that
facilitate scaling of large scale models in two commonly used open-source
configurations, 7B and 67B. Guided by the scaling laws, we introduce DeepSeek
LLM, a project dedicated to advancing open-source language models with a
long-term perspective. To support the pre-training phase, we have developed a
dataset that currently consists of 2 trillion tokens and is continuously
expanding. We further conduct supervised fine-tuning (SFT) and Direct
Preference Optimization (DPO) on DeepSeek LLM Base models, resulting in the
creation of DeepSeek Chat models. Our evaluation results demonstrate that
DeepSeek LLM 67B surpasses LLaMA-2 70B on various benchmarks, particularly in
the domains of code, mathematics, and reasoning. Furthermore, open-ended
evaluations reveal that DeepSeek LLM 67B Chat exhibits superior performance
compared to GPT-3.5.


8. [From LLM to Conversational Agent: A Memory Enhanced Architecture with
  Fine-Tuning of Large Language Models](http://arxiv.org/abs/2401.02777v1), Na Liu, Liangyu Chen, Xiaoyu Tian, Wei Zou, Kaijiang Chen, Ming Cui, 05-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    This paper introduces RAISE (Reasoning and Acting through Scratchpad and
Examples), an advanced architecture enhancing the integration of Large Language
Models (LLMs) like GPT-4 into conversational agents. RAISE, an enhancement of
the ReAct framework, incorporates a dual-component memory system, mirroring
human short-term and long-term memory, to maintain context and continuity in
conversations. It entails a comprehensive agent construction scenario,
including phases like Conversation Selection, Scene Extraction, CoT Completion,
and Scene Augmentation, leading to the LLMs Training phase. This approach
appears to enhance agent controllability and adaptability in complex,
multi-turn dialogues. Our preliminary evaluations in a real estate sales
context suggest that RAISE has some advantages over traditional agents,
indicating its potential for broader applications. This work contributes to the
AI field by providing a robust framework for developing more context-aware and
versatile conversational agents.


8. [Thousands of AI Authors on the Future of AI](http://arxiv.org/abs/2401.02843v1), Katja Grace, Harlan Stewart, Julia Fabienne Sandkühler, Stephen Thomas, Ben Weinstein-Raun, Jan Brauner, 05-01-2024
     ### Categories
     Artificial Intelligence, Machine Learning
    ### Abstract
    In the largest survey of its kind, 2,778 researchers who had published in
top-tier artificial intelligence (AI) venues gave predictions on the pace of AI
progress and the nature and impacts of advanced AI systems The aggregate
forecasts give at least a 50% chance of AI systems achieving several milestones
by 2028, including autonomously constructing a payment processing site from
scratch, creating a song indistinguishable from a new song by a popular
musician, and autonomously downloading and fine-tuning a large language model.
If science continues undisrupted, the chance of unaided machines outperforming
humans in every possible task was estimated at 10% by 2027, and 50% by 2047.
The latter estimate is 13 years earlier than that reached in a similar survey
we conducted only one year earlier [Grace et al., 2022]. However, the chance of
all human occupations becoming fully automatable was forecast to reach 10% by
2037, and 50% as late as 2116 (compared to 2164 in the 2022 survey).
  Most respondents expressed substantial uncertainty about the long-term value
of AI progress: While 68.3% thought good outcomes from superhuman AI are more
likely than bad, of these net optimists 48% gave at least a 5% chance of
extremely bad outcomes such as human extinction, and 59% of net pessimists gave
5% or more to extremely good outcomes. Between 38% and 51% of respondents gave
at least a 10% chance to advanced AI leading to outcomes as bad as human
extinction. More than half suggested that "substantial" or "extreme" concern is
warranted about six different AI-related scenarios, including misinformation,
authoritarian control, and inequality. There was disagreement about whether
faster or slower AI progress would be better for the future of humanity.
However, there was broad agreement that research aimed at minimizing potential
risks from AI systems ought to be prioritized more.


8. [Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon](http://arxiv.org/abs/2401.03462v1), Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, Zhicheng Dou, 07-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    The utilization of long contexts poses a big challenge for large language
models due to their limited context window length. Although the context window
can be extended through fine-tuning, it will result in a considerable cost at
both training and inference time, and exert an unfavorable impact to the LLM's
original capabilities. In this work, we propose Activation Beacon, which
condenses LLM's raw activations into more compact forms such that it can
perceive a much longer context with a limited context window. Activation Beacon
is introduced as a plug-and-play module for the LLM. It fully preserves the
LLM's original capability on short contexts while extending the new capability
on processing longer contexts. Besides, it works with short sliding windows to
process the long context, which achieves a competitive memory and time
efficiency in both training and inference. Activation Beacon is learned by the
auto-regression task conditioned on a mixture of beacons with diversified
condensing ratios. Thanks to such a treatment, it can be efficiently trained
purely with short-sequence data in just 10K steps, which consumes less than 9
hours on a single 8xA800 GPU machine. The experimental studies show that
Activation Beacon is able to extend Llama-2-7B's context length by $\times100$
times (from 4K to 400K), meanwhile achieving a superior result on both
long-context generation and understanding tasks. Our model and code will be
available at the BGE repository.


8. [Mixtral of Experts](http://arxiv.org/abs/2401.04088v1), Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed, 08-01-2024
     ### Categories
     Machine Learning, Computation and Language
    ### Abstract
    We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model.
Mixtral has the same architecture as Mistral 7B, with the difference that each
layer is composed of 8 feedforward blocks (i.e. experts). For every token, at
each layer, a router network selects two experts to process the current state
and combine their outputs. Even though each token only sees two experts, the
selected experts can be different at each timestep. As a result, each token has
access to 47B parameters, but only uses 13B active parameters during inference.
Mixtral was trained with a context size of 32k tokens and it outperforms or
matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular,
Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and
multilingual benchmarks. We also provide a model fine-tuned to follow
instructions, Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo,
Claude-2.1, Gemini Pro, and Llama 2 70B - chat model on human benchmarks. Both
the base and instruct models are released under the Apache 2.0 license.


8. [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table
  Understanding](http://arxiv.org/abs/2401.04398v1), Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister, 09-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    Table-based reasoning with large language models (LLMs) is a promising
direction to tackle many table understanding tasks, such as table-based
question answering and fact verification. Compared with generic reasoning,
table-based reasoning requires the extraction of underlying semantics from both
free-form questions and semi-structured tabular data. Chain-of-Thought and its
similar approaches incorporate the reasoning chain in the form of textual
context, but it is still an open question how to effectively leverage tabular
data in the reasoning chain. We propose the Chain-of-Table framework, where
tabular data is explicitly used in the reasoning chain as a proxy for
intermediate thoughts. Specifically, we guide LLMs using in-context learning to
iteratively generate operations and update the table to represent a tabular
reasoning chain. LLMs can therefore dynamically plan the next operation based
on the results of the previous ones. This continuous evolution of the table
forms a chain, showing the reasoning process for a given tabular problem. The
chain carries structured information of the intermediate results, enabling more
accurate and reliable predictions. Chain-of-Table achieves new state-of-the-art
performance on WikiTQ, FeTaQA, and TabFact benchmarks across multiple LLM
choices.


8. [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety
  Training](http://arxiv.org/abs/2401.05566v2), Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, Ethan Perez, 10-01-2024
     ### Categories
     Artificial Intelligence, Computation and Language, Machine Learning
    ### Abstract
    Humans are capable of strategically deceptive behavior: behaving helpfully in
most situations, but then behaving very differently in order to pursue
alternative objectives when given the opportunity. If an AI system learned such
a deceptive strategy, could we detect it and remove it using current
state-of-the-art safety training techniques? To study this question, we
construct proof-of-concept examples of deceptive behavior in large language
models (LLMs). For example, we train models that write secure code when the
prompt states that the year is 2023, but insert exploitable code when the
stated year is 2024. We find that such backdoor behavior can be made
persistent, so that it is not removed by standard safety training techniques,
including supervised fine-tuning, reinforcement learning, and adversarial
training (eliciting unsafe behavior and then training to remove it). The
backdoor behavior is most persistent in the largest models and in models
trained to produce chain-of-thought reasoning about deceiving the training
process, with the persistence remaining even when the chain-of-thought is
distilled away. Furthermore, rather than removing backdoors, we find that
adversarial training can teach models to better recognize their backdoor
triggers, effectively hiding the unsafe behavior. Our results suggest that,
once a model exhibits deceptive behavior, standard techniques could fail to
remove such deception and create a false impression of safety.


8. [The Impact of Reasoning Step Length on Large Language Models](http://arxiv.org/abs/2401.04925v2), Mingyu Jin, Qinkai Yu, Dong shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, Mengnan Du, 10-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Chain of Thought (CoT) is significant in improving the reasoning abilities of
large language models (LLMs). However, the correlation between the
effectiveness of CoT and the length of reasoning steps in prompts remains
largely unknown. To shed light on this, we have conducted several empirical
experiments to explore the relations. Specifically, we design experiments that
expand and compress the rationale reasoning steps within CoT demonstrations,
while keeping all other factors constant. We have the following key findings.
First, the results indicate that lengthening the reasoning steps in prompts,
even without adding new information into the prompt, considerably enhances
LLMs' reasoning abilities across multiple datasets. Alternatively, shortening
the reasoning steps, even while preserving the key information, significantly
diminishes the reasoning abilities of models. This finding highlights the
importance of the number of steps in CoT prompts and provides practical
guidance to make better use of LLMs' potential in complex problem-solving
scenarios. Second, we also investigated the relationship between the
performance of CoT and the rationales used in demonstrations. Surprisingly, the
result shows that even incorrect rationales can yield favorable outcomes if
they maintain the requisite length of inference. Third, we observed that the
advantages of increasing reasoning steps are task-dependent: simpler tasks
require fewer steps, whereas complex tasks gain significantly from longer
inference sequences.


8. [TrustLLM: Trustworthiness in Large Language Models](http://arxiv.org/abs/2401.05561v2), Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, Willian Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, Yue Zhao, 10-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    Large language models (LLMs), exemplified by ChatGPT, have gained
considerable attention for their excellent natural language processing
capabilities. Nonetheless, these LLMs present many challenges, particularly in
the realm of trustworthiness. Therefore, ensuring the trustworthiness of LLMs
emerges as an important topic. This paper introduces TrustLLM, a comprehensive
study of trustworthiness in LLMs, including principles for different dimensions
of trustworthiness, established benchmark, evaluation, and analysis of
trustworthiness for mainstream LLMs, and discussion of open challenges and
future directions. Specifically, we first propose a set of principles for
trustworthy LLMs that span eight different dimensions. Based on these
principles, we further establish a benchmark across six dimensions including
truthfulness, safety, fairness, robustness, privacy, and machine ethics. We
then present a study evaluating 16 mainstream LLMs in TrustLLM, consisting of
over 30 datasets. Our findings firstly show that in general trustworthiness and
utility (i.e., functional effectiveness) are positively related. Secondly, our
observations reveal that proprietary LLMs generally outperform most open-source
counterparts in terms of trustworthiness, raising concerns about the potential
risks of widely accessible open-source LLMs. However, a few open-source LLMs
come very close to proprietary ones. Thirdly, it is important to note that some
LLMs may be overly calibrated towards exhibiting trustworthiness, to the extent
that they compromise their utility by mistakenly treating benign prompts as
harmful and consequently not responding. Finally, we emphasize the importance
of ensuring transparency not only in the models themselves but also in the
technologies that underpin trustworthiness. Knowing the specific trustworthy
technologies that have been employed is crucial for analyzing their
effectiveness.


8. [Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language
  Model Systems](http://arxiv.org/abs/2401.05778v1), Tianyu Cui, Yanling Wang, Chuanpu Fu, Yong Xiao, Sijia Li, Xinhao Deng, Yunpeng Liu, Qinglin Zhang, Ziyi Qiu, Peiyang Li, Zhixing Tan, Junwu Xiong, Xinyu Kong, Zujie Wen, Ke Xu, Qi Li, 11-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Large language models (LLMs) have strong capabilities in solving diverse
natural language processing tasks. However, the safety and security issues of
LLM systems have become the major obstacle to their widespread application.
Many studies have extensively investigated risks in LLM systems and developed
the corresponding mitigation strategies. Leading-edge enterprises such as
OpenAI, Google, Meta, and Anthropic have also made lots of efforts on
responsible LLMs. Therefore, there is a growing need to organize the existing
studies and establish comprehensive taxonomies for the community. In this
paper, we delve into four essential modules of an LLM system, including an
input module for receiving prompts, a language model trained on extensive
corpora, a toolchain module for development and deployment, and an output
module for exporting LLM-generated content. Based on this, we propose a
comprehensive taxonomy, which systematically analyzes potential risks
associated with each module of an LLM system and discusses the corresponding
mitigation strategies. Furthermore, we review prevalent benchmarks, aiming to
facilitate the risk assessment of LLM systems. We hope that this paper can help
LLM participants embrace a systematic perspective to build their responsible
LLM systems.


8. [Seven Failure Points When Engineering a Retrieval Augmented Generation
  System](http://arxiv.org/abs/2401.05856v1), Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek, 11-01-2024
    ### Abstract
    Software engineers are increasingly adding semantic search capabilities to
applications using a strategy known as Retrieval Augmented Generation (RAG). A
RAG system involves finding documents that semantically match a query and then
passing the documents to a large language model (LLM) such as ChatGPT to
extract the right answer using an LLM. RAG systems aim to: a) reduce the
problem of hallucinated responses from LLMs, b) link sources/references to
generated responses, and c) remove the need for annotating documents with
meta-data. However, RAG systems suffer from limitations inherent to information
retrieval systems and from reliance on LLMs. In this paper, we present an
experience report on the failure points of RAG systems from three case studies
from separate domains: research, education, and biomedical. We share the
lessons learned and present 7 failure points to consider when designing a RAG
system. The two key takeaways arising from our work are: 1) validation of a RAG
system is only feasible during operation, and 2) the robustness of a RAG system
evolves rather than designed in at the start. We conclude with a list of
potential research directions on RAG systems for the software engineering
community.


8. [The Benefits of a Concise Chain of Thought on Problem-Solving in Large
  Language Models](http://arxiv.org/abs/2401.05618v1), Matthew Renze, Erhan Guven, 11-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    In this paper, we introduce Concise Chain-of-Thought (CCoT) prompting. We
compared standard CoT and CCoT prompts to see how conciseness impacts response
length and correct-answer accuracy. We evaluated this using GPT-3.5 and GPT-4
with a multiple-choice question-and-answer (MCQA) benchmark. CCoT reduced
average response length by 48.70% for both GPT-3.5 and GPT-4 while having a
negligible impact on problem-solving performance. However, on math problems,
GPT-3.5 with CCoT incurs a performance penalty of 27.69%. Overall, CCoT leads
to an average per-token cost reduction of 22.67%. These results have practical
implications for AI systems engineers using LLMs to solve real-world problems
with CoT prompt-engineering techniques. In addition, these results provide more
general insight for AI researchers studying the emergent behavior of
step-by-step reasoning in LLMs.


8. [How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to
  Challenge AI Safety by Humanizing LLMs](http://arxiv.org/abs/2401.06373v1), Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi, 12-01-2024
     ### Categories
     Computation and Language, Artificial Intelligence
    ### Abstract
    Most traditional AI safety research has approached AI models as machines and
centered on algorithm-focused attacks developed by security experts. As large
language models (LLMs) become increasingly common and competent, non-expert
users can also impose risks during daily interactions. This paper introduces a
new perspective to jailbreak LLMs as human-like communicators, to explore this
overlooked intersection between everyday language interaction and AI safety.
Specifically, we study how to persuade LLMs to jailbreak them. First, we
propose a persuasion taxonomy derived from decades of social science research.
Then, we apply the taxonomy to automatically generate interpretable persuasive
adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion
significantly increases the jailbreak performance across all risk categories:
PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b
Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused
attacks. On the defense side, we explore various mechanisms against PAP and,
found a significant gap in existing defenses, and advocate for more fundamental
mitigation for highly interactive LLMs


8. [Intention Analysis Prompting Makes Large Language Models A Good
  Jailbreak Defender](http://arxiv.org/abs/2401.06561v1), Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao, 12-01-2024
     ### Categories
     Computation and Language
    ### Abstract
    Aligning large language models (LLMs) with human values, particularly in the
face of stealthy and complex jailbreaks, presents a formidable challenge. In
this study, we present a simple yet highly effective defense strategy, i.e.,
Intention Analysis Prompting (IAPrompt). The principle behind is to trigger
LLMs' inherent self-correct and improve ability through a two-stage process: 1)
essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt
is an inference-only method, thus could enhance the safety of LLMs without
compromising their helpfulness. Extensive experiments on SAP200 and DAN
benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that
IAPrompt could consistently and significantly reduce the harmfulness in
response (averagely -46.5% attack success rate) and maintain the general
helpfulness. Further analyses present some insights into how our method works.
To facilitate reproducibility, We release our code and scripts at:
https://github.com/alphadl/SafeLLM_with_IntentionAnalysis
