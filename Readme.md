A curated list of papers on Large Language Models by year. I'll try to update the list if new papers are published. Let me know if I am missing important papers or there are errors.
### To-do: 
* Sort by publication date.
* Add keywords/categories
## 2017
1. [**Attention is All You Need**](https://arxiv.org/pdf/1706.03762.pdf) by Chull Hwan Song, Hye Joo Han, Yannis Avrithis

	### Summary

	The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

----


## 2019
1. [**Generating Long Sequences with Sparse Transformers**](http://arxiv.org/pdf/1904.10509v1) by Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever

	### Summary

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

----

2. [**Attention Sorting Combats Recency Bias In Long Context Language Models**](http://arxiv.org/pdf/2310.01427v1) by Alexander Peysakhovich, Adam Lerer

	### Summary

	Current language models often fail to incorporate long contexts efficiently
during generation. We show that a major contributor to this issue are attention
priors that are likely learned during pre-training: relevant information
located earlier in context is attended to less on average. Yet even when models
fail to use the information from a relevant document in their response, they
still pay preferential attention to that document compared to an irrelevant
document at the same position. We leverage this fact to introduce ``attention
sorting'': perform one step of decoding, sort documents by the attention they
receive (highest attention going last), repeat the process, generate the answer
with the newly sorted context. We find that attention sorting improves
performance of long context models. Our findings highlight some challenges in
using off-the-shelf language models for retrieval augmented generation.

----

3. [**Multitasking Framework for Unsupervised Simple Definition Generation**](http://arxiv.org/pdf/2203.12926v1) by Cunliang Kong, Yun Chen, Hengyuan Zhang, Liner Yang, Erhong Yang

	### Summary

	The definition generation task can help language learners by providing
explanations for unfamiliar words. This task has attracted much attention in
recent years. We propose a novel task of Simple Definition Generation (SDG) to
help language learners and low literacy readers. A significant challenge of
this task is the lack of learner's dictionaries in many languages, and
therefore the lack of data for supervised training. We explore this task and
propose a multitasking framework SimpDefiner that only requires a standard
dictionary with complex definitions and a corpus containing arbitrary simple
texts. We disentangle the complexity factors from the text by carefully
designing a parameter sharing scheme between two decoders. By jointly training
these components, the framework can generate both complex and simple
definitions simultaneously. We demonstrate that the framework can generate
relevant, simple definitions for the target words through automatic and manual
evaluations on English and Chinese datasets. Our method outperforms the
baseline model by a 1.77 SARI score on the English dataset, and raises the
proportion of the low level (HSK level 1-3) words in Chinese definitions by
3.87%.

----

4. [**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](http://arxiv.org/pdf/1910.10683v4) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

	### Summary

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

----

5. [**T3L: Translate-and-Test Transfer Learning for Cross-Lingual Text Classification**](http://arxiv.org/pdf/2306.04996v1) by Inigo Jauregi Unanue, Gholamreza Haffari, Massimo Piccardi

	### Summary

	Cross-lingual text classification leverages text classifiers trained in a
high-resource language to perform text classification in other languages with
no or minimal fine-tuning (zero/few-shots cross-lingual transfer). Nowadays,
cross-lingual text classifiers are typically built on large-scale, multilingual
language models (LMs) pretrained on a variety of languages of interest.
However, the performance of these models vary significantly across languages
and classification tasks, suggesting that the superposition of the language
modelling and classification tasks is not always effective. For this reason, in
this paper we propose revisiting the classic "translate-and-test" pipeline to
neatly separate the translation and classification stages. The proposed
approach couples 1) a neural machine translator translating from the targeted
language to a high-resource language, with 2) a text classifier trained in the
high-resource language, but the neural machine translator generates "soft"
translations to permit end-to-end backpropagation during fine-tuning of the
pipeline. Extensive experiments have been carried out over three cross-lingual
text classification datasets (XNLI, MLDoc and MultiEURLEX), with the results
showing that the proposed approach has significantly improved performance over
a competitive baseline.

----

6. [**CoLAKE: Contextualized Language and Knowledge Embedding**](http://arxiv.org/pdf/2010.00309v1) by Tianxiang Sun, Yunfan Shao, Xipeng Qiu, Qipeng Guo, Yaru Hu, Xuanjing Huang, Zheng Zhang

	### Summary

	With the emerging branch of incorporating factual knowledge into pre-trained
language models such as BERT, most existing models consider shallow, static,
and separately pre-trained entity embeddings, which limits the performance
gains of these models. Few works explore the potential of deep contextualized
knowledge representation when injecting knowledge. In this paper, we propose
the Contextualized Language and Knowledge Embedding (CoLAKE), which jointly
learns contextualized representation for both language and knowledge with the
extended MLM objective. Instead of injecting only entity embeddings, CoLAKE
extracts the knowledge context of an entity from large-scale knowledge bases.
To handle the heterogeneity of knowledge context and language context, we
integrate them in a unified data structure, word-knowledge graph (WK graph).
CoLAKE is pre-trained on large-scale WK graphs with the modified Transformer
encoder. We conduct experiments on knowledge-driven tasks, knowledge probing
tasks, and language understanding tasks. Experimental results show that CoLAKE
outperforms previous counterparts on most of the tasks. Besides, CoLAKE
achieves surprisingly high performance on our synthetic task called
word-knowledge graph completion, which shows the superiority of simultaneously
contextualizing language and knowledge representation.

----

7. [**BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**](http://arxiv.org/pdf/1910.13461v1) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer

	### Summary

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

----

8. [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](http://arxiv.org/pdf/1810.04805v2) by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

	### Summary

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

----


## 2020
1. [**AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts**](http://arxiv.org/pdf/2010.15980v2) by Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh

	### Summary

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

----

2. [**Longformer: The Long-Document Transformer**](http://arxiv.org/pdf/2004.05150v2) by Iz Beltagy, Matthew E. Peters, Arman Cohan

	### Summary

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

----

3. [**Measuring Value Understanding in Language Models through Discriminator-Critique Gap**](http://arxiv.org/pdf/2310.00378v3) by Zhaowei Zhang, Fengshuo Bai, Jun Gao, Yaodong Yang

	### Summary

	Recent advancements in Large Language Models (LLMs) have heightened concerns
about their potential misalignment with human values. However, evaluating their
grasp of these values is complex due to their intricate and adaptable nature.
We argue that truly understanding values in LLMs requires considering both
"know what" and "know why". To this end, we present the Value Understanding
Measurement (VUM) framework that quantitatively assesses both "know what" and
"know why" by measuring the discriminator-critique gap related to human values.
Using the Schwartz Value Survey, we specify our evaluation values and develop a
thousand-level dialogue dataset with GPT-4. Our assessment looks at both the
value alignment of LLM's outputs compared to baseline answers and how LLM
responses align with reasons for value recognition versus GPT-4's annotations.
We evaluate five representative LLMs and provide strong evidence that the
scaling law significantly impacts "know what" but not much on "know why", which
has consistently maintained a high level. This may further suggest that LLMs
might craft plausible explanations based on the provided context without truly
understanding their inherent value, indicating potential risks.

----

4. [**Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification**](http://arxiv.org/pdf/2010.13641v1) by Timo Schick, Helmut Schmid, Hinrich Schütze

	### Summary

	A recent approach for few-shot text classification is to convert textual
inputs to cloze questions that contain some form of task description, process
them with a pretrained language model and map the predicted words to labels.
Manually defining this mapping between words and labels requires both domain
expertise and an understanding of the language model's abilities. To mitigate
this issue, we devise an approach that automatically finds such a mapping given
small amounts of training data. For a number of tasks, the mapping found by our
approach performs almost as well as hand-crafted label-to-word mappings.

----

5. [**Evaluating Computational Language Models with Scaling Properties of Natural Language**](http://arxiv.org/pdf/1906.09379v1) by Shuntaro Takahashi, Kumiko Tanaka-Ishii

	### Summary

	In this article, we evaluate computational models of natural language with
respect to the universal statistical behaviors of natural language. Statistical
mechanical analyses have revealed that natural language text is characterized
by scaling properties, which quantify the global structure in the vocabulary
population and the long memory of a text. We study whether five scaling
properties (given by Zipf's law, Heaps' law, Ebeling's method, Taylor's law,
and long-range correlation analysis) can serve for evaluation of computational
models. Specifically, we test $n$-gram language models, a probabilistic
context-free grammar (PCFG), language models based on Simon/Pitman-Yor
processes, neural language models, and generative adversarial networks (GANs)
for text generation. Our analysis reveals that language models based on
recurrent neural networks (RNNs) with a gating mechanism (i.e., long short-term
memory, LSTM; a gated recurrent unit, GRU; and quasi-recurrent neural networks,
QRNNs) are the only computational models that can reproduce the long memory
behavior of natural language. Furthermore, through comparison with recently
proposed model-based evaluation methods, we find that the exponent of Taylor's
law is a good indicator of model quality.

----

6. [**LMTurk: Few-Shot Learners as Crowdsourcing Workers in a Language-Model-as-a-Service Framework**](http://arxiv.org/pdf/2112.07522v2) by Mengjie Zhao, Fei Mi, Yasheng Wang, Minglei Li, Xin Jiang, Qun Liu, Hinrich Schütze

	### Summary

	Vast efforts have been devoted to creating high-performance few-shot
learners, i.e., large-scale pretrained language models (PLMs) that perform well
with little downstream task training data. Training PLMs has incurred
significant cost, but utilizing the few-shot learners is still challenging due
to their enormous size. This work focuses on a crucial question: How to make
effective use of these few-shot learners? We propose LMTurk, a novel approach
that treats few-shot learners as crowdsourcing workers. The rationale is that
crowdsourcing workers are in fact few-shot learners: They are shown a few
illustrative examples to learn about a task and then start annotating. LMTurk
employs few-shot learners built upon PLMs as workers. We show that the
resulting annotations can be utilized to train models that solve the task well
and are small enough to be deployable in practical scenarios. Active learning
is integrated into LMTurk to reduce the amount of queries made to PLMs,
minimizing the computational cost of running PLM inference passes. Altogether,
LMTurk is an important step towards making effective use of current PLMs.

----

7. [**An axiomatic limitation on the deterministic scope required for superdeterminism and its consequentially greater likelihood**](http://arxiv.org/pdf/2311.15470v1) by Cameron Shackell

	### Summary

	By positing a universe where all events are determined by initial conditions,
superdeterminism as conceded by Bell frames correlations observed in quantum
measurements as the consequence of an inherently predetermined cosmic order
that shapes even our experimental choices. I use an axiomatic formulation of
superdeterminism to demonstrate that Bell overstated the scope of determinism
required. Assuming only the existence of a universe containing observers, I
show that determinism in just the observer scope is sufficient. I then discuss
how this sufficiency increases the theory's plausibility and suggest a path to
its integration with results from other disciplines.

----

8. [**Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**](http://arxiv.org/pdf/1909.08053v4) by Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro

	### Summary

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

----

9. [**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**](http://arxiv.org/pdf/2004.12832v2) by Omar Khattab, Matei Zaharia

	### Summary

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

----

10. [**Datasheet for the Pile**](http://arxiv.org/pdf/2201.07311v1) by Stella Biderman, Kieran Bicheno, Leo Gao

	### Summary

	This datasheet describes the Pile, a 825 GiB dataset of human-authored text
compiled by EleutherAI for use in large-scale language modeling. The Pile is
comprised of 22 different text sources, ranging from original scrapes done for
this project, to text data made available by the data owners, to third-party
scrapes available online.

----


## 2021
1. [**Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners**](http://arxiv.org/pdf/2108.13161v7) by Ningyu Zhang, Luoqiu Li, Xiang Chen, Shumin Deng, Zhen Bi, Chuanqi Tan, Fei Huang, Huajun Chen

	### Summary

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

----

2. [**Imagination-Augmented Natural Language Understanding**](http://arxiv.org/pdf/2204.08535v3) by Yujie Lu, Wanrong Zhu, Xin Eric Wang, Miguel Eckstein, William Yang Wang

	### Summary

	Human brains integrate linguistic and perceptual information simultaneously
to understand natural language, and hold the critical ability to render
imaginations. Such abilities enable us to construct new abstract concepts or
concrete objects, and are essential in involving practical knowledge to solve
problems in low-resource scenarios. However, most existing methods for Natural
Language Understanding (NLU) are mainly focused on textual signals. They do not
simulate human visual imagination ability, which hinders models from inferring
and learning efficiently from limited data samples. Therefore, we introduce an
Imagination-Augmented Cross-modal Encoder (iACE) to solve natural language
understanding tasks from a novel learning perspective -- imagination-augmented
cross-modal understanding. iACE enables visual imagination with external
knowledge transferred from the powerful generative and pre-trained
vision-and-language models. Extensive experiments on GLUE and SWAG show that
iACE achieves consistent improvement over visually-supervised pre-trained
models. More importantly, results in extreme and normal few-shot settings
validate the effectiveness of iACE in low-resource natural language
understanding circumstances.

----

3. [**Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training**](http://arxiv.org/pdf/1810.11895v3) by Hila Gonen, Yoav Goldberg

	### Summary

	We focus on the problem of language modeling for code-switched language, in
the context of automatic speech recognition (ASR). Language modeling for
code-switched language is challenging for (at least) three reasons: (1) lack of
available large-scale code-switched data for training; (2) lack of a replicable
evaluation setup that is ASR directed yet isolates language modeling
performance from the other intricacies of the ASR system; and (3) the reliance
on generative modeling. We tackle these three issues: we propose an
ASR-motivated evaluation setup which is decoupled from an ASR system and the
choice of vocabulary, and provide an evaluation dataset for English-Spanish
code-switching. This setup lends itself to a discriminative training approach,
which we demonstrate to work better than generative language modeling. Finally,
we explore a variety of training protocols and verify the effectiveness of
training with large amounts of monolingual data followed by fine-tuning with
small amounts of code-switched data, for both the generative and discriminative
cases.

----

4. [**Can Language Models be Biomedical Knowledge Bases?**](http://arxiv.org/pdf/2109.07154v1) by Mujeen Sung, Jinhyuk Lee, Sean Yi, Minji Jeon, Sungdong Kim, Jaewoo Kang

	### Summary

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

----

5. [**A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models**](http://arxiv.org/pdf/2307.12980v1) by Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, Philip Torr

	### Summary

	Prompt engineering is a technique that involves augmenting a large
pre-trained model with task-specific hints, known as prompts, to adapt the
model to new tasks. Prompts can be created manually as natural language
instructions or generated automatically as either natural language instructions
or vector representations. Prompt engineering enables the ability to perform
predictions based solely on prompts without updating model parameters, and the
easier application of large pre-trained models in real-world tasks. In past
years, Prompt engineering has been well-studied in natural language processing.
Recently, it has also been intensively studied in vision-language modeling.
However, there is currently a lack of a systematic overview of prompt
engineering on pre-trained vision-language models. This paper aims to provide a
comprehensive survey of cutting-edge research in prompt engineering on three
types of vision-language models: multimodal-to-text generation models (e.g.
Flamingo), image-text matching models (e.g. CLIP), and text-to-image generation
models (e.g. Stable Diffusion). For each type of model, a brief model summary,
prompting methods, prompting-based applications, and the corresponding
responsibility and integrity issues are summarized and discussed. Furthermore,
the commonalities and differences between prompting on vision-language models,
language models, and vision models are also discussed. The challenges, future
directions, and research opportunities are summarized to foster future research
on this topic.

----

6. [**CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP**](http://arxiv.org/pdf/2104.08835v2) by Qinyuan Ye, Bill Yuchen Lin, Xiang Ren

	### Summary

	Humans can learn a new language task efficiently with only few examples, by
leveraging their knowledge obtained when learning prior tasks. In this paper,
we explore whether and how such cross-task generalization ability can be
acquired, and further applied to build better few-shot learners across diverse
NLP tasks. We introduce CrossFit, a problem setup for studying cross-task
generalization ability, which standardizes seen/unseen task partitions, data
access during different learning stages, and the evaluation protocols. To
instantiate different seen/unseen task partitions in CrossFit and facilitate
in-depth analysis, we present the NLP Few-shot Gym, a repository of 160 diverse
few-shot NLP tasks created from open-access NLP datasets and converted to a
unified text-to-text format. Our analysis reveals that the few-shot learning
ability on unseen tasks can be improved via an upstream learning stage using a
set of seen tasks. We also observe that the selection of upstream learning
tasks can significantly influence few-shot performance on unseen tasks, asking
further analysis on task similarity and transferability.

----

7. [**PTR: Prompt Tuning with Rules for Text Classification**](http://arxiv.org/pdf/2105.11259v3) by Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, Maosong Sun

	### Summary

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

----

8. [**Learning How to Ask: Querying LMs with Mixtures of Soft Prompts**](http://arxiv.org/pdf/2104.06599v1) by Guanghui Qin, Jason Eisner

	### Summary

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

----

9. [**Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm**](http://arxiv.org/pdf/2102.07350v1) by Laria Reynolds, Kyle McDonell

	### Summary

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

----

10. [**How Many Data Points is a Prompt Worth?**](http://arxiv.org/pdf/2103.08493v2) by Teven Le Scao, Alexander M. Rush

	### Summary

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

----

11. [**Meta-X$_{NLG}$: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation**](http://arxiv.org/pdf/2203.10250v1) by Kaushal Kumar Maurya, Maunendra Sankar Desarkar

	### Summary

	Recently, the NLP community has witnessed a rapid advancement in multilingual
and cross-lingual transfer research where the supervision is transferred from
high-resource languages (HRLs) to low-resource languages (LRLs). However, the
cross-lingual transfer is not uniform across languages, particularly in the
zero-shot setting. Towards this goal, one promising research direction is to
learn shareable structures across multiple tasks with limited annotated data.
The downstream multilingual applications may benefit from such a learning setup
as most of the languages across the globe are low-resource and share some
structures with other languages. In this paper, we propose a novel
meta-learning framework (called Meta-X$_{NLG}$) to learn shareable structures
from typologically diverse languages based on meta-learning and language
clustering. This is a step towards uniform cross-lingual transfer for unseen
languages. We first cluster the languages based on language representations and
identify the centroid language of each cluster. Then, a meta-learning algorithm
is trained with all centroid languages and evaluated on the other languages in
the zero-shot setting. We demonstrate the effectiveness of this modeling on two
NLG tasks (Abstractive Text Summarization and Question Generation), 5 popular
datasets and 30 typologically diverse languages. Consistent improvements over
strong baselines demonstrate the efficacy of the proposed framework. The
careful design of the model makes this end-to-end NLG setup less vulnerable to
the accidental translation problem, which is a prominent concern in zero-shot
cross-lingual NLG tasks.

----

12. [**Open Aspect Target Sentiment Classification with Natural Language Prompts**](http://arxiv.org/pdf/2109.03685v1) by Ronald Seoh, Ian Birle, Mrinal Tak, Haw-Shiuan Chang, Brian Pinette, Alfred Hough

	### Summary

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

----

13. [**General-Purpose Question-Answering with Macaw**](http://arxiv.org/pdf/2109.02593v1) by Oyvind Tafjord, Peter Clark

	### Summary

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

----

14. [**Arithmetic-Based Pretraining -- Improving Numeracy of Pretrained Language Models**](http://arxiv.org/pdf/2205.06733v2) by Dominic Petrak, Nafise Sadat Moosavi, Iryna Gurevych

	### Summary

	State-of-the-art pretrained language models tend to perform below their
capabilities when applied out-of-the-box on tasks that require understanding
and working with numbers. Recent work suggests two main reasons for this: (1)
popular tokenisation algorithms have limited expressiveness for numbers, and
(2) common pretraining objectives do not target numeracy. Approaches that
address these shortcomings usually require architectural changes or pretraining
from scratch. In this paper, we propose a new extended pretraining approach
called Arithmetic-Based Pretraining that jointly addresses both in one extended
pretraining step without requiring architectural changes or pretraining from
scratch. Arithmetic-Based Pretraining combines contrastive learning to improve
the number representation, and a novel extended pretraining objective called
Inferable Number Prediction Task to improve numeracy. Our experiments show the
effectiveness of Arithmetic-Based Pretraining in three different tasks that
require improved numeracy, i.e., reading comprehension in the DROP dataset,
inference-on-tables in the InfoTabs dataset, and table-to-text generation in
the WikiBio and SciGen datasets.

----

15. [**FLIN: A Flexible Natural Language Interface for Web Navigation**](http://arxiv.org/pdf/2010.12844v2) by Sahisnu Mazumder, Oriana Riva

	### Summary

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

----

16. [**Prefix-Tuning: Optimizing Continuous Prompts for Generation**](http://arxiv.org/pdf/2101.00190v1) by Xiang Lisa Li, Percy Liang

	### Summary

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

----

17. [**Want To Reduce Labeling Cost? GPT-3 Can Help**](http://arxiv.org/pdf/2108.13487v1) by Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, Michael Zeng

	### Summary

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

----

18. [**SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis**](http://arxiv.org/pdf/2109.08306v1) by Chengxi Li, Feiyu Gao, Jiajun Bu, Lu Xu, Xiang Chen, Yu Gu, Zirui Shao, Qi Zheng, Ningyu Zhang, Yongpan Wang, Zhi Yu

	### Summary

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

----

19. [**A Modular Task-oriented Dialogue System Using a Neural Mixture-of-Experts**](http://arxiv.org/pdf/1907.05346v1) by Jiahuan Pei, Pengjie Ren, Maarten de Rijke

	### Summary

	End-to-end Task-oriented Dialogue Systems (TDSs) have attracted a lot of
attention for their superiority (e.g., in terms of global optimization) over
pipeline modularized TDSs. Previous studies on end-to-end TDSs use a
single-module model to generate responses for complex dialogue contexts.
However, no model consistently outperforms the others in all cases. We propose
a neural Modular Task-oriented Dialogue System(MTDS) framework, in which a few
expert bots are combined to generate the response for a given dialogue context.
MTDS consists of a chair bot and several expert bots. Each expert bot is
specialized for a particular situation, e.g., one domain, one type of action of
a system, etc. The chair bot coordinates multiple expert bots and adaptively
selects an expert bot to generate the appropriate response. We further propose
a Token-level Mixture-of-Expert (TokenMoE) model to implement MTDS, where the
expert bots predict multiple tokens at each timestamp and the chair bot
determines the final generated token by fully taking into consideration the
outputs of all expert bots. Both the chair bot and the expert bots are jointly
trained in an end-to-end fashion. To verify the effectiveness of TokenMoE, we
carry out extensive experiments on a benchmark dataset. Compared with the
baseline using a single-module model, our TokenMoE improves the performance by
8.1% of inform rate and 0.8% of success rate.

----

20. [**LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning**](http://arxiv.org/pdf/2308.03303v1) by Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, Bo Li

	### Summary

	The low-rank adaptation (LoRA) method can largely reduce the amount of
trainable parameters for fine-tuning large language models (LLMs), however, it
still requires expensive activation memory to update low-rank weights. Reducing
the number of LoRA layers or using activation recomputation could harm the
fine-tuning performance or increase the computational overhead. In this work,
we present LoRA-FA, a memory-efficient fine-tuning method that reduces the
activation memory without performance degradation and expensive recomputation.
LoRA-FA chooses to freeze the projection-down weight of $A$ and update the
projection-up weight of $B$ in each LoRA layer. It ensures the change of model
weight reside in a low-rank space during LLMs fine-tuning, while eliminating
the requirement to store full-rank input activations. We conduct extensive
experiments across multiple model types (RoBERTa, T5, LLaMA) and model scales.
Our results show that LoRA-FA can always achieve close fine-tuning accuracy
across different tasks compared to full parameter fine-tuning and LoRA.
Furthermore, LoRA-FA can reduce the overall memory cost by up to 1.4$\times$
compared to LoRA.

----

21. [**Past-present temporal programs over finite traces**](http://arxiv.org/pdf/2307.12620v1) by Pedro Cabalar, Martín Diéguez, François Laferrière, Torsten Schaub

	### Summary

	Extensions of Answer Set Programming with language constructs from temporal
logics, such as temporal equilibrium logic over finite traces (TELf), provide
an expressive computational framework for modeling dynamic applications. In
this paper, we study the so-called past-present syntactic subclass, which
consists of a set of logic programming rules whose body references to the past
and head to the present. Such restriction ensures that the past remains
independent of the future, which is the case in most dynamic domains. We extend
the definitions of completion and loop formulas to the case of past-present
formulas, which allows capturing the temporal stable models of a set of
past-present temporal programs by means of an LTLf expression.

----

22. [**Avoiding Inference Heuristics in Few-shot Prompt-based Finetuning**](http://arxiv.org/pdf/2109.04144v1) by Prasetya Ajie Utama, Nafise Sadat Moosavi, Victor Sanh, Iryna Gurevych

	### Summary

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

----

23. [**Transductive Auxiliary Task Self-Training for Neural Multi-Task Models**](http://arxiv.org/pdf/1908.06136v2) by Johannes Bjerva, Katharina Kann, Isabelle Augenstein

	### Summary

	Multi-task learning and self-training are two common ways to improve a
machine learning model's performance in settings with limited training data.
Drawing heavily on ideas from those two approaches, we suggest transductive
auxiliary task self-training: training a multi-task model on (i) a combination
of main and auxiliary task training data, and (ii) test instances with
auxiliary task labels which a single-task version of the model has previously
generated. We perform extensive experiments on 86 combinations of languages and
tasks. Our results are that, on average, transductive auxiliary task
self-training improves absolute accuracy by up to 9.56% over the pure
multi-task model for dependency relation tagging and by up to 13.03% for
semantic tagging.

----

24. [**Discrete and Soft Prompting for Multilingual Models**](http://arxiv.org/pdf/2109.03630v1) by Mengjie Zhao, Hinrich Schütze

	### Summary

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

----

25. [**Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models**](http://arxiv.org/pdf/2106.13353v2) by Robert L. Logan IV, Ivana Balažević, Eric Wallace, Fabio Petroni, Sameer Singh, Sebastian Riedel

	### Summary

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

----

26. [**GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation**](http://arxiv.org/pdf/2104.08826v2) by Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, Woomyeong Park

	### Summary

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

----

27. [**Calibrate Before Use: Improving Few-Shot Performance of Language Models**](http://arxiv.org/pdf/2102.09690v2) by Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh

	### Summary

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

----

28. [**It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners**](http://arxiv.org/pdf/2009.07118v2) by Timo Schick, Hinrich Schütze

	### Summary

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

----

29. [**Multilingual Sentence-Level Semantic Search using Meta-Distillation Learning**](http://arxiv.org/pdf/2309.08185v1) by Meryem M'hamdi, Jonathan May, Franck Dernoncourt, Trung Bui, Seunghyun Yoon

	### Summary

	Multilingual semantic search is the task of retrieving relevant contents to a
query expressed in different language combinations. This requires a better
semantic understanding of the user's intent and its contextual meaning.
Multilingual semantic search is less explored and more challenging than its
monolingual or bilingual counterparts, due to the lack of multilingual parallel
resources for this task and the need to circumvent "language bias". In this
work, we propose an alignment approach: MAML-Align, specifically for
low-resource scenarios. Our approach leverages meta-distillation learning based
on MAML, an optimization-based Model-Agnostic Meta-Learner. MAML-Align distills
knowledge from a Teacher meta-transfer model T-MAML, specialized in
transferring from monolingual to bilingual semantic search, to a Student model
S-MAML, which meta-transfers from bilingual to multilingual semantic search. To
the best of our knowledge, we are the first to extend meta-distillation to a
multilingual search application. Our empirical results show that on top of a
strong baseline based on sentence transformers, our meta-distillation approach
boosts the gains provided by MAML and significantly outperforms naive
fine-tuning methods. Furthermore, multilingual meta-distillation learning
improves generalization even to unseen languages.

----

30. [**The Power of Scale for Parameter-Efficient Prompt Tuning**](http://arxiv.org/pdf/2104.08691v2) by Brian Lester, Rami Al-Rfou, Noah Constant

	### Summary

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

----

31. [**True Few-Shot Learning with Language Models**](http://arxiv.org/pdf/2105.11447v1) by Ethan Perez, Douwe Kiela, Kyunghyun Cho

	### Summary

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

----


## 2022
1. [**Reframing Instructional Prompts to GPTk's Language**](http://arxiv.org/pdf/2109.07830v3) by Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, Hannaneh Hajishirzi

	### Summary

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

----

2. [**WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**](http://arxiv.org/pdf/2306.07906v1) by Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong, Jie Tang

	### Summary

	We present WebGLM, a web-enhanced question-answering system based on the
General Language Model (GLM). Its goal is to augment a pre-trained large
language model (LLM) with web search and retrieval capabilities while being
efficient for real-world deployments. To achieve this, we develop WebGLM with
strategies for the LLM-augmented retriever, bootstrapped generator, and human
preference-aware scorer. Specifically, we identify and address the limitations
of WebGPT (OpenAI), through which WebGLM is enabled with accuracy, efficiency,
and cost-effectiveness advantages. In addition, we propose systematic criteria
for evaluating web-enhanced QA systems. We conduct multi-dimensional human
evaluation and quantitative ablation studies, which suggest the outperformance
of the proposed WebGLM designs over existing systems. WebGLM with the
10-billion-parameter GLM (10B) is shown to perform better than the
similar-sized WebGPT (13B) and even comparably to WebGPT (175B) in human
evaluation. The code, demo, and data are at
\url{https://github.com/THUDM/WebGLM}.

----

3. [**Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango**](http://arxiv.org/pdf/2209.07686v2) by Aman Madaan, Amir Yazdanbakhsh

	### Summary

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

----

4. [**PoKE: A Prompt-based Knowledge Eliciting Approach for Event Argument Extraction**](http://arxiv.org/pdf/2109.05190v3) by Jiaju Lin, Qin Chen

	### Summary

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

----

5. [**Zero-Shot Task Transfer**](http://arxiv.org/pdf/1903.01092v1) by Arghya Pal, Vineeth N Balasubramanian

	### Summary

	In this work, we present a novel meta-learning algorithm, i.e. TTNet, that
regresses model parameters for novel tasks for which no ground truth is
available (zero-shot tasks). In order to adapt to novel zero-shot tasks, our
meta-learner learns from the model parameters of known tasks (with ground
truth) and the correlation of known tasks to zero-shot tasks. Such intuition
finds its foothold in cognitive science, where a subject (human baby) can adapt
to a novel-concept (depth understanding) by correlating it with old concepts
(hand movement or self-motion), without receiving explicit supervision. We
evaluated our model on the Taskonomy dataset, with four tasks as zero-shot:
surface-normal, room layout, depth, and camera pose estimation. These tasks
were chosen based on the data acquisition complexity and the complexity
associated with the learning process using a deep network. Our proposed
methodology out-performs state-of-the-art models (which use ground truth)on
each of our zero-shot tasks, showing promise on zero-shot task transfer. We
also conducted extensive experiments to study the various choices of our
methodology, as well as showed how the proposed method can also be used in
transfer learning. To the best of our knowledge, this is the firstsuch effort
on zero-shot learning in the task space.

----

6. [**PPT: Pre-trained Prompt Tuning for Few-shot Learning**](http://arxiv.org/pdf/2109.04332v3) by Yuxian Gu, Xu Han, Zhiyuan Liu, Minlie Huang

	### Summary

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

----

7. [**MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**](http://arxiv.org/pdf/2206.08853v2) by Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, Anima Anandkumar

	### Summary

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

----

8. [**Investigating the Translation Performance of a Large Multilingual Language Model: the Case of BLOOM**](http://arxiv.org/pdf/2303.01911v2) by Rachel Bawden, François Yvon

	### Summary

	The NLP community recently saw the release of a new large open-access
multilingual language model, BLOOM (BigScience et al., 2022) covering 46
languages. We focus on BLOOM's multilingual ability by evaluating its machine
translation performance across several datasets (WMT, Flores-101 and DiaBLa)
and language pairs (high- and low-resourced). Our results show that 0-shot
performance suffers from overgeneration and generating in the wrong language,
but this is greatly improved in the few-shot setting, with very good results
for a number of language pairs. We study several aspects including prompt
design, model sizes, cross-lingual transfer and the use of discursive context.

----

9. [**Cramming: Training a Language Model on a Single GPU in One Day**](http://arxiv.org/pdf/2212.14034v1) by Jonas Geiping, Tom Goldstein

	### Summary

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

----

10. [**SPT: Semi-Parametric Prompt Tuning for Multitask Prompted Learning**](http://arxiv.org/pdf/2212.10929v1) by M Saiful Bari, Aston Zhang, Shuai Zheng, Xingjian Shi, Yi Zhu, Shafiq Joty, Mu Li

	### Summary

	Pre-trained large language models can efficiently interpolate human-written
prompts in a natural way. Multitask prompted learning can help generalization
through a diverse set of tasks at once, thus enhancing the potential for more
effective downstream fine-tuning. To perform efficient multitask-inference in
the same batch, parameter-efficient fine-tuning methods such as prompt tuning
have been proposed. However, the existing prompt tuning methods may lack
generalization. We propose SPT, a semi-parametric prompt tuning method for
multitask prompted learning. The novel component of SPT is a memory bank from
where memory prompts are retrieved based on discrete prompts. Extensive
experiments, such as (i) fine-tuning a full language model with SPT on 31
different tasks from 8 different domains and evaluating zero-shot
generalization on 9 heldout datasets under 5 NLP task categories and (ii)
pretraining SPT on the GLUE datasets and evaluating fine-tuning on the
SuperGLUE datasets, demonstrate effectiveness of SPT.

----

11. [**Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**](http://arxiv.org/pdf/2204.05862v1) by Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan

	### Summary

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

----

12. [**CLUES: A Benchmark for Learning Classifiers using Natural Language Explanations**](http://arxiv.org/pdf/2204.07142v1) by Rakesh R Menon, Sayan Ghosh, Shashank Srivastava

	### Summary

	Supervised learning has traditionally focused on inductive learning by
observing labeled examples of a task. In contrast, humans have the ability to
learn new concepts from language. Here, we explore training zero-shot
classifiers for structured data purely from language. For this, we introduce
CLUES, a benchmark for Classifier Learning Using natural language ExplanationS,
consisting of a range of classification tasks over structured data along with
natural language supervision in the form of explanations. CLUES consists of 36
real-world and 144 synthetic classification tasks. It contains crowdsourced
explanations describing real-world tasks from multiple teachers and
programmatically generated explanations for the synthetic tasks. To model the
influence of explanations in classifying an example, we develop ExEnt, an
entailment-based model that learns classifiers using explanations. ExEnt
generalizes up to 18% better (relative) on novel tasks than a baseline that
does not use explanations. We delineate key challenges for automated learning
from explanations, addressing which can lead to progress on CLUES in the
future. Code and datasets are available at: https://clues-benchmark.github.io.

----

13. [**Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**](http://arxiv.org/pdf/2204.01691v2) by Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, Andy Zeng

	### Summary

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

----

14. [**Ask Me Anything: A simple strategy for prompting language models**](http://arxiv.org/pdf/2210.02441v3) by Simran Arora, Avanika Narayan, Mayee F. Chen, Laurel Orr, Neel Guha, Kush Bhatia, Ines Chami, Frederic Sala, Christopher Ré

	### Summary

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

----

15. [**Can Large Language Models design a Robot?**](http://arxiv.org/pdf/2303.15324v1) by Francesco Stella, Cosimo Della Santina, Josie Hughes

	### Summary

	Large Language Models can lead researchers in the design of robots.

----

16. [**Noisy Channel Language Model Prompting for Few-Shot Text Classification**](http://arxiv.org/pdf/2108.04106v3) by Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer

	### Summary

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

----

17. [**Crosslingual Generalization through Multitask Finetuning**](http://arxiv.org/pdf/2211.01786v2) by Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, Colin Raffel

	### Summary

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

----

18. [**UL2: Unifying Language Learning Paradigms**](http://arxiv.org/pdf/2205.05131v3) by Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, Donald Metzler

	### Summary

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

----

19. [**ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**](http://arxiv.org/pdf/2112.01488v3) by Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia

	### Summary

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

----

20. [**ProgPrompt: Generating Situated Robot Task Plans using Large Language Models**](http://arxiv.org/pdf/2209.11302v1) by Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg

	### Summary

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

----

21. [**Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**](http://arxiv.org/pdf/2211.12588v4) by Wenhu Chen, Xueguang Ma, Xinyi Wang, William W. Cohen

	### Summary

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

----

22. [**Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos**](http://arxiv.org/pdf/2206.11795v1) by Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, Jeff Clune

	### Summary

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

----

23. [**Surface Form Competition: Why the Highest Probability Answer Isn't Always Right**](http://arxiv.org/pdf/2104.08315v9) by Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, Luke Zettlemoyer

	### Summary

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

----

24. [**Code Prompting: a Neural Symbolic Method for Complex Reasoning in Large Language Models**](http://arxiv.org/pdf/2305.18507v2) by Yi Hu, Haotong Yang, Zhouchen Lin, Muhan Zhang

	### Summary

	Large language models (LLMs) have scaled up to unlock a wide range of complex
reasoning tasks with the aid of various prompting methods. However, current
prompting methods generate natural language intermediate steps to help
reasoning, which can cause imperfect task reduction and confusion. To mitigate
such limitations, we explore code prompting, a neural symbolic prompting method
with both zero-shot and few-shot versions which triggers code as intermediate
steps. We conduct experiments on 7 widely-used benchmarks involving symbolic
reasoning and arithmetic reasoning. Code prompting generally outperforms
chain-of-thought (CoT) prompting. To further understand the performance and
limitations of code prompting, we perform extensive ablation studies and error
analyses, and identify several exclusive advantages of using symbolic
promptings compared to natural language. We also consider the ensemble of code
prompting and CoT prompting to combine the strengths of both. Finally, we show
through experiments how code annotations and their locations affect code
prompting.

----

25. [**Language Models of Code are Few-Shot Commonsense Learners**](http://arxiv.org/pdf/2210.07128v3) by Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, Graham Neubig

	### Summary

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

----

26. [**Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning**](http://arxiv.org/pdf/2106.09226v2) by Colin Wei, Sang Michael Xie, Tengyu Ma

	### Summary

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

----

27. [**STaR: Bootstrapping Reasoning With Reasoning**](http://arxiv.org/pdf/2203.14465v2) by Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman

	### Summary

	Generating step-by-step "chain-of-thought" rationales improves language model
performance on complex reasoning tasks like mathematics or commonsense
question-answering. However, inducing language model rationale generation
currently requires either constructing massive rationale datasets or
sacrificing accuracy by using only few-shot inference. We propose a technique
to iteratively leverage a small number of rationale examples and a large
dataset without rationales, to bootstrap the ability to perform successively
more complex reasoning. This technique, the "Self-Taught Reasoner" (STaR),
relies on a simple loop: generate rationales to answer many questions, prompted
with a few rationale examples; if the generated answers are wrong, try again to
generate a rationale given the correct answer; fine-tune on all the rationales
that ultimately yielded correct answers; repeat. We show that STaR
significantly improves performance on multiple datasets compared to a model
fine-tuned to directly predict final answers, and performs comparably to
fine-tuning a 30$\times$ larger state-of-the-art language model on
CommensenseQA. Thus, STaR lets a model improve itself by learning from its own
generated reasoning.

----

28. [**CINS: Comprehensive Instruction for Few-shot Learning in Task-oriented Dialog Systems**](http://arxiv.org/pdf/2109.04645v4) by Fei Mi, Yitong Li, Yasheng Wang, Xin Jiang, Qun Liu

	### Summary

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

----

29. [**BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage**](http://arxiv.org/pdf/2208.03188v3) by Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, Morteza Behrooz, William Ngan, Spencer Poff, Naman Goyal, Arthur Szlam, Y-Lan Boureau, Melanie Kambadur, Jason Weston

	### Summary

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

----

30. [**Exploring Prompt-based Few-shot Learning for Grounded Dialog Generation**](http://arxiv.org/pdf/2109.06513v2) by Chujie Zheng, Minlie Huang

	### Summary

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

----

31. [**LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action**](http://arxiv.org/pdf/2207.04429v2) by Dhruv Shah, Blazej Osinski, Brian Ichter, Sergey Levine

	### Summary

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

----

32. [**Deduplicating Training Data Makes Language Models Better**](http://arxiv.org/pdf/2107.06499v2) by Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini

	### Summary

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

----

33. [**Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification**](http://arxiv.org/pdf/2108.02035v2) by Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li, Wei Wu, Maosong Sun

	### Summary

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

----

34. [**Inner Monologue: Embodied Reasoning through Planning with Language Models**](http://arxiv.org/pdf/2207.05608v1) by Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, Brian Ichter

	### Summary

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

----

35. [**Constitutional AI: Harmlessness from AI Feedback**](http://arxiv.org/pdf/2212.08073v1) by Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan

	### Summary

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

----

36. [**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**](http://arxiv.org/pdf/2205.14135v2) by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

	### Summary

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

----

37. [**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**](http://arxiv.org/pdf/2110.07602v3) by Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang

	### Summary

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

----

38. [**Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**](http://arxiv.org/pdf/2209.09513v2) by Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan

	### Summary

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

----

39. [**Cedille: A large autoregressive French language model**](http://arxiv.org/pdf/2202.03371v1) by Martin Müller, Florian Laurent

	### Summary

	Scaling up the size and training of autoregressive language models has
enabled novel ways of solving Natural Language Processing tasks using zero-shot
and few-shot learning. While extreme-scale language models such as GPT-3 offer
multilingual capabilities, zero-shot learning for languages other than English
remain largely unexplored. Here, we introduce Cedille, a large open source
auto-regressive language model, specifically trained for the French language.
Our results show that Cedille outperforms existing French language models and
is competitive with GPT-3 on a range of French zero-shot benchmarks.
Furthermore, we provide an in-depth comparison of the toxicity exhibited by
these models, showing that Cedille marks an improvement in language model
safety thanks to dataset filtering.

----

40. [**Question Answering and Question Generation for Finnish**](http://arxiv.org/pdf/2211.13794v1) by Ilmari Kylliäinen, Roman Yangarber

	### Summary

	Recent advances in the field of language modeling have improved the
state-of-the-art in question answering (QA) and question generation (QG).
However, the development of modern neural models, their benchmarks, and
datasets for training them has mainly focused on English. Finnish, like many
other languages, faces a shortage of large QA/QG model training resources,
which has prevented experimenting with state-of-the-art QA/QG fine-tuning
methods. We present the first neural QA and QG models that work with Finnish.
To train the models, we automatically translate the SQuAD dataset and then use
normalization methods to reduce the amount of problematic data created during
the translation. Using the synthetic data, together with the Finnish partition
of the TyDi-QA dataset, we fine-tune several transformer-based models to both
QA and QG and evaluate their performance. To the best of our knowledge, the
resulting dataset is the first large-scale QA/QG resource for Finnish. This
paper also sets the initial benchmarks for Finnish-language QA and QG.

----

41. [**Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents**](http://arxiv.org/pdf/2201.07207v2) by Wenlong Huang, Pieter Abbeel, Deepak Pathak, Igor Mordatch

	### Summary

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

----

42. [**Inferring Implicit Relations in Complex Questions with Language Models**](http://arxiv.org/pdf/2204.13778v2) by Uri Katz, Mor Geva, Jonathan Berant

	### Summary

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

----

43. [**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**](http://arxiv.org/pdf/2204.00598v2) by Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, Pete Florence

	### Summary

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

----

44. [**Mind's Eye: Grounded Language Model Reasoning through Simulation**](http://arxiv.org/pdf/2210.05359v1) by Ruibo Liu, Jason Wei, Shixiang Shane Gu, Te-Yen Wu, Soroush Vosoughi, Claire Cui, Denny Zhou, Andrew M. Dai

	### Summary

	Successful and effective communication between humans and AI relies on a
shared experience of the world. By training solely on written text, current
language models (LMs) miss the grounded experience of humans in the real-world
-- their failure to relate language to the physical world causes knowledge to
be misrepresented and obvious mistakes in their reasoning. We present Mind's
Eye, a paradigm to ground language model reasoning in the physical world. Given
a physical reasoning question, we use a computational physics engine
(DeepMind's MuJoCo) to simulate the possible outcomes, and then use the
simulation results as part of the input, which enables language models to
perform reasoning. Experiments on 39 tasks in a physics alignment benchmark
demonstrate that Mind's Eye can improve reasoning ability by a large margin
(27.9% zero-shot, and 46.0% few-shot absolute accuracy improvement on average).
Smaller language models armed with Mind's Eye can obtain similar performance to
models that are 100x larger. Finally, we confirm the robustness of Mind's Eye
through ablation studies.

----

45. [**The Wisdom of Hindsight Makes Language Models Better Instruction Followers**](http://arxiv.org/pdf/2302.05206v1) by Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, Joseph E. Gonzalez

	### Summary

	Reinforcement learning has seen wide success in finetuning large language
models to better align with instructions via human feedback. The so-called
algorithm, Reinforcement Learning with Human Feedback (RLHF) demonstrates
impressive performance on the GPT series models. However, the underlying
Reinforcement Learning (RL) algorithm is complex and requires an additional
training pipeline for reward and value networks. In this paper, we consider an
alternative approach: converting feedback to instruction by relabeling the
original one and training the model for better alignment in a supervised
manner. Such an algorithm doesn't require any additional parameters except for
the original language model and maximally reuses the pretraining pipeline. To
achieve this, we formulate instruction alignment problem for language models as
a goal-reaching problem in decision making. We propose Hindsight Instruction
Relabeling (HIR), a novel algorithm for aligning language models with
instructions. The resulting two-stage algorithm shed light to a family of
reward-free approaches that utilize the hindsightly relabeled instructions
based on feedback. We evaluate the performance of HIR extensively on 12
challenging BigBench reasoning tasks and show that HIR outperforms the baseline
algorithms and is comparable to or even surpasses supervised finetuning.

----

46. [**Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning**](http://arxiv.org/pdf/2308.12219v2) by Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, Quanquan Gu

	### Summary

	The recent surge of generative AI has been fueled by the generative power of
diffusion probabilistic models and the scalable capabilities of large language
models. Despite their potential, it remains elusive whether diffusion language
models can solve general language tasks comparable to their autoregressive
counterparts. This paper demonstrates that scaling diffusion models w.r.t.
data, sizes, and tasks can effectively make them strong language learners. We
build competent diffusion language models at scale by first acquiring knowledge
from massive data via masked language modeling pretraining thanks to their
intrinsic connections. We then reprogram pretrained masked language models into
diffusion language models via diffusive adaptation, wherein task-specific
finetuning and instruction finetuning are explored to unlock their versatility
in solving general language tasks. Experiments show that scaling diffusion
language models consistently improves performance across downstream language
tasks. We further discover that instruction finetuning can elicit zero-shot and
few-shot in-context learning abilities that help tackle many unseen tasks by
following natural language instructions, and show promise in advanced and
challenging abilities such as reasoning.

----

47. [**Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners**](http://arxiv.org/pdf/2108.13161v7) by Ningyu Zhang, Luoqiu Li, Xiang Chen, Shumin Deng, Zhen Bi, Chuanqi Tan, Fei Huang, Huajun Chen

	### Summary

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

----

48. [**Interactive Language: Talking to Robots in Real Time**](http://arxiv.org/pdf/2210.06407v1) by Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong, Pete Florence

	### Summary

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

----

49. [**PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains**](http://arxiv.org/pdf/2102.12206v4) by Eyal Ben-David, Nadav Oved, Roi Reichart

	### Summary

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

----

50. [**The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning**](http://arxiv.org/pdf/2205.03401v2) by Xi Ye, Greg Durrett

	### Summary

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

----

51. [**MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization**](http://arxiv.org/pdf/2401.06838v1) by Shuaijie She, Shujian Huang, Wei Zou, Wenhao Zhu, Xiang Liu, Xiang Geng, Jiajun Chen

	### Summary

	Though reasoning abilities are considered language-agnostic, existing LLMs
exhibit inconsistent reasoning abilities across different languages, e.g.,
reasoning in a pivot language is superior to other languages due to the
imbalance of multilingual training data.To enhance reasoning abilities in
non-pivot languages, we propose an alignment-as-preference optimization
framework. Specifically, we adopt an open-source translation model to estimate
the consistency between answers in non-pivot and pivot languages. We further
adopt the answer consistency as the preference for DPO or PPO thus optimizing
the lesser reasoning. Experiments show that our method significantly improves
the model's multilingual reasoning, with better reasoning consistency across
languages. Our framework achieved a 13.7% accuracy improvement on out-of-domain
datasets MSVAMP while preserving the competitive performance on MGSM. Moreover,
we find that iterative DPO is helpful for further alignment and improvement of
the model's multilingual mathematical reasoning ability, further pushing the
improvement to 16.7%

----

52. [**Generated Knowledge Prompting for Commonsense Reasoning**](http://arxiv.org/pdf/2110.08387v3) by Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, Hannaneh Hajishirzi

	### Summary

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

----

53. [**Catastrophic Forgetting in the Context of Model Updates**](http://arxiv.org/pdf/2306.10181v1) by Rich Harang, Hillary Sanders

	### Summary

	A large obstacle to deploying deep learning models in practice is the process
of updating models post-deployment (ideally, frequently). Deep neural networks
can cost many thousands of dollars to train. When new data comes in the
pipeline, you can train a new model from scratch (randomly initialized weights)
on all existing data. Instead, you can take an existing model and fine-tune
(continue to train) it on new data. The former is costly and slow. The latter
is cheap and fast, but catastrophic forgetting generally causes the new model
to 'forget' how to classify older data well. There are a plethora of
complicated techniques to keep models from forgetting their past learnings.
Arguably the most basic is to mix in a small amount of past data into the new
data during fine-tuning: also known as 'data rehearsal'. In this paper, we
compare various methods of limiting catastrophic forgetting and conclude that
if you can maintain access to a portion of your past data (or tasks), data
rehearsal is ideal in terms of overall accuracy across all time periods, and
performs even better when combined with methods like Elastic Weight
Consolidation (EWC). Especially when the amount of past data (past 'tasks') is
large compared to new data, the cost of updating an existing model is far
cheaper and faster than training a new model from scratch.

----

54. [**Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning**](http://arxiv.org/pdf/2205.09712v1) by Antonia Creswell, Murray Shanahan, Irina Higgins

	### Summary

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

----

55. [**Ignore Previous Prompt: Attack Techniques For Language Models**](http://arxiv.org/pdf/2211.09527v1) by Fábio Perez, Ian Ribeiro

	### Summary

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

----

56. [**Self-Instruct: Aligning Language Models with Self-Generated Instructions**](http://arxiv.org/pdf/2212.10560v2) by Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi

	### Summary

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

----

57. [**Do Prompt-Based Models Really Understand the Meaning of their Prompts?**](http://arxiv.org/pdf/2109.01247v2) by Albert Webson, Ellie Pavlick

	### Summary

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

----

58. [**KronA: Parameter Efficient Tuning with Kronecker Adapter**](http://arxiv.org/pdf/2212.10650v1) by Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J. Clark, Mehdi Rezagholizadeh

	### Summary

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

----

59. [**Harnessing Pre-trained Generalist Agents for Software Engineering Tasks**](http://arxiv.org/pdf/2312.15536v1) by Paulina Stevia Nouwou Mindom, Amin Nikanjam, Foutse Khomh

	### Summary

	Nowadays, we are witnessing an increasing adoption of Artificial Intelligence
(AI) to develop techniques aimed at improving the reliability, effectiveness,
and overall quality of software systems. Deep reinforcement learning (DRL) has
recently been successfully used for automation in complex tasks such as game
testing and solving the job-shop scheduling problem. However, these specialized
DRL agents, trained from scratch on specific tasks, suffer from a lack of
generalizability to other tasks and they need substantial time to be developed
and re-trained effectively. Recently, DRL researchers have begun to develop
generalist agents, able to learn a policy from various environments and capable
of achieving performances similar to or better than specialist agents in new
tasks. In the Natural Language Processing or Computer Vision domain, these
generalist agents are showing promising adaptation capabilities to
never-before-seen tasks after a light fine-tuning phase and achieving high
performance. This paper investigates the potential of generalist agents for
solving SE tasks. Specifically, we conduct an empirical study aimed at
assessing the performance of two generalist agents on two important SE tasks:
the detection of bugs in games (for two games) and the minimization of makespan
in a scheduling task, to solve the job-shop scheduling problem (for two
instances). Our results show that the generalist agents outperform the
specialist agents with very little effort for fine-tuning, achieving a 20%
reduction of the makespan over specialized agent performance on task-based
scheduling. In the context of game testing, some generalist agent
configurations detect 85% more bugs than the specialist agents. Building on our
analysis, we provide recommendations for researchers and practitioners looking
to select generalist agents for SE tasks, to ensure that they perform
effectively.

----

60. [**Are Emergent Abilities of Large Language Models a Mirage?**](http://arxiv.org/pdf/2304.15004v2) by Rylan Schaeffer, Brando Miranda, Sanmi Koyejo

	### Summary

	Recent work claims that large language models display emergent abilities,
abilities not present in smaller-scale models that are present in larger-scale
models. What makes emergent abilities intriguing is two-fold: their sharpness,
transitioning seemingly instantaneously from not present to present, and their
unpredictability, appearing at seemingly unforeseeable model scales. Here, we
present an alternative explanation for emergent abilities: that for a
particular task and model family, when analyzing fixed model outputs, emergent
abilities appear due to the researcher's choice of metric rather than due to
fundamental changes in model behavior with scale. Specifically, nonlinear or
discontinuous metrics produce apparent emergent abilities, whereas linear or
continuous metrics produce smooth, continuous predictable changes in model
performance. We present our alternative explanation in a simple mathematical
model, then test it in three complementary ways: we (1) make, test and confirm
three predictions on the effect of metric choice using the InstructGPT/GPT-3
family on tasks with claimed emergent abilities; (2) make, test and confirm two
predictions about metric choices in a meta-analysis of emergent abilities on
BIG-Bench; and (3) show to choose metrics to produce never-before-seen
seemingly emergent abilities in multiple vision tasks across diverse deep
networks. Via all three analyses, we provide evidence that alleged emergent
abilities evaporate with different metrics or with better statistics, and may
not be a fundamental property of scaling AI models.

----

61. [**Promptagator: Few-shot Dense Retrieval From 8 Examples**](http://arxiv.org/pdf/2209.11755v1) by Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, Ming-Wei Chang

	### Summary

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

----

62. [**Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity**](http://arxiv.org/pdf/2104.08786v2) by Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp

	### Summary

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

----

63. [**Chain-Of-Thought Prompting Under Streaming Batch: A Case Study**](http://arxiv.org/pdf/2306.00550v1) by Yuxin Tang

	### Summary

	Recently, Large Language Models (LLMs) have demonstrated remarkable
capabilities. Chain-of-Thought (CoT) has been proposed as a way of assisting
LLMs in performing complex reasoning. However, developing effective prompts can
be a challenging and labor-intensive task. Many studies come out of some way to
automatically construct CoT from test data. Most of them assume that all test
data is visible before testing and only select a small subset to generate
rationales, which is an unrealistic assumption. In this paper, we present a
case study on how to construct and optimize chain-of-thought prompting using
batch data in streaming settings.

----

## 2023
1. [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](http://arxiv.org/pdf/2303.17580v4) by Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang
	## Summary
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

----

2. [**Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch**](http://arxiv.org/pdf/2311.03099v1) by Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li
	## Summary
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

----

3. [**An Evaluation of Generative Pre-Training Model-based Therapy Chatbot for Caregivers**](http://arxiv.org/pdf/2107.13115v1) by Lu Wang, Munif Ishad Mujib, Jake Williams, George Demiris, Jina Huh-Yoo
	## Summary
	With the advent of off-the-shelf intelligent home products and broader
internet adoption, researchers increasingly explore smart computing
applications that provide easier access to health and wellness resources.
AI-based systems like chatbots have the potential to provide services that
could provide mental health support. However, existing therapy chatbots are
often retrieval-based, requiring users to respond with a constrained set of
answers, which may not be appropriate given that such pre-determined inquiries
may not reflect each patient's unique circumstances. Generative-based
approaches, such as the OpenAI GPT models, could allow for more dynamic
conversations in therapy chatbot contexts than previous approaches. To
investigate the generative-based model's potential in therapy chatbot contexts,
we built a chatbot using the GPT-2 model. We fine-tuned it with 306 therapy
session transcripts between family caregivers of individuals with dementia and
therapists conducting Problem Solving Therapy. We then evaluated the model's
pre-trained and the fine-tuned model in terms of basic qualities using three
meta-information measurements: the proportion of non-word outputs, the length
of response, and sentiment components. Results showed that: (1) the fine-tuned
model created more non-word outputs than the pre-trained model; (2) the
fine-tuned model generated outputs whose length was more similar to that of the
therapists compared to the pre-trained model; (3) both the pre-trained model
and fine-tuned model were likely to generate more negative and fewer positive
outputs than the therapists. We discuss potential reasons for the problem, the
implications, and solutions for developing therapy chatbots and call for
investigations of the AI-based system application.

----

4. [**Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature**](http://arxiv.org/pdf/2310.16146v1) by Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, Nigam Shah
	## Summary
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

----

5. [**Social Simulacra: Creating Populated Prototypes for Social Computing Systems**](http://arxiv.org/pdf/2208.04024v1) by Joon Sung Park, Lindsay Popowski, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
	## Summary
	Social computing prototypes probe the social behaviors that may arise in an
envisioned system design. This prototyping practice is currently limited to
recruiting small groups of people. Unfortunately, many challenges do not arise
until a system is populated at a larger scale. Can a designer understand how a
social system might behave when populated, and make adjustments to the design
before the system falls prey to such challenges? We introduce social simulacra,
a prototyping technique that generates a breadth of realistic social
interactions that may emerge when a social computing system is populated.
Social simulacra take as input the designer's description of a community's
design -- goal, rules, and member personas -- and produce as output an instance
of that design with simulated behavior, including posts, replies, and
anti-social behaviors. We demonstrate that social simulacra shift the behaviors
that they generate appropriately in response to design changes, and that they
enable exploration of "what if?" scenarios where community members or
moderators intervene. To power social simulacra, we contribute techniques for
prompting a large language model to generate thousands of distinct community
members and their social interactions with each other; these techniques are
enabled by the observation that large language models' training data already
includes a wide variety of positive and negative behavior on social media
platforms. In evaluations, we show that participants are often unable to
distinguish social simulacra from actual community behavior and that social
computing designers successfully refine their social computing designs when
using social simulacra.

----

6. [**Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond**](http://arxiv.org/pdf/2304.13712v2) by Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu
	## Summary
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

----

7. [**Conversational Health Agents: A Personalized LLM-Powered Agent Framework**](http://arxiv.org/pdf/2310.02374v3) by Mahyar Abbasian, Iman Azimi, Amir M. Rahmani, Ramesh Jain
	## Summary
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

----

8. [**ReviewerGPT? An Exploratory Study on Using Large Language Models for Paper Reviewing**](http://arxiv.org/pdf/2306.00622v1) by Ryan Liu, Nihar B. Shah
	## Summary
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

----

9. [**OpenAssistant Conversations -- Democratizing Large Language Model Alignment**](http://arxiv.org/pdf/2304.07327v2) by Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick
	## Summary
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

----

10. [**Large Language Models as Evolutionary Optimizers**](http://arxiv.org/pdf/2310.19046v2) by Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang, Yew-Soon Ong
	## Summary
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

----

11. [**S-LoRA: Serving Thousands of Concurrent LoRA Adapters**](http://arxiv.org/pdf/2311.03285v2) by Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica
	## Summary
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

----

12. [**A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models**](http://arxiv.org/pdf/2312.10982v1) by Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan
	## Summary
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

----

13. [**AutoAgents: A Framework for Automatic Agent Generation**](http://arxiv.org/pdf/2309.17288v2) by Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F. Karlsson, Jie Fu, Yemin Shi
	## Summary
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

----

14. [**Foundation Models for Weather and Climate Data Understanding: A Comprehensive Survey**](http://arxiv.org/pdf/2312.03014v1) by Shengchao Chen, Guodong Long, Jing Jiang, Dikai Liu, Chengqi Zhang
	## Summary
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

----

15. [**ChatGPT's One-year Anniversary: Are Open-Source Large Language Models Catching up?**](http://arxiv.org/pdf/2311.16989v4) by Hailin Chen, Fangkai Jiao, Xingxuan Li, Chengwei Qin, Mathieu Ravaut, Ruochen Zhao, Caiming Xiong, Shafiq Joty
	## Summary
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

----

16. [**Contrastive Preference Learning: Learning from Human Feedback without RL**](http://arxiv.org/pdf/2310.13639v2) by Joey Hejna, Rafael Rafailov, Harshit Sikchi, Chelsea Finn, Scott Niekum, W. Bradley Knox, Dorsa Sadigh
	## Summary
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

----

17. [**Large Language Models Are Reasoning Teachers**](http://arxiv.org/pdf/2212.10071v2) by Namgyu Ho, Laura Schmid, Se-Young Yun
	## Summary
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

----

18. [**From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting**](http://arxiv.org/pdf/2309.04269v1) by Griffin Adams, Alexander Fabbri, Faisal Ladhak, Eric Lehman, Noémie Elhadad
	## Summary
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

----

19. [**Less, but Stronger: On the Value of Strong Heuristics in Semi-supervised Learning for Software Analytics**](http://arxiv.org/pdf/2302.01997v1) by Huy Tu, Tim Menzies
	## Summary
	In many domains, there are many examples and far fewer labels for those
examples; e.g. we may have access to millions of lines of source code, but
access to only a handful of warnings about that code. In those domains,
semi-supervised learners (SSL) can extrapolate labels from a small number of
examples to the rest of the data. Standard SSL algorithms use ``weak''
knowledge (i.e. those not based on specific SE knowledge) such as (e.g.)
co-train two learners and use good labels from one to train the other. Another
approach of SSL in software analytics is potentially use ``strong'' knowledge
that use SE knowledge. For example, an often-used heuristic in SE is that
unusually large artifacts contain undesired properties (e.g. more bugs). This
paper argues that such ``strong'' algorithms perform better than those
standard, weaker, SSL algorithms. We show this by learning models from labels
generated using weak SSL or our ``stronger'' FRUGAL algorithm. In four domains
(distinguishing security-related bug reports; mitigating bias in
decision-making; predicting issue close time; and (reducing false alarms in
static code warnings), FRUGAL required only 2.5% of the data to be labeled yet
out-performed standard semi-supervised learners that relied on (e.g.) some
domain-independent graph theory concepts. Hence, for future work, we strongly
recommend the use of strong heuristics for semi-supervised learning for SE
applications. To better support other researchers, our scripts and data are
on-line at https://github.com/HuyTu7/FRUGAL.

----

20. [**Generalized Graph Prompt: Toward a Unification of Pre-Training and Downstream Tasks on Graphs**](http://arxiv.org/pdf/2311.15317v2) by Xingtong Yu, Zhenghao Liu, Yuan Fang, Zemin Liu, Sihong Chen, Xinming Zhang
	## Summary
	Graph neural networks have emerged as a powerful tool for graph
representation learning, but their performance heavily relies on abundant
task-specific supervision. To reduce labeling requirement, the "pre-train,
prompt" paradigms have become increasingly common. However, existing study of
prompting on graphs is limited, lacking a universal treatment to appeal to
different downstream tasks. In this paper, we propose GraphPrompt, a novel
pre-training and prompting framework on graphs. GraphPrompt not only unifies
pre-training and downstream tasks into a common task template but also employs
a learnable prompt to assist a downstream task in locating the most relevant
knowledge from the pre-trained model in a task-specific manner. To further
enhance GraphPrompt in these two stages, we extend it into GraphPrompt+ with
two major enhancements. First, we generalize several popular graph pre-training
tasks beyond simple link prediction to broaden the compatibility with our task
template. Second, we propose a more generalized prompt design that incorporates
a series of prompt vectors within every layer of the pre-trained graph encoder,
in order to capitalize on the hierarchical information across different layers
beyond just the readout layer. Finally, we conduct extensive experiments on
five public datasets to evaluate and analyze GraphPrompt and GraphPrompt+.

----

21. [**LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models**](http://arxiv.org/pdf/2310.08659v4) by Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao
	## Summary
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

----

22. [**Are We Testing or Being Tested? Exploring the Practical Applications of Large Language Models in Software Testing**](http://arxiv.org/pdf/2312.04860v1) by Robson Santos, Italo Santos, Cleyton Magalhaes, Ronnie de Souza Santos
	## Summary
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

----

23. [**Tree of Thoughts: Deliberate Problem Solving with Large Language Models**](http://arxiv.org/pdf/2305.10601v2) by Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
	## Summary
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

----

24. [**Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts**](http://arxiv.org/pdf/2309.07430v3) by Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari
	## Summary
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

----

25. [**An LLM Compiler for Parallel Function Calling**](http://arxiv.org/pdf/2312.04511v1) by Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami
	## Summary
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

----

26. [**Large-scale Training of Foundation Models for Wearable Biosignals**](http://arxiv.org/pdf/2312.05409v1) by Salar Abbaspourazad, Oussama Elachqar, Andrew C. Miller, Saba Emrani, Udhyakumar Nallasamy, Ian Shapiro
	## Summary
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

----

27. [**Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond**](http://arxiv.org/pdf/2310.02071v4) by Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Tianyu Liu, Baobao Chang
	## Summary
	In this study, we explore the potential of Multimodal Large Language Models
(MLLMs) in improving embodied decision-making processes for agents. While Large
Language Models (LLMs) have been widely used due to their advanced reasoning
skills and vast world knowledge, MLLMs like GPT4-Vision offer enhanced visual
understanding and reasoning capabilities. We investigate whether
state-of-the-art MLLMs can handle embodied decision-making in an end-to-end
manner and whether collaborations between LLMs and MLLMs can enhance
decision-making. To address these questions, we introduce a new benchmark
called PCA-EVAL, which evaluates embodied decision-making from the perspectives
of Perception, Cognition, and Action. Additionally, we propose HOLMES, a
multi-agent cooperation framework that allows LLMs to leverage MLLMs and APIs
to gather multimodal information for informed decision-making. We compare
end-to-end embodied decision-making and HOLMES on our benchmark and find that
the GPT4-Vision model demonstrates strong end-to-end embodied decision-making
abilities, outperforming GPT4-HOLMES in terms of average decision accuracy
(+3%). However, this performance is exclusive to the latest GPT4-Vision model,
surpassing the open-source state-of-the-art MLLM by 26%. Our results indicate
that powerful MLLMs like GPT4-Vision hold promise for decision-making in
embodied agents, offering new avenues for MLLM research. Code and data are open
at https://github.com/pkunlp-icler/PCA-EVAL/.

----

28. [**Towards Reasoning in Large Language Models: A Survey**](http://arxiv.org/pdf/2212.10403v2) by Jie Huang, Kevin Chen-Chuan Chang
	## Summary
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

----

29. [**Alignment for Honesty**](http://arxiv.org/pdf/2312.07000v1) by Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, Pengfei Liu
	## Summary
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

----

30. [**MindAgent: Emergent Gaming Interaction**](http://arxiv.org/pdf/2309.09971v2) by Ran Gong, Qiuyuan Huang, Xiaojian Ma, Hoi Vo, Zane Durante, Yusuke Noda, Zilong Zheng, Song-Chun Zhu, Demetri Terzopoulos, Li Fei-Fei, Jianfeng Gao
	## Summary
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

----

31. [**Compressing Context to Enhance Inference Efficiency of Large Language Models**](http://arxiv.org/pdf/2310.06201v1) by Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin
	## Summary
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

----

32. [**Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt**](http://arxiv.org/pdf/2305.11186v2) by Zhaozhuo Xu, Zirui Liu, Beidi Chen, Yuxin Tang, Jue Wang, Kaixiong Zhou, Xia Hu, Anshumali Shrivastava
	## Summary
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

----

33. [**BloombergGPT: A Large Language Model for Finance**](http://arxiv.org/pdf/2303.17564v3) by Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann
	## Summary
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

----

34. [**JudgeLM: Fine-tuned Large Language Models are Scalable Judges**](http://arxiv.org/pdf/2310.17631v1) by Lianghui Zhu, Xinggang Wang, Xinlong Wang
	## Summary
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

----

35. [**Factored Verification: Detecting and Reducing Hallucination in Summaries of Academic Papers**](http://arxiv.org/pdf/2310.10627v1) by Charlie George, Andreas Stuhlmüller
	## Summary
	Hallucination plagues even frontier LLMs--but how bad is it really for
summarizing academic papers? We evaluate Factored Verification, a simple
automated method for detecting hallucinations in abstractive summaries. This
method sets a new SotA on hallucination detection in the summarization task of
the HaluEval benchmark, achieving 76.2% accuracy. We then use this method to
estimate how often language models hallucinate when summarizing across multiple
academic papers and find 0.62 hallucinations in the average ChatGPT (16k)
summary, 0.84 for GPT-4, and 1.55 for Claude 2. We ask models to self-correct
using Factored Critiques and find that this lowers the number of hallucinations
to 0.49 for ChatGPT, 0.46 for GPT-4, and 0.95 for Claude 2. The hallucinations
we find are often subtle, so we advise caution when using models to synthesize
academic papers.

----

36. [**Mind2Web: Towards a Generalist Agent for the Web**](http://arxiv.org/pdf/2306.06070v3) by Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, Yu Su
	## Summary
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

----

37. [**MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting**](http://arxiv.org/pdf/2305.16896v1) by Tatsuro Inaba, Hirokazu Kiyomaru, Fei Cheng, Sadao Kurohashi
	## Summary
	Large language models (LLMs) have achieved impressive performance on various
reasoning tasks. To further improve the performance, we propose MultiTool-CoT,
a novel framework that leverages chain-of-thought (CoT) prompting to
incorporate multiple external tools, such as a calculator and a knowledge
retriever, during the reasoning process. We apply MultiTool-CoT to the Task 2
dataset of NumGLUE, which requires both numerical reasoning and domain-specific
knowledge. The experiments show that our method significantly outperforms
strong baselines and achieves state-of-the-art performance.

----

38. [**Time is Encoded in the Weights of Finetuned Language Models**](http://arxiv.org/pdf/2312.13401v2) by Kai Nylund, Suchin Gururangan, Noah A. Smith
	## Summary
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

----

39. [**A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**](http://arxiv.org/pdf/2311.05232v1) by Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu
	## Summary
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

----

40. [**Code Prompting: a Neural Symbolic Method for Complex Reasoning in Large Language Models**](http://arxiv.org/pdf/2305.18507v2) by Yi Hu, Haotong Yang, Zhouchen Lin, Muhan Zhang
	## Summary
	Large language models (LLMs) have scaled up to unlock a wide range of complex
reasoning tasks with the aid of various prompting methods. However, current
prompting methods generate natural language intermediate steps to help
reasoning, which can cause imperfect task reduction and confusion. To mitigate
such limitations, we explore code prompting, a neural symbolic prompting method
with both zero-shot and few-shot versions which triggers code as intermediate
steps. We conduct experiments on 7 widely-used benchmarks involving symbolic
reasoning and arithmetic reasoning. Code prompting generally outperforms
chain-of-thought (CoT) prompting. To further understand the performance and
limitations of code prompting, we perform extensive ablation studies and error
analyses, and identify several exclusive advantages of using symbolic
promptings compared to natural language. We also consider the ensemble of code
prompting and CoT prompting to combine the strengths of both. Finally, we show
through experiments how code annotations and their locations affect code
prompting.

----

41. [**A Bibliometric Review of Large Language Models Research from 2017 to 2023**](http://arxiv.org/pdf/2304.02020v1) by Lizhou Fan, Lingyao Li, Zihui Ma, Sanggyu Lee, Huizi Yu, Libby Hemphill
	## Summary
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

----

42. [**AppAgent: Multimodal Agents as Smartphone Users**](http://arxiv.org/pdf/2312.13771v2) by Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, Gang Yu
	## Summary
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

----

43. [**Breaking the Language Barrier: Improving Cross-Lingual Reasoning with Structured Self-Attention**](http://arxiv.org/pdf/2310.15258v1) by Negar Foroutan, Mohammadreza Banaei, Karl Aberer, Antoine Bosselut
	## Summary
	In this work, we study whether multilingual language models (MultiLMs) can
transfer logical reasoning abilities to other languages when they are
fine-tuned for reasoning in a different language. We evaluate the cross-lingual
reasoning abilities of MultiLMs in two schemes: (1) where the language of the
context and the question remain the same in the new languages that are tested
(i.e., the reasoning is still monolingual, but the model must transfer the
learned reasoning ability across languages), and (2) where the language of the
context and the question is different (which we term code-switched reasoning).
On two logical reasoning datasets, RuleTaker and LeapOfThought, we demonstrate
that although MultiLMs can transfer reasoning ability across languages in a
monolingual setting, they struggle to transfer reasoning abilities in a
code-switched setting. Following this observation, we propose a novel attention
mechanism that uses a dedicated set of parameters to encourage cross-lingual
attention in code-switched sequences, which improves the reasoning performance
by up to 14% and 4% on the RuleTaker and LeapOfThought datasets, respectively.

----

44. [**FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**](http://arxiv.org/pdf/2310.03214v2) by Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong
	## Summary
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

----

45. [**Efficient Finetuning Large Language Models For Vietnamese Chatbot**](http://arxiv.org/pdf/2309.04646v1) by Vu-Thuan Doan, Quoc-Truong Truong, Duc-Vu Nguyen, Vinh-Tiep Nguyen, Thuy-Ngan Nguyen Luu
	## Summary
	Large language models (LLMs), such as GPT-4, PaLM, and LLaMa, have been shown
to achieve remarkable performance across a variety of natural language tasks.
Recent advancements in instruction tuning bring LLMs with ability in following
user's instructions and producing human-like responses. However, the high costs
associated with training and implementing LLMs pose challenges to academic
research. Furthermore, the availability of pretrained LLMs and instruction-tune
datasets for Vietnamese language is limited. To tackle these concerns, we
leverage large-scale instruction-following datasets from open-source projects,
namely Alpaca, GPT4All, and Chat-Doctor, which cover general domain and
specific medical domain. To the best of our knowledge, these are the first
instructional dataset for Vietnamese. Subsequently, we utilize
parameter-efficient tuning through Low-Rank Adaptation (LoRA) on two open LLMs:
Bloomz (Multilingual) and GPTJ-6B (Vietnamese), resulting four models:
Bloomz-Chat, Bloomz-Doctor, GPTJ-Chat, GPTJ-Doctor.Finally, we assess the
effectiveness of our methodology on a per-sample basis, taking into
consideration the helpfulness, relevance, accuracy, level of detail in their
responses. This evaluation process entails the utilization of GPT-4 as an
automated scoring mechanism. Despite utilizing a low-cost setup, our method
demonstrates about 20-30\% improvement over the original models in our
evaluation tasks.

----

46. [**FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning**](http://arxiv.org/pdf/2108.06098v3) by Nam Hyeon-Woo, Moon Ye-Bin, Tae-Hyun Oh
	## Summary
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

----

47. [**Towards Automatic Support of Software Model Evolution with Large Language~Models**](http://arxiv.org/pdf/2312.12404v1) by Christof Tinnes, Thomas Fuchß, Uwe Hohenstein, Sven Apel
	## Summary
	Modeling structure and behavior of software systems plays a crucial role, in
various areas of software engineering. As with other software engineering
artifacts, software models are subject to evolution. Supporting modelers in
evolving models by model completion facilities and providing high-level edit
operations such as frequently occurring editing patterns is still an open
problem. Recently, large language models (i.e., generative neural networks)
have garnered significant attention in various research areas, including
software engineering. In this paper, we explore the potential of large language
models in supporting the evolution of software models in software engineering.
We propose an approach that utilizes large language models for model completion
and discovering editing patterns in model histories of software systems.
Through controlled experiments using simulated model repositories, we conduct
an evaluation of the potential of large language models for these two tasks. We
have found that large language models are indeed a promising technology for
supporting software model evolution, and that it is worth investigating further
in the area of software model evolution.

----

48. [**Exploring the Landscape of Large Language Models In Medical Question Answering: Observations and Open Questions**](http://arxiv.org/pdf/2310.07225v1) by Karolina Korgul, Andrew M. Bean, Felix Krones, Robert McCraith, Adam Mahdi
	## Summary
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

----

49. [**LLMEval: A Preliminary Study on How to Evaluate Large Language Models**](http://arxiv.org/pdf/2312.07398v2) by Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, Xuanjing Huang
	## Summary
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

----

50. [**A Survey of GPT-3 Family Large Language Models Including ChatGPT and GPT-4**](http://arxiv.org/pdf/2310.12321v1) by Katikapalli Subramanyam Kalyan
	## Summary
	Large language models (LLMs) are a special class of pretrained language
models obtained by scaling model size, pretraining corpus and computation.
LLMs, because of their large size and pretraining on large volumes of text
data, exhibit special abilities which allow them to achieve remarkable
performances without any task-specific training in many of the natural language
processing tasks. The era of LLMs started with OpenAI GPT-3 model, and the
popularity of LLMs is increasing exponentially after the introduction of models
like ChatGPT and GPT4. We refer to GPT-3 and its successor OpenAI models,
including ChatGPT and GPT4, as GPT-3 family large language models (GLLMs). With
the ever-rising popularity of GLLMs, especially in the research community,
there is a strong need for a comprehensive survey which summarizes the recent
research progress in multiple dimensions and can guide the research community
with insightful future research directions. We start the survey paper with
foundation concepts like transformers, transfer learning, self-supervised
learning, pretrained language models and large language models. We then present
a brief overview of GLLMs and discuss the performances of GLLMs in various
downstream tasks, specific domains and multiple languages. We also discuss the
data labelling and data augmentation abilities of GLLMs, the robustness of
GLLMs, the effectiveness of GLLMs as evaluators, and finally, conclude with
multiple insightful future research directions. To summarize, this
comprehensive survey paper will serve as a good resource for both academic and
industry people to stay updated with the latest research related to GPT-3
family large language models.

----

51. [**Applications of Artificial Intelligence in Live Action Role-Playing Games (LARP)**](http://arxiv.org/pdf/2008.11003v1) by Christoph Salge, Emily Short, Mike Preuss, Spyridion Samothrakis, Pieter Spronck
	## Summary
	Live Action Role-Playing (LARP) games and similar experiences are becoming a
popular game genre. Here, we discuss how artificial intelligence techniques,
particularly those commonly used in AI for Games, could be applied to LARP. We
discuss the specific properties of LARP that make it a surprisingly suitable
application field, and provide a brief overview of some existing approaches. We
then outline several directions where utilizing AI seems beneficial, by both
making LARPs easier to organize, and by enhancing the player experience with
elements not possible without AI.

----

52. [**MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**](http://arxiv.org/pdf/2308.00352v5) by Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, Jürgen Schmidhuber
	## Summary
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

----

53. [**SmartPlay: A Benchmark for LLMs as Intelligent Agents**](http://arxiv.org/pdf/2310.01557v3) by Yue Wu, Xuan Tang, Tom M. Mitchell, Yuanzhi Li
	## Summary
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

----

54. [**Transfer Fine-Tuning: A BERT Case Study**](http://arxiv.org/pdf/1909.00931v1) by Yuki Arase, Junichi Tsujii
	## Summary
	A semantic equivalence assessment is defined as a task that assesses semantic
equivalence in a sentence pair by binary judgment (i.e., paraphrase
identification) or grading (i.e., semantic textual similarity measurement). It
constitutes a set of tasks crucial for research on natural language
understanding. Recently, BERT realized a breakthrough in sentence
representation learning (Devlin et al., 2019), which is broadly transferable to
various NLP tasks. While BERT's performance improves by increasing its model
size, the required computational power is an obstacle preventing practical
applications from adopting the technology. Herein, we propose to inject phrasal
paraphrase relations into BERT in order to generate suitable representations
for semantic equivalence assessment instead of increasing the model size.
Experiments on standard natural language understanding tasks confirm that our
method effectively improves a smaller BERT model while maintaining the model
size. The generated model exhibits superior performance compared to a larger
BERT model on semantic equivalence assessment tasks. Furthermore, it achieves
larger performance gains on tasks with limited training datasets for
fine-tuning, which is a property desirable for transfer learning.

----

55. [**MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**](http://arxiv.org/pdf/2310.09478v3) by Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny
	## Summary
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

----

56. [**PolicyGPT: Automated Analysis of Privacy Policies with Large Language Models**](http://arxiv.org/pdf/2309.10238v1) by Chenhao Tang, Zhengliang Liu, Chong Ma, Zihao Wu, Yiwei Li, Wei Liu, Dajiang Zhu, Quanzheng Li, Xiang Li, Tianming Liu, Lei Fan
	## Summary
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

----

57. [**R-Tuning: Teaching Large Language Models to Refuse Unknown Questions**](http://arxiv.org/pdf/2311.09677v1) by Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, Tong Zhang
	## Summary
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

----

58. [**Guiding Pretraining in Reinforcement Learning with Large Language Models**](http://arxiv.org/pdf/2302.06692v2) by Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, Jacob Andreas
	## Summary
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

----

59. [**ExpertPrompting: Instructing Large Language Models to be Distinguished Experts**](http://arxiv.org/pdf/2305.14688v1) by Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, Zhendong Mao
	## Summary
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

----

60. [**SPRING: Studying the Paper and Reasoning to Play Games**](http://arxiv.org/pdf/2305.15486v3) by Yue Wu, Shrimai Prabhumoye, So Yeon Min, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Tom Mitchell, Yuanzhi Li
	## Summary
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

----

61. [**PromptBench: A Unified Library for Evaluation of Large Language Models**](http://arxiv.org/pdf/2312.07910v2) by Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie
	## Summary
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

----

62. [**ConsPrompt: Exploiting Contrastive Samples for Fewshot Prompt Learning**](http://arxiv.org/pdf/2211.04118v2) by Jinta Weng, Yifan Deng, d Donghao Li, Hao You, Yue Hu, Heyan Huang
	## Summary
	Prompt recently have become an effective linguistic tool on utilizing the
pre-trained language models. However, in few-shot scenarios, subtle changes of
prompt's design always make the result widely different, and the prompt design
is also easy to overfit the current limited samples. To alleviate this, we
explore how to utilize suitable contrastive samples and multiple contrastive
learning methods to realize a more robust prompt's representation. Therefore,
the contrastive prompt model ConsPrompt combining with prompt encoding network,
contrastive sampling modules, and contrastive scoring modules are introduced to
realize differential contrastive learning. Our results exhibit the
state-of-the-art performance in different few-shot settings, and the ablation
experiments also certificate the effectiveness in utilizing multi-degree
contrastive learning in prompt-based fine-tuning process.

----

63. [**Evaluating ChatGPT text-mining of clinical records for obesity monitoring**](http://arxiv.org/pdf/2308.01666v1) by Ivo S. Fins, Heather Davies, Sean Farrell, Jose R. Torres, Gina Pinchbeck, Alan D. Radford, Peter-John Noble
	## Summary
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

----

64. [**Plan, Eliminate, and Track -- Language Models are Good Teachers for Embodied Agents**](http://arxiv.org/pdf/2305.02412v2) by Yue Wu, So Yeon Min, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Yuanzhi Li, Tom Mitchell, Shrimai Prabhumoye
	## Summary
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

----

65. [**PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change**](http://arxiv.org/pdf/2206.10498v4) by Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, Subbarao Kambhampati
	## Summary
	Generating plans of action, and reasoning about change have long been
considered a core competence of intelligent agents. It is thus no surprise that
evaluating the planning and reasoning capabilities of large language models
(LLMs) has become a hot topic of research. Most claims about LLM planning
capabilities are however based on common sense tasks-where it becomes hard to
tell whether LLMs are planning or merely retrieving from their vast world
knowledge. There is a strong need for systematic and extensible planning
benchmarks with sufficient diversity to evaluate whether LLMs have innate
planning capabilities. Motivated by this, we propose PlanBench, an extensible
benchmark suite based on the kinds of domains used in the automated planning
community, especially in the International Planning Competition, to test the
capabilities of LLMs in planning or reasoning about actions and change.
PlanBench provides sufficient diversity in both the task domains and the
specific planning capabilities. Our studies also show that on many critical
capabilities-including plan generation-LLM performance falls quite short, even
with the SOTA models. PlanBench can thus function as a useful marker of
progress of LLMs in planning and reasoning.

----

66. [**Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**](http://arxiv.org/pdf/2310.01061v1) by Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, Shirui Pan
	## Summary
	Large language models (LLMs) have demonstrated impressive reasoning abilities
in complex tasks. However, they lack up-to-date knowledge and experience
hallucinations during reasoning, which can lead to incorrect reasoning
processes and diminish their performance and trustworthiness. Knowledge graphs
(KGs), which capture vast amounts of facts in a structured format, offer a
reliable source of knowledge for reasoning. Nevertheless, existing KG-based LLM
reasoning methods only treat KGs as factual knowledge bases and overlook the
importance of their structural information for reasoning. In this paper, we
propose a novel method called reasoning on graphs (RoG) that synergizes LLMs
with KGs to enable faithful and interpretable reasoning. Specifically, we
present a planning-retrieval-reasoning framework, where RoG first generates
relation paths grounded by KGs as faithful plans. These plans are then used to
retrieve valid reasoning paths from the KGs for LLMs to conduct faithful
reasoning. Furthermore, RoG not only distills knowledge from KGs to improve the
reasoning ability of LLMs through training but also allows seamless integration
with any arbitrary LLMs during inference. Extensive experiments on two
benchmark KGQA datasets demonstrate that RoG achieves state-of-the-art
performance on KG reasoning tasks and generates faithful and interpretable
reasoning results.

----

67. [**KwaiAgents: Generalized Information-seeking Agent System with Large Language Models**](http://arxiv.org/pdf/2312.04889v3) by Haojie Pan, Zepeng Zhai, Hao Yuan, Yaojia Lv, Ruiji Fu, Ming Liu, Zhongyuan Wang, Bing Qin
	## Summary
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

----

68. [**Agent Instructs Large Language Models to be General Zero-Shot Reasoners**](http://arxiv.org/pdf/2310.03710v1) by Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, Chenguang Wang
	## Summary
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

----

69. [**Exploring the intersection of Generative AI and Software Development**](http://arxiv.org/pdf/2312.14262v1) by Filipe Calegario, Vanilson Burégio, Francisco Erivaldo, Daniel Moraes Costa Andrade, Kailane Felix, Nathalia Barbosa, Pedro Lucas da Silva Lucena, César França
	## Summary
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

----

70. [**MLCopilot: Unleashing the Power of Large Language Models in Solving Machine Learning Tasks**](http://arxiv.org/pdf/2304.14979v1) by Lei Zhang, Yuge Zhang, Kan Ren, Dongsheng Li, Yuqing Yang
	## Summary
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

----

71. [**Sparks of Artificial General Intelligence: Early experiments with GPT-4**](http://arxiv.org/pdf/2303.12712v5) by Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang
	## Summary
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

----

72. [**Probabilistic Agent Programs**](http://arxiv.org/pdf/cs/9910016v1) by Juergen Dix, Mirco Nanni, VS Subrahmanian
	## Summary
	Agents are small programs that autonomously take actions based on changes in
their environment or ``state.'' Over the last few years, there have been an
increasing number of efforts to build agents that can interact and/or
collaborate with other agents. In one of these efforts, Eiter, Subrahmanian amd
Pick (AIJ, 108(1-2), pages 179-255) have shown how agents may be built on top
of legacy code. However, their framework assumes that agent states are
completely determined, and there is no uncertainty in an agent's state. Thus,
their framework allows an agent developer to specify how his agents will react
when the agent is 100% sure about what is true/false in the world state. In
this paper, we propose the concept of a \emph{probabilistic agent program} and
show how, given an arbitrary program written in any imperative language, we may
build a declarative ``probabilistic'' agent program on top of it which supports
decision making in the presence of uncertainty. We provide two alternative
semantics for probabilistic agent programs. We show that the second semantics,
though more epistemically appealing, is more complex to compute. We provide
sound and complete algorithms to compute the semantics of \emph{positive} agent
programs.

----

73. [**PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**](http://arxiv.org/pdf/2312.12456v1) by Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen
	## Summary
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

----

74. [**Beyond ChatBots: ExploreLLM for Structured Thoughts and Personalized Model Responses**](http://arxiv.org/pdf/2312.00763v1) by Xiao Ma, Swaroop Mishra, Ariel Liu, Sophie Su, Jilin Chen, Chinmay Kulkarni, Heng-Tze Cheng, Quoc Le, Ed Chi
	## Summary
	Large language model (LLM) powered chatbots are primarily text-based today,
and impose a large interactional cognitive load, especially for exploratory or
sensemaking tasks such as planning a trip or learning about a new city. Because
the interaction is textual, users have little scaffolding in the way of
structure, informational "scent", or ability to specify high-level preferences
or goals. We introduce ExploreLLM that allows users to structure thoughts, help
explore different options, navigate through the choices and recommendations,
and to more easily steer models to generate more personalized responses. We
conduct a user study and show that users find it helpful to use ExploreLLM for
exploratory or planning tasks, because it provides a useful schema-like
structure to the task, and guides users in planning. The study also suggests
that users can more easily personalize responses with high-level preferences
with ExploreLLM. Together, ExploreLLM points to a future where users interact
with LLMs beyond the form of chatbots, and instead designed to support complex
user tasks with a tighter integration between natural language and graphical
user interfaces.

----

75. [**Word sense disambiguation: a survey**](http://arxiv.org/pdf/1508.01346v1) by Alok Ranjan Pal, Diganta Saha
	## Summary
	In this paper, we made a survey on Word Sense Disambiguation (WSD). Near
about in all major languages around the world, research in WSD has been
conducted upto different extents. In this paper, we have gone through a survey
regarding the different approaches adopted in different research works, the
State of the Art in the performance in this domain, recent works in different
Indian languages and finally a survey in Bengali language. We have made a
survey on different competitions in this field and the bench mark results,
obtained from those competitions.

----

76. [**Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review**](http://arxiv.org/pdf/2311.01918v1) by Mingze Yuan, Peng Bao, Jiajia Yuan, Yunhao Shen, Zifan Chen, Yi Xie, Jie Zhao, Yang Chen, Li Zhang, Lin Shen, Bin Dong
	## Summary
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

----

77. [**Large Language Models Perform Diagnostic Reasoning**](http://arxiv.org/pdf/2307.08922v1) by Cheng-Kuang Wu, Wei-Lin Chen, Hsin-Hsi Chen
	## Summary
	We explore the extension of chain-of-thought (CoT) prompting to medical
reasoning for the task of automatic diagnosis. Motivated by doctors' underlying
reasoning process, we present Diagnostic-Reasoning CoT (DR-CoT). Empirical
results demonstrate that by simply prompting large language models trained only
on general text corpus with two DR-CoT exemplars, the diagnostic accuracy
improves by 15% comparing to standard prompting. Moreover, the gap reaches a
pronounced 18% in out-domain settings. Our findings suggest expert-knowledge
reasoning in large language models can be elicited through proper promptings.

----

78. [**Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves**](http://arxiv.org/pdf/2311.04205v1) by Yihe Deng, Weitong Zhang, Zixiang Chen, Quanquan Gu
	## Summary
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

----

79. [**Guiding Large Language Models via Directional Stimulus Prompting**](http://arxiv.org/pdf/2302.11520v4) by Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan
	## Summary
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

----

80. [**When Large Language Model based Agent Meets User Behavior Analysis: A Novel User Simulation Paradigm**](http://arxiv.org/pdf/2306.02552v2) by Lei Wang, Jingsen Zhang, Hao Yang, Zhiyuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, Ji-Rong Wen
	## Summary
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

----

81. [**Beyond Memorization: Violating Privacy Via Inference with Large Language Models**](http://arxiv.org/pdf/2310.07298v1) by Robin Staab, Mark Vero, Mislav Balunović, Martin Vechev
	## Summary
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

----

82. [**Creative Agents: Empowering Agents with Imagination for Creative Tasks**](http://arxiv.org/pdf/2312.02519v1) by Chi Zhang, Penglin Cai, Yuhui Fu, Haoqi Yuan, Zongqing Lu
	## Summary
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

----

83. [**Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning**](http://arxiv.org/pdf/2401.06805v1) by Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, Hongxia Yang
	## Summary
	Strong Artificial Intelligence (Strong AI) or Artificial General Intelligence
(AGI) with abstract reasoning ability is the goal of next-generation AI. Recent
advancements in Large Language Models (LLMs), along with the emerging field of
Multimodal Large Language Models (MLLMs), have demonstrated impressive
capabilities across a wide range of multimodal tasks and applications.
Particularly, various MLLMs, each with distinct model architectures, training
data, and training stages, have been evaluated across a broad range of MLLM
benchmarks. These studies have, to varying degrees, revealed different aspects
of the current capabilities of MLLMs. However, the reasoning abilities of MLLMs
have not been systematically investigated. In this survey, we comprehensively
review the existing evaluation protocols of multimodal reasoning, categorize
and illustrate the frontiers of MLLMs, introduce recent trends in applications
of MLLMs on reasoning-intensive tasks, and finally discuss current practices
and future directions. We believe our survey establishes a solid base and sheds
light on this important topic, multimodal reasoning.

----

84. [**The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4**](http://arxiv.org/pdf/2311.07361v2) by Microsoft Research AI4Science, Microsoft Azure Quantum
	## Summary
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

----

85. [**Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL**](http://arxiv.org/pdf/2309.06553v3) by Hao Sun, Alihan Hüyük, Mihaela van der Schaar
	## Summary
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

----

86. [**Transformer models: an introduction and catalog**](http://arxiv.org/pdf/2302.07730v3) by Xavier Amatriain, Ananth Sankar, Jie Bing, Praveen Kumar Bodigutla, Timothy J. Hazen, Michaeel Kazi
	## Summary
	In the past few years we have seen the meteoric appearance of dozens of
foundation models of the Transformer family, all of which have memorable and
sometimes funny, but not self-explanatory, names. The goal of this paper is to
offer a somewhat comprehensive but simple catalog and classification of the
most popular Transformer models. The paper also includes an introduction to the
most important aspects and innovations in Transformer models. Our catalog will
include models that are trained using self-supervised learning (e.g., BERT or
GPT3) as well as those that are further trained using a human-in-the-loop (e.g.
the InstructGPT model used by ChatGPT).

----

87. [**Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain**](http://arxiv.org/pdf/2307.03042v2) by Aryo Pradipta Gema, Luke Daines, Pasquale Minervini, Beatrice Alex
	## Summary
	Adapting pretrained language models to novel domains, such as clinical
applications, traditionally involves retraining their entire set of parameters.
However, this approach is increasingly proven to be impractical owing to the
substantial computational requirements associated with training such large
language models. To address this issue, Parameter-Efficient Fine-Tuning (PEFT)
techniques offer a viable solution by selectively fine-tuning a small subset of
additional parameters, significantly reducing the computational requirements
for domain adaptation. In this study, we propose Clinical LLaMA-LoRA, a PEFT
adapter layer built upon the open-sourced LLaMA model. Clinical LLaMA-LoRA is
trained using clinical notes obtained from the MIMIC-IV database, thereby
creating a specialised adapter designed for the clinical domain. Additionally,
we propose a two-step PEFT framework which fuses Clinical LLaMA-LoRA with
Downstream LLaMA-LoRA, another PEFT adapter specialised for downstream tasks.
We evaluate this framework on multiple clinical outcome prediction datasets,
comparing it to clinically trained language models. Our proposed framework
achieves a state-of-the-art AUROC score averaged across all clinical downstream
tasks. We observe substantial improvements of 6-9% AUROC score in the
large-scale multilabel classification tasks, such as diagnoses and procedures
classification.

----

88. [**OWL: A Large Language Model for IT Operations**](http://arxiv.org/pdf/2309.09298v1) by Hongcheng Guo, Jian Yang, Jiaheng Liu, Liqun Yang, Linzheng Chai, Jiaqi Bai, Junran Peng, Xiaorong Hu, Chao Chen, Dongfeng Zhang, Xu Shi, Tieqiao Zheng, Liangfan Zheng, Bo Zhang, Ke Xu, Zhoujun Li
	## Summary
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

----

89. [**Language Is Not All You Need: Aligning Perception with Language Models**](http://arxiv.org/pdf/2302.14045v2) by Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei
	## Summary
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

----

90. [**Generalized products and Lorentzian length spaces**](http://arxiv.org/pdf/2311.10691v1) by Elefterios Soultanis
	## Summary
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

----

91. [**LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models**](http://arxiv.org/pdf/2212.04088v3) by Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, Yu Su
	## Summary
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

----

92. [**TigerBot: An Open Multilingual Multitask LLM**](http://arxiv.org/pdf/2312.08688v2) by Ye Chen, Wei Cai, Liangmin Wu, Xiaowei Li, Zhanxuan Xin, Cong Fu
	## Summary
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

----

93. [**PaLM 2 Technical Report**](http://arxiv.org/pdf/2305.10403v3) by Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, Yonghui Wu
	## Summary
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

----

94. [**TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language Modeling Likewise**](http://arxiv.org/pdf/2310.19019v2) by Nan He, Hanyu Lai, Chenyang Zhao, Zirui Cheng, Junting Pan, Ruoyu Qin, Ruofan Lu, Rui Lu, Yunchen Zhang, Gangming Zhao, Zhaohui Hou, Zhiyuan Huang, Shaoqing Lu, Ding Liang, Mingjie Zhan
	## Summary
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

----

95. [**CogAgent: A Visual Language Model for GUI Agents**](http://arxiv.org/pdf/2312.08914v2) by Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, Jie Tang
	## Summary
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

----

96. [**Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks**](http://arxiv.org/pdf/2311.10609v1) by Benjamin Feuer, Chinmay Hegde, Niv Cohen
	## Summary
	Tabular classification has traditionally relied on supervised algorithms,
which estimate the parameters of a prediction model using its training data.
Recently, Prior-Data Fitted Networks (PFNs) such as TabPFN have successfully
learned to classify tabular data in-context: the model parameters are designed
to classify new samples based on labelled training samples given after the
model training. While such models show great promise, their applicability to
real-world data remains limited due to the computational scale needed. Here we
study the following question: given a pre-trained PFN for tabular data, what is
the best way to summarize the labelled training samples before feeding them to
the model? We conduct an initial investigation of sketching and
feature-selection methods for TabPFN, and note certain key differences between
it and conventionally fitted tabular models.

----

97. [**Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers**](http://arxiv.org/pdf/2309.08532v1) by Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang
	## Summary
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

----

98. [**Prompted Opinion Summarization with GPT-3.5**](http://arxiv.org/pdf/2211.15914v2) by Adithya Bhaskar, Alexander R. Fabbri, Greg Durrett
	## Summary
	Large language models have shown impressive performance across a wide variety
of tasks, including text summarization. In this paper, we show that this strong
performance extends to opinion summarization. We explore several pipeline
methods for applying GPT-3.5 to summarize a large collection of user reviews in
a prompted fashion. To handle arbitrarily large numbers of user reviews, we
explore recursive summarization as well as methods for selecting salient
content to summarize through supervised clustering or extraction. On two
datasets, an aspect-oriented summarization dataset of hotel reviews (SPACE) and
a generic summarization dataset of Amazon and Yelp reviews (FewSum), we show
that GPT-3.5 models achieve very strong performance in human evaluation. We
argue that standard evaluation metrics do not reflect this, and introduce three
new metrics targeting faithfulness, factuality, and genericity to contrast
these different methods.

----

99. [**Chain-of-Thought Tuning: Masked Language Models can also Think Step By Step in Natural Language Understanding**](http://arxiv.org/pdf/2310.11721v1) by Caoyun Fan, Jidong Tian, Yitian Li, Wenqing Chen, Hao He, Yaohui Jin
	## Summary
	Chain-of-Thought (CoT) is a technique that guides Large Language Models
(LLMs) to decompose complex tasks into multi-step reasoning through
intermediate steps in natural language form. Briefly, CoT enables LLMs to think
step by step. However, although many Natural Language Understanding (NLU) tasks
also require thinking step by step, LLMs perform less well than small-scale
Masked Language Models (MLMs). To migrate CoT from LLMs to MLMs, we propose
Chain-of-Thought Tuning (CoTT), a two-step reasoning framework based on prompt
tuning, to implement step-by-step thinking for MLMs on NLU tasks. From the
perspective of CoT, CoTT's two-step framework enables MLMs to implement task
decomposition; CoTT's prompt tuning allows intermediate steps to be used in
natural language form. Thereby, the success of CoT can be extended to NLU tasks
through MLMs. To verify the effectiveness of CoTT, we conduct experiments on
two NLU tasks: hierarchical classification and relation extraction, and the
results show that CoTT outperforms baselines and achieves state-of-the-art
performance.

----

100. [**Large Language Models Can Be Easily Distracted by Irrelevant Context**](http://arxiv.org/pdf/2302.00093v3) by Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, Denny Zhou
	## Summary
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

----

101. [**Reasoning Implicit Sentiment with Chain-of-Thought Prompting**](http://arxiv.org/pdf/2305.11255v4) by Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua
	## Summary
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

----

102. [**How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation**](http://arxiv.org/pdf/2312.17115v1) by Yang Xiao, Yi Cheng, Jinlan Fu, Jiashuo Wang, Wenjie Li, Pengfei Liu
	## Summary
	Human behavior simulation of AI agents necessitates the agents to possess a
quality of believability, which is crucial as it facilitates users in
establishing trust toward the agents and streamlines the fulfillment of the
agents' goal. While recent advancements in Large Language Model (LLM) based
agents have improved human behavior simulation, challenges inherent to LLMs
(e.g., long context modeling) can undermine their believability. Consequently,
evaluating AI agent believability becomes imperative. Unfortunately, prior
research often neglects the negative impacts of LLM deficiencies. To address
these gaps, we introduce two metrics for assessing LLM-based agent
believability: consistency, and robustness, together with a benchmark,
SimulateBench, with which, we evaluate the consistency and robustness of agents
implemented with popular LLMs. We find that agents (i) struggle to accurately
depict character information when presented with lengthy profile inputs; (ii)
exhibit vulnerability to profile perturbations; and (iii) are significantly
affected by certain key factors that impact their overall believability. Code
and SimulateBench are public at https://github.com/GAIR-NLP/GPTMan.

----

103. [**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**](http://arxiv.org/pdf/2310.11511v1) by Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
	## Summary
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

----

104. [**Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters**](http://arxiv.org/pdf/2212.10001v2) by Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, Huan Sun
	## Summary
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

----

105. [**Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**](http://arxiv.org/pdf/2304.01373v2) by Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, Oskar van der Wal
	## Summary
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

----

106. [**PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting**](http://arxiv.org/pdf/2210.08964v5) by Hao Xue, Flora D. Salim
	## Summary
	This paper presents a new perspective on time series forecasting. In existing
time series forecasting methods, the models take a sequence of numerical values
as input and yield numerical values as output. The existing SOTA models are
largely based on the Transformer architecture, modified with multiple encoding
mechanisms to incorporate the context and semantics around the historical data.
Inspired by the successes of pre-trained language foundation models, we pose a
question about whether these models can also be adapted to solve time-series
forecasting. Thus, we propose a new forecasting paradigm: prompt-based time
series forecasting (PromptCast). In this novel task, the numerical input and
output are transformed into prompts and the forecasting task is framed in a
sentence-to-sentence manner, making it possible to directly apply language
models for forecasting purposes. To support and facilitate the research of this
task, we also present a large-scale dataset (PISA) that includes three
real-world forecasting scenarios. We evaluate different SOTA numerical-based
forecasting methods and language generation models. The benchmark results with
various forecasting settings demonstrate the proposed PromptCast with language
generation models is a promising research direction. Additionally, in
comparison to conventional numerical-based forecasting, PromptCast shows a much
better generalization ability under the zero-shot setting.

----

107. [**The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only**](http://arxiv.org/pdf/2306.01116v1) by Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay
	## Summary
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

----

108. [**SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot**](http://arxiv.org/pdf/2301.00774v3) by Elias Frantar, Dan Alistarh
	## Summary
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

----

109. [**Extending Context Window of Large Language Models via Semantic Compression**](http://arxiv.org/pdf/2312.09571v1) by Weizhi Fei, Xueyan Niu, Pingyi Zhou, Lu Hou, Bo Bai, Lei Deng, Wei Han
	## Summary
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

----

110. [**Robust fine-tuning of zero-shot models**](http://arxiv.org/pdf/2109.01903v3) by Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt
	## Summary
	Large pre-trained models such as CLIP or ALIGN offer consistent accuracy
across a range of data distributions when performing zero-shot inference (i.e.,
without fine-tuning on a specific dataset). Although existing fine-tuning
methods substantially improve accuracy on a given target distribution, they
often reduce robustness to distribution shifts. We address this tension by
introducing a simple and effective method for improving robustness while
fine-tuning: ensembling the weights of the zero-shot and fine-tuned models
(WiSE-FT). Compared to standard fine-tuning, WiSE-FT provides large accuracy
improvements under distribution shift, while preserving high accuracy on the
target distribution. On ImageNet and five derived distribution shifts, WiSE-FT
improves accuracy under distribution shift by 4 to 6 percentage points (pp)
over prior work while increasing ImageNet accuracy by 1.6 pp. WiSE-FT achieves
similarly large robustness gains (2 to 23 pp) on a diverse set of six further
distribution shifts, and accuracy gains of 0.8 to 3.3 pp compared to standard
fine-tuning on seven commonly used transfer learning datasets. These
improvements come at no additional computational cost during fine-tuning or
inference.

----

111. [**Textbooks Are All You Need II: phi-1.5 technical report**](http://arxiv.org/pdf/2309.05463v1) by Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee
	## Summary
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

----

112. [**Faithfulness-Aware Decoding Strategies for Abstractive Summarization**](http://arxiv.org/pdf/2303.03278v1) by David Wan, Mengwen Liu, Kathleen McKeown, Markus Dreyer, Mohit Bansal
	## Summary
	Despite significant progress in understanding and improving faithfulness in
abstractive summarization, the question of how decoding strategies affect
faithfulness is less studied. We present a systematic study of the effect of
generation techniques such as beam search and nucleus sampling on faithfulness
in abstractive summarization. We find a consistent trend where beam search with
large beam sizes produces the most faithful summaries while nucleus sampling
generates the least faithful ones. We propose two faithfulness-aware generation
methods to further improve faithfulness over current generation techniques: (1)
ranking candidates generated by beam search using automatic faithfulness
metrics and (2) incorporating lookahead heuristics that produce a faithfulness
score on the future summary. We show that both generation methods significantly
improve faithfulness across two datasets as evaluated by four automatic
faithfulness metrics and human evaluation. To reduce computational cost, we
demonstrate a simple distillation approach that allows the model to generate
faithful summaries with just greedy decoding. Our code is publicly available at
https://github.com/amazon-science/faithful-summarization-generation

----

113. [**QLoRA: Efficient Finetuning of Quantized LLMs**](http://arxiv.org/pdf/2305.14314v1) by Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
	## Summary
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

----

114. [**Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4**](http://arxiv.org/pdf/2312.16171v1) by Sondos Mahmoud Bsharat, Aidar Myrzakhan, Zhiqiang Shen
	## Summary
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

----

115. [**Code Llama: Open Foundation Models for Code**](http://arxiv.org/pdf/2308.12950v2) by Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve
	## Summary
	We release Code Llama, a family of large language models for code based on
Llama 2 providing state-of-the-art performance among open models, infilling
capabilities, support for large input contexts, and zero-shot instruction
following ability for programming tasks. We provide multiple flavors to cover a
wide range of applications: foundation models (Code Llama), Python
specializations (Code Llama - Python), and instruction-following models (Code
Llama - Instruct) with 7B, 13B and 34B parameters each. All models are trained
on sequences of 16k tokens and show improvements on inputs with up to 100k
tokens. 7B and 13B Code Llama and Code Llama - Instruct variants support
infilling based on surrounding content. Code Llama reaches state-of-the-art
performance among open models on several code benchmarks, with scores of up to
53% and 55% on HumanEval and MBPP, respectively. Notably, Code Llama - Python
7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform
every other publicly available model on MultiPL-E. We release Code Llama under
a permissive license that allows for both research and commercial use.

----

116. [**Less Likely Brainstorming: Using Language Models to Generate Alternative Hypotheses**](http://arxiv.org/pdf/2305.19339v1) by Liyan Tang, Yifan Peng, Yanshan Wang, Ying Ding, Greg Durrett, Justin F. Rousseau
	## Summary
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

----

117. [**You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content**](http://arxiv.org/pdf/2308.05596v1) by Xinlei He, Savvas Zannettou, Yun Shen, Yang Zhang
	## Summary
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

----

118. [**Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning**](http://arxiv.org/pdf/2301.13808v3) by Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, Yongbin Li
	## Summary
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

----

119. [**Fairness Evaluation in Text Classification: Machine Learning Practitioner Perspectives of Individual and Group Fairness**](http://arxiv.org/pdf/2303.00673v1) by Zahra Ashktorab, Benjamin Hoover, Mayank Agarwal, Casey Dugan, Werner Geyer, Hao Bang Yang, Mikhail Yurochkin
	## Summary
	Mitigating algorithmic bias is a critical task in the development and
deployment of machine learning models. While several toolkits exist to aid
machine learning practitioners in addressing fairness issues, little is known
about the strategies practitioners employ to evaluate model fairness and what
factors influence their assessment, particularly in the context of text
classification. Two common approaches of evaluating the fairness of a model are
group fairness and individual fairness. We run a study with Machine Learning
practitioners (n=24) to understand the strategies used to evaluate models.
Metrics presented to practitioners (group vs. individual fairness) impact which
models they consider fair. Participants focused on risks associated with
underpredicting/overpredicting and model sensitivity relative to identity token
manipulations. We discover fairness assessment strategies involving personal
experiences or how users form groups of identity tokens to test model fairness.
We provide recommendations for interactive tools for evaluating fairness in
text classification.

----

120. [**RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation**](http://arxiv.org/pdf/2310.04408v1) by Fangyuan Xu, Weijia Shi, Eunsol Choi
	## Summary
	Retrieving documents and prepending them in-context at inference time
improves performance of language model (LMs) on a wide range of tasks. However,
these documents, often spanning hundreds of words, make inference substantially
more expensive. We propose compressing the retrieved documents into textual
summaries prior to in-context integration. This not only reduces the
computational costs but also relieves the burden of LMs to identify relevant
information in long retrieved documents. We present two compressors -- an
extractive compressor which selects useful sentences from retrieved documents
and an abstractive compressor which generates summaries by synthesizing
information from multiple documents. Both compressors are trained to improve
LMs' performance on end tasks when the generated summaries are prepended to the
LMs' input, while keeping the summary concise.If the retrieved documents are
irrelevant to the input or offer no additional information to LM, our
compressor can return an empty string, implementing selective augmentation.We
evaluate our approach on language modeling task and open domain question
answering task. We achieve a compression rate of as low as 6% with minimal loss
in performance for both tasks, significantly outperforming the off-the-shelf
summarization models. We show that our compressors trained for one LM can
transfer to other LMs on the language modeling task and provide summaries
largely faithful to the retrieved documents.

----

121. [**Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**](http://arxiv.org/pdf/2310.13191v3) by Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu
	## Summary
	The pruning objective has recently extended beyond accuracy and sparsity to
robustness in language models. Despite this, existing methods struggle to
enhance robustness against adversarial attacks when continually increasing
model sparsity and require a retraining process. As humans step into the era of
large language models, these issues become increasingly prominent. This paper
proposes that the robustness of language models is proportional to the extent
of pre-trained knowledge they encompass. Accordingly, we introduce a
post-training pruning strategy designed to faithfully replicate the embedding
space and feature space of dense language models, aiming to conserve more
pre-trained knowledge during the pruning process. In this setup, each layer's
reconstruction error not only originates from itself but also includes
cumulative error from preceding layers, followed by an adaptive rectification.
Compared to other state-of-art baselines, our approach demonstrates a superior
balance between accuracy, sparsity, robustness, and pruning cost with BERT on
datasets SST2, IMDB, and AGNews, marking a significant stride towards robust
pruning in language models.

----

122. [**EcoAssistant: Using LLM Assistant More Affordably and Accurately**](http://arxiv.org/pdf/2310.03046v1) by Jieyu Zhang, Ranjay Krishna, Ahmed H. Awadallah, Chi Wang
	## Summary
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

----

123. [**GPT-4 Vision on Medical Image Classification -- A Case Study on COVID-19 Dataset**](http://arxiv.org/pdf/2310.18498v1) by Ruibo Chen, Tianyi Xiong, Yihan Wu, Guodong Liu, Zhengmian Hu, Lichang Chen, Yanshuo Chen, Chenxi Liu, Heng Huang
	## Summary
	This technical report delves into the application of GPT-4 Vision (GPT-4V) in
the nuanced realm of COVID-19 image classification, leveraging the
transformative potential of in-context learning to enhance diagnostic
processes.

----

124. [**Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams**](http://arxiv.org/pdf/2310.08678v1) by Ethan Callanan, Amarachi Mbakwe, Antony Papadimitriou, Yulong Pei, Mathieu Sibue, Xiaodan Zhu, Zhiqiang Ma, Xiaomo Liu, Sameena Shah
	## Summary
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

----

125. [**Can large language models provide useful feedback on research papers? A large-scale empirical analysis**](http://arxiv.org/pdf/2310.01783v1) by Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Smith, Yian Yin, Daniel McFarland, James Zou
	## Summary
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

----

126. [**Natural Language Reasoning, A Survey**](http://arxiv.org/pdf/2303.14725v2) by Fei Yu, Hongbo Zhang, Prayag Tiwari, Benyou Wang
	## Summary
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

----

127. [**MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning**](http://arxiv.org/pdf/2311.10537v1) by Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, Mark Gerstein
	## Summary
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

----

128. [**WebArena: A Realistic Web Environment for Building Autonomous Agents**](http://arxiv.org/pdf/2307.13854v3) by Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig
	## Summary
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

----

129. [**Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models**](http://arxiv.org/pdf/2306.03799v1) by Fobo Shi, Peijun Qing, Dong Yang, Nan Wang, Youbo Lei, Haonan Lu, Xiaodong Lin
	## Summary
	Prompt engineering is an essential technique for enhancing the abilities of
large language models (LLMs) by providing explicit and specific instructions.
It enables LLMs to excel in various tasks, such as arithmetic reasoning,
question answering, summarization, relation extraction, machine translation,
and sentiment analysis. Researchers have been actively exploring different
prompt engineering strategies, such as Chain of Thought (CoT), Zero-CoT, and
In-context learning. However, an unresolved problem arises from the fact that
current approaches lack a solid theoretical foundation for determining optimal
prompts. To address this issue in prompt engineering, we propose a new and
effective approach called Prompt Space. Our methodology utilizes text
embeddings to obtain basis vectors by matrix decomposition, and then constructs
a space for representing all prompts. Prompt Space significantly outperforms
state-of-the-art prompt paradigms on ten public reasoning benchmarks. Notably,
without the help of the CoT method and the prompt "Let's think step by step",
Prompt Space shows superior performance over the few-shot method. Overall, our
approach provides a robust and fundamental theoretical framework for selecting
simple and effective prompts. This advancement marks a significant step towards
improving prompt engineering for a wide variety of applications in LLMs.

----

130. [**Prompt to be Consistent is Better than Self-Consistent? Few-Shot and Zero-Shot Fact Verification with Pre-trained Language Models**](http://arxiv.org/pdf/2306.02569v1) by Fengzhu Zeng, Wei Gao
	## Summary
	Few-shot or zero-shot fact verification only relies on a few or no labeled
training examples. In this paper, we propose a novel method called ProToCo, to
\underline{Pro}mpt pre-trained language models (PLMs) \underline{To} be
\underline{Co}nsistent, for improving the factuality assessment capability of
PLMs in the few-shot and zero-shot settings. Given a claim-evidence pair,
ProToCo generates multiple variants of the claim with different relations and
frames a simple consistency mechanism as constraints for making compatible
predictions across these variants. We update PLMs by using parameter-efficient
fine-tuning (PEFT), leading to more accurate predictions in few-shot and
zero-shot fact verification tasks. Our experiments on three public verification
datasets show that ProToCo significantly outperforms state-of-the-art few-shot
fact verification baselines. With a small number of unlabeled instances,
ProToCo also outperforms the strong zero-shot learner T0 on zero-shot
verification. Compared to large PLMs using in-context learning (ICL) method,
ProToCo outperforms OPT-30B and the Self-Consistency-enabled OPT-6.7B model in
both few- and zero-shot settings.

----

131. [**A Survey of Large Language Models in Medicine: Principles, Applications, and Challenges**](http://arxiv.org/pdf/2311.05112v2) by Hongjian Zhou, Fenglin Liu, Boyang Gu, Xinyu Zou, Jinfa Huang, Jinge Wu, Yiru Li, Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Xian Wu, Yefeng Zheng, Lei Clifton, Zheng Li, Jiebo Luo, David A. Clifton
	## Summary
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

----

132. [**A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT**](http://arxiv.org/pdf/2302.11382v1) by Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt
	## Summary
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

----

133. [**LLaMA Pro: Progressive LLaMA with Block Expansion**](http://arxiv.org/pdf/2401.02415v1) by Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, Ying Shan
	## Summary
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

----

134. [**Large Language Models for Generative Information Extraction: A Survey**](http://arxiv.org/pdf/2312.17617v1) by Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Enhong Chen
	## Summary
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

----

135. [**Large Language Model Alignment: A Survey**](http://arxiv.org/pdf/2309.15025v1) by Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong
	## Summary
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

----

136. [**The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models**](http://arxiv.org/pdf/2305.14999v2) by Jingyuan Qi, Zhiyang Xu, Ying Shen, Minqian Liu, Di Jin, Qifan Wang, Lifu Huang
	## Summary
	Chain-of-Thought (CoT) prompting enables large language models to solve
complex reasoning problems by generating intermediate steps. However, confined
by its inherent single-pass and sequential generation process, CoT heavily
relies on the initial decisions, causing errors in early steps to accumulate
and impact the final answers. In contrast, humans adopt recursive thinking when
tackling complex reasoning problems, i.e., iteratively breaking the original
problem into approachable sub-problems and aggregating their answers to resolve
the original one. Inspired by the human cognitive process, we propose SOCRATIC
QUESTIONING, a divide-and-conquer style algorithm that mimics the recursive
thinking process. Specifically, SOCRATIC QUESTIONING leverages large language
models to raise and answer sub-questions until collecting enough information to
tackle the original question. Unlike CoT, SOCRATIC QUESTIONING explicitly
navigates the thinking space, stimulates effective recursive thinking, and is
more robust towards errors in the thinking process. Extensive experiments on
several complex reasoning tasks, including MMLU, MATH, LogiQA, and visual
question-answering demonstrate significant performance improvements over the
state-of-the-art prompting methods, such as CoT, and Tree-of-Thought. The
qualitative analysis clearly shows that the intermediate reasoning steps
elicited by SOCRATIC QUESTIONING are similar to humans' recursively thinking
process of complex reasoning problems.

----

137. [**Large Language Models Perform Diagnostic Reasoning**](http://arxiv.org/pdf/2307.08922v1) by Cheng-Kuang Wu, Wei-Lin Chen, Hsin-Hsi Chen
	## Summary
	We explore the extension of chain-of-thought (CoT) prompting to medical
reasoning for the task of automatic diagnosis. Motivated by doctors' underlying
reasoning process, we present Diagnostic-Reasoning CoT (DR-CoT). Empirical
results demonstrate that by simply prompting large language models trained only
on general text corpus with two DR-CoT exemplars, the diagnostic accuracy
improves by 15% comparing to standard prompting. Moreover, the gap reaches a
pronounced 18% in out-domain settings. Our findings suggest expert-knowledge
reasoning in large language models can be elicited through proper promptings.

----

138. [**Tracking and managing deemed abilities**](http://arxiv.org/pdf/2104.14892v1) by Nicolas Troquard
	## Summary
	Information about the powers and abilities of acting entities is used to
coordinate their actions in societies, either physical or digital. Yet, the
commonsensical meaning of an acting entity being deemed able to do something is
still missing from the existing specification languages for the web or for
multi-agent systems. We advance a general purpose abstract logical account of
evidence-based ability. A basic model can be thought of as the ongoing trace of
a multi-agent system. Every state records systemic confirmations and
disconfirmations of whether an acting entity is able to bring about something.
Qualitative inductive reasoning is then used in order to infer what acting
entities are deemed able to bring about in the multi-agent system. A
temporalised modal language is used to talk about deemed ability, actual
agency, and confirmation and disconfirmation of deemed ability. What
constitutes a confirmation and a disconfirmation is left to the modeller as in
general it depends on the application at hand. So to illustrate the methodology
we propose two extended examples, one in practical philosophy, the other in
system engineering. We first use a logic of agency and ability to obtain a
version of Mele's general practical abilities. Then, we look at the management
of abilities in a supervised system.

----

139. [**Experiential Explanations for Reinforcement Learning**](http://arxiv.org/pdf/2210.04723v4) by Amal Alabdulkarim, Madhuri Singh, Gennie Mansi, Kaely Hall, Mark O. Riedl
	## Summary
	Reinforcement Learning (RL) systems can be complex and non-interpretable,
making it challenging for non-AI experts to understand or intervene in their
decisions. This is due in part to the sequential nature of RL in which actions
are chosen because of future rewards. However, RL agents discard the
qualitative features of their training, making it difficult to recover
user-understandable information for "why" an action is chosen. We propose a
technique, Experiential Explanations, to generate counterfactual explanations
by training influence predictors along with the RL policy. Influence predictors
are models that learn how sources of reward affect the agent in different
states, thus restoring information about how the policy reflects the
environment. A human evaluation study revealed that participants presented with
experiential explanations were better able to correctly guess what an agent
would do than those presented with other standard types of explanation.
Participants also found that experiential explanations are more understandable,
satisfying, complete, useful, and accurate. The qualitative analysis provides
insights into the factors of experiential explanations that are most useful.

----

140. [**Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil Cognitive Depth in LLMs**](http://arxiv.org/pdf/2312.17080v1) by Zhongshen Zeng, Pengguang Chen, Haiyun Jiang, Jiaya Jia
	## Summary
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

----

141. [**Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals**](http://arxiv.org/pdf/2302.04449v3) by Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom M. Mitchell
	## Summary
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

----

142. [**Catwalk: A Unified Language Model Evaluation Framework for Many Datasets**](http://arxiv.org/pdf/2312.10253v1) by Dirk Groeneveld, Anas Awadalla, Iz Beltagy, Akshita Bhagia, Ian Magnusson, Hao Peng, Oyvind Tafjord, Pete Walsh, Kyle Richardson, Jesse Dodge
	## Summary
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

----

143. [**SM70: A Large Language Model for Medical Devices**](http://arxiv.org/pdf/2312.06974v1) by Anubhav Bhatti, Surajsinh Parmar, San Lee
	## Summary
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

----

144. [**From Human Days to Machine Seconds: Automatically Answering and Generating Machine Learning Final Exams**](http://arxiv.org/pdf/2206.05442v7) by Iddo Drori, Sarah J. Zhang, Reece Shuttleworth, Sarah Zhang, Keith Tyser, Zad Chin, Pedro Lantigua, Saisamrit Surbehera, Gregory Hunter, Derek Austin, Leonard Tang, Yann Hicke, Sage Simhon, Sathwik Karnik, Darnell Granberry, Madeleine Udell
	## Summary
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

----

145. [**MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices**](http://arxiv.org/pdf/2312.16886v2) by Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen
	## Summary
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

----

146. [**LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**](http://arxiv.org/pdf/2303.16199v2) by Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Yu Qiao
	## Summary
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

----

147. [**Large Language Models in the Workplace: A Case Study on Prompt Engineering for Job Type Classification**](http://arxiv.org/pdf/2303.07142v3) by Benjamin Clavié, Alexandru Ciceu, Frederick Naylor, Guillaume Soulié, Thomas Brightwell
	## Summary
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

----

148. [**Chain-Of-Thought Prompting Under Streaming Batch: A Case Study**](http://arxiv.org/pdf/2306.00550v1) by Yuxin Tang
	## Summary
	Recently, Large Language Models (LLMs) have demonstrated remarkable
capabilities. Chain-of-Thought (CoT) has been proposed as a way of assisting
LLMs in performing complex reasoning. However, developing effective prompts can
be a challenging and labor-intensive task. Many studies come out of some way to
automatically construct CoT from test data. Most of them assume that all test
data is visible before testing and only select a small subset to generate
rationales, which is an unrealistic assumption. In this paper, we present a
case study on how to construct and optimize chain-of-thought prompting using
batch data in streaming settings.

----

149. [**Adapting Large Language Models for Document-Level Machine Translation**](http://arxiv.org/pdf/2401.06468v1) by Minghao Wu, Thuy-Trang Vu, Lizhen Qu, George Foster, Gholamreza Haffari
	## Summary
	Large language models (LLMs) have made significant strides in various natural
language processing (NLP) tasks. Recent research shows that the
moderately-sized LLMs often outperform their larger counterparts after
task-specific fine-tuning. In this work, we delve into the process of adapting
LLMs to specialize in document-level machine translation (DocMT) for a specific
language pair. Firstly, we explore how prompt strategies affect downstream
translation performance. Then, we conduct extensive experiments with two
fine-tuning methods, three LLM backbones, and 18 translation tasks across nine
language pairs. Our findings indicate that in some cases, these specialized
models even surpass GPT-4 in translation performance, while they still
significantly suffer from the off-target translation issue in others, even if
they are exclusively fine-tuned on bilingual parallel documents. Furthermore,
we provide an in-depth analysis of these LLMs tailored for DocMT, exploring
aspects such as translation errors, the scaling law of parallel documents,
out-of-domain generalization, and the impact of zero-shot crosslingual
transfer. The findings of this research not only shed light on the strengths
and limitations of LLM-based DocMT models but also provide a foundation for
future research in DocMT.

----

150. [**LIMA: Less Is More for Alignment**](http://arxiv.org/pdf/2305.11206v1) by Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
	## Summary
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

----

151. [**LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents**](http://arxiv.org/pdf/2311.05437v1) by Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, Chunyuan Li
	## Summary
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

----

152. [**Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning**](http://arxiv.org/pdf/2303.02861v1) by Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, Yoon Kim
	## Summary
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

----

153. [**A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis**](http://arxiv.org/pdf/2307.12856v3) by Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, Aleksandra Faust
	## Summary
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

----

154. [**The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning**](http://arxiv.org/pdf/2305.14045v2) by Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, Minjoon Seo
	## Summary
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

----

155. [**Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data**](http://arxiv.org/pdf/2304.01196v4) by Canwen Xu, Daya Guo, Nan Duan, Julian McAuley
	## Summary
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

----

156. [**Experiences with Remote Examination Formats in Light of GPT-4**](http://arxiv.org/pdf/2305.02198v1) by Felix Dobslaw, Peter Bergh
	## Summary
	Sudden access to the rapidly improving large language model GPT by open-ai
forces educational institutions worldwide to revisit their exam procedures. In
the pre-GPT era, we successfully applied oral and open-book home exams for two
courses in the third year of our predominantly remote Software Engineering BSc
program. We ask in this paper whether our current open-book exams are still
viable or whether a move back to a legally compliant but less scalable oral
exam is the only workable alternative. We further compare work-effort estimates
between oral and open-book exams and report on differences in throughput and
grade distribution over eight years to better understand the impact of
examination format on the outcome. Examining GPT v4 on the most recent
open-book exams showed that our current Artificial Intelligence and Reactive
Programming exams are not GPT v4 proof. Three potential weaknesses of GPT are
outlined. We also found that grade distributions have largely been unaffected
by the examination format, opening up for a move to oral examinations only if
needed. Throughput was higher for open-book exam course instances (73% vs 64%),
while fail rates were too (12% vs 7%), with teacher workload increasing even
for smaller classes. We also report on our experience regarding effort. Oral
examinations are efficient for smaller groups but come with caveats regarding
intensity and stress.

----

157. [**Towards Verifiable Text Generation with Symbolic References**](http://arxiv.org/pdf/2311.09188v1) by Lucas Torroba Hennigen, Shannon Shen, Aniruddha Nrusimha, Bernhard Gapp, David Sontag, Yoon Kim
	## Summary
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

----

158. [**A Survey of Reasoning with Foundation Models**](http://arxiv.org/pdf/2312.11562v4) by Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, Yue Wu, Wenhai Wang, Junsong Chen, Zhangyue Yin, Xiaozhe Ren, Jie Fu, Junxian He, Wu Yuan, Qi Liu, Xihui Liu, Yu Li, Hao Dong, Yu Cheng, Ming Zhang, Pheng Ann Heng, Jifeng Dai, Ping Luo, Jingdong Wang, Ji-Rong Wen, Xipeng Qiu, Yike Guo, Hui Xiong, Qun Liu, Zhenguo Li
	## Summary
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

----

159. [**ALCUNA: Large Language Models Meet New Knowledge**](http://arxiv.org/pdf/2310.14820v1) by Xunjian Yin, Baizhou Huang, Xiaojun Wan
	## Summary
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

----

160. [**ADaPT: As-Needed Decomposition and Planning with Language Models**](http://arxiv.org/pdf/2311.05772v1) by Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, Tushar Khot
	## Summary
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

----

161. [**Entity and Evidence Guided Relation Extraction for DocRED**](http://arxiv.org/pdf/2008.12283v1) by Kevin Huang, Guangtao Wang, Tengyu Ma, Jing Huang
	## Summary
	Document-level relation extraction is a challenging task which requires
reasoning over multiple sentences in order to predict relations in a document.
In this paper, we pro-pose a joint training frameworkE2GRE(Entity and Evidence
Guided Relation Extraction)for this task. First, we introduce entity-guided
sequences as inputs to a pre-trained language model (e.g. BERT, RoBERTa). These
entity-guided sequences help a pre-trained language model (LM) to focus on
areas of the document related to the entity. Secondly, we guide the fine-tuning
of the pre-trained language model by using its internal attention probabilities
as additional features for evidence prediction.Our new approach encourages the
pre-trained language model to focus on the entities and supporting/evidence
sentences. We evaluate our E2GRE approach on DocRED, a recently released
large-scale dataset for relation extraction. Our approach is able to achieve
state-of-the-art results on the public leaderboard across all metrics, showing
that our E2GRE is both effective and synergistic on relation extraction and
evidence prediction.

----

162. [**Automatic Prompt Optimization with "Gradient Descent" and Beam Search**](http://arxiv.org/pdf/2305.03495v2) by Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng
	## Summary
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

----

163. [**Teaching Probabilistic Logical Reasoning to Transformers**](http://arxiv.org/pdf/2305.13179v1) by Aliakbar Nafar, Kristen Brent Venable, Parisa Kordjamshidi
	## Summary
	Recent research on transformer-based language models investigates their
reasoning ability over logical rules expressed in natural language text.
However, their logic is not yet well-understood as we cannot explain the
abstractions made by the models that help them in reasoning. These models are
criticized for merely memorizing complex patterns in the data, which often
creates issues for their generalizability in unobserved situations. In this
work, we analyze the use of probabilistic logical rules in transformer-based
language models. In particular, we propose a new approach, Probabilistic
Constraint Training (PCT), that explicitly models probabilistic logical
reasoning by imposing the rules of reasoning as constraints during training. We
create a new QA benchmark for evaluating probabilistic reasoning over uncertain
textual rules, which creates instance-specific rules, unlike the only existing
relevant benchmark. Experimental results show that our proposed technique
improves the base language models' accuracy and explainability when
probabilistic logical reasoning is required for question answering. Moreover,
we show that the learned probabilistic reasoning abilities are transferable to
novel situations.

----

164. [**Response: Emergent analogical reasoning in large language models**](http://arxiv.org/pdf/2308.16118v1) by Damian Hodel, Jevin West
	## Summary
	In their recent Nature Human Behaviour paper, "Emergent analogical reasoning
in large language models," (Webb, Holyoak, and Lu, 2023) the authors argue that
"large language models such as GPT-3 have acquired an emergent ability to find
zero-shot solutions to a broad range of analogy problems." In this response, we
provide counterexamples of the letter string analogies. In our tests, GPT-3
fails to solve even the easiest variants of the problems presented in the
original paper. Zero-shot reasoning is an extraordinary claim that requires
extraordinary evidence. We do not see that evidence in our experiments. To
strengthen claims of humanlike reasoning such as zero-shot reasoning, it is
important that the field develop approaches that rule out data memorization.

----

165. [**The Rise and Potential of Large Language Model Based Agents: A Survey**](http://arxiv.org/pdf/2309.07864v3) by Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui
	## Summary
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

----

166. [**Multimodal Automated Fact-Checking: A Survey**](http://arxiv.org/pdf/2305.13507v3) by Mubashara Akhtar, Michael Schlichtkrull, Zhijiang Guo, Oana Cocarascu, Elena Simperl, Andreas Vlachos
	## Summary
	Misinformation is often conveyed in multiple modalities, e.g. a miscaptioned
image. Multimodal misinformation is perceived as more credible by humans, and
spreads faster than its text-only counterparts. While an increasing body of
research investigates automated fact-checking (AFC), previous surveys mostly
focus on text. In this survey, we conceptualise a framework for AFC including
subtasks unique to multimodal misinformation. Furthermore, we discuss related
terms used in different communities and map them to our framework. We focus on
four modalities prevalent in real-world fact-checking: text, image, audio, and
video. We survey benchmarks and models, and discuss limitations and promising
directions for future research

----

167. [**TinyStories: How Small Can Language Models Be and Still Speak Coherent English?**](http://arxiv.org/pdf/2305.07759v2) by Ronen Eldan, Yuanzhi Li
	## Summary
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

----

168. [**Contrastive Chain-of-Thought Prompting**](http://arxiv.org/pdf/2311.09277v1) by Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing
	## Summary
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

----

169. [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](http://arxiv.org/pdf/2303.11381v1) by Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, Lijuan Wang
	## Summary
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

----

170. [**Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code**](http://arxiv.org/pdf/2311.07989v3) by Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, Rui Wang
	## Summary
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

----

171. [**LLM4Jobs: Unsupervised occupation extraction and standardization leveraging Large Language Models**](http://arxiv.org/pdf/2309.09708v2) by Nan Li, Bo Kang, Tijl De Bie
	## Summary
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

----

172. [**ART: Automatic multi-step reasoning and tool-use for large language models**](http://arxiv.org/pdf/2303.09014v1) by Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, Marco Tulio Ribeiro
	## Summary
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

----

173. [**Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**](http://arxiv.org/pdf/2310.06117v1) by Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, Denny Zhou
	## Summary
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

----

174. [**Prompt, Condition, and Generate: Classification of Unsupported Claims with In-Context Learning**](http://arxiv.org/pdf/2309.10359v1) by Peter Ebert Christensen, Srishti Yadav, Serge Belongie
	## Summary
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

----

175. [**Zephyr: Direct Distillation of LM Alignment**](http://arxiv.org/pdf/2310.16944v1) by Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, Thomas Wolf
	## Summary
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

----

176. [**Data Management For Large Language Models: A Survey**](http://arxiv.org/pdf/2312.01700v2) by Zige Wang, Wanjun Zhong, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Lifeng Shang, Xin Jiang, Qun Liu
	## Summary
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

----

177. [**AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**](http://arxiv.org/pdf/2303.10512v2) by Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
	## Summary
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

----

178. [**TopicGPT: A Prompt-based Topic Modeling Framework**](http://arxiv.org/pdf/2311.01449v1) by Chau Minh Pham, Alexander Hoyle, Simeng Sun, Mohit Iyyer
	## Summary
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

----

179. [**Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models**](http://arxiv.org/pdf/2312.17661v1) by Yuqing Wang, Yun Zhao
	## Summary
	The burgeoning interest in Multimodal Large Language Models (MLLMs), such as
OpenAI's GPT-4V(ision), has significantly impacted both academic and industrial
realms. These models enhance Large Language Models (LLMs) with advanced visual
understanding capabilities, facilitating their application in a variety of
multimodal tasks. Recently, Google introduced Gemini, a cutting-edge MLLM
designed specifically for multimodal integration. Despite its advancements,
preliminary benchmarks indicate that Gemini lags behind GPT models in
commonsense reasoning tasks. However, this assessment, based on a limited
dataset (i.e., HellaSWAG), does not fully capture Gemini's authentic
commonsense reasoning potential. To address this gap, our study undertakes a
thorough evaluation of Gemini's performance in complex reasoning tasks that
necessitate the integration of commonsense knowledge across modalities. We
carry out a comprehensive analysis of 12 commonsense reasoning datasets,
ranging from general to domain-specific tasks. This includes 11 datasets
focused solely on language, as well as one that incorporates multimodal
elements. Our experiments across four LLMs and two MLLMs demonstrate Gemini's
competitive commonsense reasoning capabilities. Additionally, we identify
common challenges faced by current LLMs and MLLMs in addressing commonsense
problems, underscoring the need for further advancements in enhancing the
commonsense reasoning abilities of these models.

----

180. [**ReAct: Synergizing Reasoning and Acting in Language Models**](http://arxiv.org/pdf/2210.03629v3) by Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
	## Summary
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

----

181. [**Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents**](http://arxiv.org/pdf/2311.11797v1) by Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, Hai Zhao
	## Summary
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

----

182. [**A collection of principles for guiding and evaluating large language models**](http://arxiv.org/pdf/2312.10059v1) by Konstantin Hebenstreit, Robert Praas, Matthias Samwald
	## Summary
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

----

183. [**OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**](http://arxiv.org/pdf/2309.11235v1) by Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, Yang Liu
	## Summary
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

----

184. [**Textbooks Are All You Need II: phi-1.5 technical report**](http://arxiv.org/pdf/2309.05463v1) by Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee
	## Summary
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

----

185. [**Designing Effective Interview Chatbots: Automatic Chatbot Profiling and Design Suggestion Generation for Chatbot Debugging**](http://arxiv.org/pdf/2104.04842v1) by Xu Han, Michelle Zhou, Matthew Turner, Tom Yeh
	## Summary
	Recent studies show the effectiveness of interview chatbots for information
elicitation. However, designing an effective interview chatbot is non-trivial.
Few tools exist to help designers design, evaluate, and improve an interview
chatbot iteratively. Based on a formative study and literature reviews, we
propose a computational framework for quantifying the performance of interview
chatbots. Incorporating the framework, we have developed iChatProfile, an
assistive chatbot design tool that can automatically generate a profile of an
interview chatbot with quantified performance metrics and offer design
suggestions for improving the chatbot based on such metrics. To validate the
effectiveness of iChatProfile, we designed and conducted a between-subject
study that compared the performance of 10 interview chatbots designed with or
without using iChatProfile. Based on the live chats between the 10 chatbots and
1349 users, our results show that iChatProfile helped the designers build
significantly more effective interview chatbots, improving both interview
quality and user experience.

----

186. [**Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations**](http://arxiv.org/pdf/2311.05584v1) by Joey Hong, Sergey Levine, Anca Dragan
	## Summary
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

----

187. [**Large Language Models Understand and Can be Enhanced by Emotional Stimuli**](http://arxiv.org/pdf/2307.11760v7) by Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, Xing Xie
	## Summary
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

----

188. [**Testing Language Model Agents Safely in the Wild**](http://arxiv.org/pdf/2311.10538v3) by Silen Naihin, David Atkinson, Marc Green, Merwane Hamadi, Craig Swift, Douglas Schonholtz, Adam Tauman Kalai, David Bau
	## Summary
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

----

189. [**A Precis of Language Models are not Models of Language**](http://arxiv.org/pdf/2205.07634v1) by Csaba Veres
	## Summary
	Natural Language Processing is one of the leading application areas in the
current resurgence of Artificial Intelligence, spearheaded by Artificial Neural
Networks. We show that despite their many successes at performing linguistic
tasks, Large Neural Language Models are ill-suited as comprehensive models of
natural language. The wider implication is that, in spite of the often
overbearing optimism about AI, modern neural models do not represent a
revolution in our understanding of cognition.

----

190. [**"I Want It That Way": Enabling Interactive Decision Support Using Large Language Models and Constraint Programming**](http://arxiv.org/pdf/2312.06908v1) by Connor Lawless, Jakob Schoeffer, Lindy Le, Kael Rowan, Shilad Sen, Cristina St. Hill, Jina Suh, Bahar Sarrafzadeh
	## Summary
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

----

191. [**Algorithm Evolution Using Large Language Model**](http://arxiv.org/pdf/2311.15249v1) by Fei Liu, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang
	## Summary
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

----

192. [**TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning**](http://arxiv.org/pdf/2312.09039v1) by Yuan Sui, Jiaru Zou, Mengyu Zhou, Xinyi He, Lun Du, Shi Han, Dongmei Zhang
	## Summary
	Table reasoning has shown remarkable progress in a wide range of table-based
tasks. These challenging tasks require reasoning over both free-form natural
language (NL) questions and semi-structured tabular data. However, previous
table reasoning solutions suffer from significant performance degradation on
"huge" tables. In addition, most existing methods struggle to reason over
complex questions since they lack essential information or they are scattered
in different places. To alleviate these challenges, we exploit a table
provider, namely TAP4LLM, on versatile sampling, augmentation, and packing
methods to achieve effective semi-structured data reasoning using large
language models (LLMs), which 1) decompose raw tables into sub-tables with
specific rows or columns based on the rules or semantic similarity; 2) augment
table information by extracting semantic and statistical metadata from raw
tables while retrieving relevant knowledge from trustworthy knowledge sources
(e.g., Wolfram Alpha, Wikipedia); 3) pack sampled tables with augmented
knowledge into sequence prompts for LLMs reasoning while balancing the token
allocation trade-off. We show that TAP4LLM allows for different components as
plug-ins, enhancing LLMs' understanding of structured data in diverse tabular
tasks.

----

193. [**NoteChat: A Dataset of Synthetic Doctor-Patient Conversations Conditioned on Clinical Notes**](http://arxiv.org/pdf/2310.15959v2) by Junda Wang, Zonghai Yao, Zhichao Yang, Huixue Zhou, Rumeng Li, Xun Wang, Yucheng Xu, Hong Yu
	## Summary
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

----

194. [**Orca 2: Teaching Small Language Models How to Reason**](http://arxiv.org/pdf/2311.11045v2) by Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah
	## Summary
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

----

195. [**LLM in a flash: Efficient Large Language Model Inference with Limited Memory**](http://arxiv.org/pdf/2312.11514v2) by Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar
	## Summary
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

----

196. [**EHRTutor: Enhancing Patient Understanding of Discharge Instructions**](http://arxiv.org/pdf/2310.19212v1) by Zihao Zhang, Zonghai Yao, Huixue Zhou, Feiyun ouyang, Hong Yu
	## Summary
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

----

197. [**Mini-GPTs: Efficient Large Language Models through Contextual Pruning**](http://arxiv.org/pdf/2312.12682v1) by Tim Valicenti, Justice Vidal, Ritik Patnaik
	## Summary
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

----

198. [**Questioning the Survey Responses of Large Language Models**](http://arxiv.org/pdf/2306.07951v2) by Ricardo Dominguez-Olmedo, Moritz Hardt, Celestine Mendler-Dünner
	## Summary
	As large language models increase in capability, researchers have started to
conduct surveys of all kinds on these models with varying scientific
motivations. In this work, we examine what we can learn from language models'
survey responses on the basis of the well-established American Community Survey
(ACS) by the U.S. Census Bureau. Using a de-facto standard multiple-choice
prompting technique and evaluating 40 different language models, hundreds of
thousands of times each on questions from the ACS, we systematically establish
two dominant patterns. First, models have significant position and labeling
biases, for example, towards survey responses labeled with the letter "A".
Second, when adjusting for labeling biases through randomized answer ordering,
models across the board trend towards uniformly random survey responses. In
fact, binary classifiers can almost perfectly differentiate between models'
responses to the ACS and the responses of the US census. Taken together, our
findings suggest caution in treating survey responses from language models as
equivalent to those of human populations at present time.

----

199. [**Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs**](http://arxiv.org/pdf/2311.02262v1) by Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao
	## Summary
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

----

200. [**Learning From Mistakes Makes LLM Better Reasoner**](http://arxiv.org/pdf/2310.20689v2) by Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen
	## Summary
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

----

201. [**Ecosystem Graphs: The Social Footprint of Foundation Models**](http://arxiv.org/pdf/2303.15772v1) by Rishi Bommasani, Dilara Soylu, Thomas I. Liao, Kathleen A. Creel, Percy Liang
	## Summary
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

----

202. [**Explainable Automated Fact-Checking: A Survey**](http://arxiv.org/pdf/2011.03870v1) by Neema Kotonya, Francesca Toni
	## Summary
	A number of exciting advances have been made in automated fact-checking
thanks to increasingly larger datasets and more powerful systems, leading to
improvements in the complexity of claims which can be accurately fact-checked.
However, despite these advances, there are still desirable functionalities
missing from the fact-checking pipeline. In this survey, we focus on the
explanation functionality -- that is fact-checking systems providing reasons
for their predictions. We summarize existing methods for explaining the
predictions of fact-checking systems and we explore trends in this topic.
Further, we consider what makes for good explanations in this specific domain
through a comparative analysis of existing fact-checking explanations against
some desirable properties. Finally, we propose further research directions for
generating fact-checking explanations, and describe how these may lead to
improvements in the research area.

----

203. [**Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study**](http://arxiv.org/pdf/2305.13860v1) by Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, Yang Liu
	## Summary
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

----

204. [**A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models**](http://arxiv.org/pdf/2307.12980v1) by Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, Philip Torr
	## Summary
	Prompt engineering is a technique that involves augmenting a large
pre-trained model with task-specific hints, known as prompts, to adapt the
model to new tasks. Prompts can be created manually as natural language
instructions or generated automatically as either natural language instructions
or vector representations. Prompt engineering enables the ability to perform
predictions based solely on prompts without updating model parameters, and the
easier application of large pre-trained models in real-world tasks. In past
years, Prompt engineering has been well-studied in natural language processing.
Recently, it has also been intensively studied in vision-language modeling.
However, there is currently a lack of a systematic overview of prompt
engineering on pre-trained vision-language models. This paper aims to provide a
comprehensive survey of cutting-edge research in prompt engineering on three
types of vision-language models: multimodal-to-text generation models (e.g.
Flamingo), image-text matching models (e.g. CLIP), and text-to-image generation
models (e.g. Stable Diffusion). For each type of model, a brief model summary,
prompting methods, prompting-based applications, and the corresponding
responsibility and integrity issues are summarized and discussed. Furthermore,
the commonalities and differences between prompting on vision-language models,
language models, and vision models are also discussed. The challenges, future
directions, and research opportunities are summarized to foster future research
on this topic.

----

205. [**Technical Report: Large Language Models can Strategically Deceive their Users when Put Under Pressure**](http://arxiv.org/pdf/2311.07590v2) by Jérémy Scheurer, Mikita Balesni, Marius Hobbhahn
	## Summary
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

----

206. [**The ART of LLM Refinement: Ask, Refine, and Trust**](http://arxiv.org/pdf/2311.07961v1) by Kumar Shridhar, Koustuv Sinha, Andrew Cohen, Tianlu Wang, Ping Yu, Ram Pasunuru, Mrinmaya Sachan, Jason Weston, Asli Celikyilmaz
	## Summary
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

----

207. [**Tab-CoT: Zero-shot Tabular Chain of Thought**](http://arxiv.org/pdf/2305.17812v1) by Ziqi Jin, Wei Lu
	## Summary
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

----

208. [**MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks**](http://arxiv.org/pdf/2311.07463v1) by Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, Sunayana Sitaram
	## Summary
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

----

209. [**Foundation Models for Decision Making: Problems, Methods, and Opportunities**](http://arxiv.org/pdf/2303.04129v1) by Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, Dale Schuurmans
	## Summary
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

----

210. [**Language Generation from Brain Recordings**](http://arxiv.org/pdf/2311.09889v3) by Ziyi Ye, Qingyao Ai, Yiqun Liu, Min Zhang, Christina Lioma, Tuukka Ruotsalo
	## Summary
	Generating human language through non-invasive brain-computer interfaces
(BCIs) has the potential to unlock many applications, such as serving disabled
patients and improving communication. Currently, however, generating language
via BCIs has been previously successful only within a classification setup for
selecting pre-generated sentence continuation candidates with the most likely
cortical semantic representation. Inspired by recent research that revealed
associations between the brain and the large computational language models, we
propose a generative language BCI that utilizes the capacity of a large
language model (LLM) jointly with a semantic brain decoder to directly generate
language from functional magnetic resonance imaging (fMRI) input. The proposed
model can generate coherent language sequences aligned with the semantic
content of visual or auditory language stimuli perceived, without prior
knowledge of any pre-generated candidates. We compare the language generated
from the presented model with a random control, pre-generated language
selection approach, and a standard LLM, which generates common coherent text
solely based on the next word likelihood according to statistical language
training data. The proposed model is found to generate language that is more
aligned with semantic stimulus in response to which brain input is sampled. Our
findings demonstrate the potential and feasibility of employing BCIs in direct
language generation.

----

211. [**Cumulative Reasoning with Large Language Models**](http://arxiv.org/pdf/2308.04371v5) by Yifan Zhang, Jingqin Yang, Yang Yuan, Andrew Chi-Chih Yao
	## Summary
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

----

212. [**LLM360: Towards Fully Transparent Open-Source LLMs**](http://arxiv.org/pdf/2312.06550v1) by Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, Eric P. Xing
	## Summary
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

----

213. [**One Small Step for Generative AI, One Giant Leap for AGI: A Complete Survey on ChatGPT in AIGC Era**](http://arxiv.org/pdf/2304.06488v1) by Chaoning Zhang, Chenshuang Zhang, Chenghao Li, Yu Qiao, Sheng Zheng, Sumit Kumar Dam, Mengchun Zhang, Jung Uk Kim, Seong Tae Kim, Jinwoo Choi, Gyeong-Moon Park, Sung-Ho Bae, Lik-Hang Lee, Pan Hui, In So Kweon, Choong Seon Hong
	## Summary
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

----

214. [**Fast Chain-of-Thought: A Glance of Future from Parallel Decoding Leads to Answers Faster**](http://arxiv.org/pdf/2311.08263v1) by Hongxuan Zhang, Zhining Liu, Jiaqi Zheng, Chenyi Zhuang, Jinjie Gu, Guihai Chen
	## Summary
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

----

215. [**Generative Multimodal Models are In-Context Learners**](http://arxiv.org/pdf/2312.13286v1) by Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, Xinlong Wang
	## Summary
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

----

216. [**PDFTriage: Question Answering over Long, Structured Documents**](http://arxiv.org/pdf/2309.08872v2) by Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A. Rossi, Franck Dernoncourt
	## Summary
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

----

217. [**Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**](http://arxiv.org/pdf/2302.01560v2) by Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang
	## Summary
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

----

218. [**From Classification to Clinical Insights: Towards Analyzing and Reasoning About Mobile and Behavioral Health Data With Large Language Models**](http://arxiv.org/pdf/2311.13063v2) by Zachary Englhardt, Chengqian Ma, Margaret E. Morris, Xuhai "Orson" Xu, Chun-Cheng Chang, Lianhui Qin, Daniel McDuff, Xin Liu, Shwetak Patel, Vikram Iyer
	## Summary
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

----

219. [**Can LLMs Follow Simple Rules?**](http://arxiv.org/pdf/2311.04235v1) by Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, David Wagner
	## Summary
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

----

220. [**Making Large Language Models Better Reasoners with Step-Aware Verifier**](http://arxiv.org/pdf/2206.02336v3) by Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, Weizhu Chen
	## Summary
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

----

## 2024
1. [**Thousands of AI Authors on the Future of AI**](http://arxiv.org/pdf/2401.02843v1) by Katja Grace, Harlan Stewart, Julia Fabienne Sandkühler, Stephen Thomas, Ben Weinstein-Raun, Jan Brauner
	## Summary
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

----

2. [**Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models**](http://arxiv.org/pdf/2401.00625v2) by Guangji Bai, Zheng Chai, Chen Ling, Shiyu Wang, Jiaying Lu, Nan Zhang, Tingwei Shi, Ziyang Yu, Mengdan Zhu, Yifei Zhang, Carl Yang, Yue Cheng, Liang Zhao
	## Summary
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

----

3. [**ChatGPT's One-year Anniversary: Are Open-Source Large Language Models Catching up?**](http://arxiv.org/pdf/2311.16989v4) by Hailin Chen, Fangkai Jiao, Xingxuan Li, Chengwei Qin, Mathieu Ravaut, Ruochen Zhao, Caiming Xiong, Shafiq Joty
	## Summary
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

----

4. [**The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models**](http://arxiv.org/pdf/2401.05618v1) by Matthew Renze, Erhan Guven
	## Summary
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

----

5. [**Seven Failure Points When Engineering a Retrieval Augmented Generation System**](http://arxiv.org/pdf/2401.05856v1) by Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek
	## Summary
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

----

6. [**From LLM to Conversational Agent: A Memory Enhanced Architecture with Fine-Tuning of Large Language Models**](http://arxiv.org/pdf/2401.02777v1) by Na Liu, Liangyu Chen, Xiaoyu Tian, Wei Zou, Kaijiang Chen, Ming Cui
	## Summary
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

----

7. [**Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**](http://arxiv.org/pdf/2401.05566v2) by Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, Ethan Perez
	## Summary
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

----

8. [**Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**](http://arxiv.org/pdf/2401.06561v1) by Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao
	## Summary
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

----

9. [**A Computational Framework for Behavioral Assessment of LLM Therapists**](http://arxiv.org/pdf/2401.00820v1) by Yu Ying Chiu, Ashish Sharma, Inna Wanyin Lin, Tim Althoff
	## Summary
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

----

10. [**P^3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning**](http://arxiv.org/pdf/2205.01886v2) by Xiaomeng Hu, Shi Yu, Chenyan Xiong, Zhenghao Liu, Zhiyuan Liu, Ge Yu
	## Summary
	Compared to other language tasks, applying pre-trained language models (PLMs)
for search ranking often requires more nuances and training signals. In this
paper, we identify and study the two mismatches between pre-training and
ranking fine-tuning: the training schema gap regarding the differences in
training objectives and model architectures, and the task knowledge gap
considering the discrepancy between the knowledge needed in ranking and that
learned during pre-training. To mitigate these gaps, we propose Pre-trained,
Prompt-learned and Pre-finetuned Neural Ranker (P^3 Ranker). P^3 Ranker
leverages prompt-based learning to convert the ranking task into a pre-training
like schema and uses pre-finetuning to initialize the model on intermediate
supervised tasks. Experiments on MS MARCO and Robust04 show the superior
performances of P^3 Ranker in few-shot ranking. Analyses reveal that P^3 Ranker
is able to better accustom to the ranking task through prompt-based learning
and retrieve necessary ranking-oriented knowledge gleaned in pre-finetuning,
resulting in data-efficient PLM adaptation. Our code is available at
https://github.com/NEUIR/P3Ranker.

----

11. [**DocLLM: A layout-aware generative language model for multimodal document understanding**](http://arxiv.org/pdf/2401.00908v1) by Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, Xiaomo Liu
	## Summary
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

----

12. [**Understanding LLMs: A Comprehensive Overview from Training to Inference**](http://arxiv.org/pdf/2401.02038v2) by Yiheng Liu, Hao He, Tianle Han, Xu Zhang, Mengyuan Liu, Jiaming Tian, Yutong Zhang, Jiaqi Wang, Xiaohui Gao, Tianyang Zhong, Yi Pan, Shaochen Xu, Zihao Wu, Zhengliang Liu, Xin Zhang, Shu Zhang, Xintao Hu, Tuo Zhang, Ning Qiang, Tianming Liu, Bao Ge
	## Summary
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

----

13. [**Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**](http://arxiv.org/pdf/2401.04398v1) by Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister
	## Summary
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

----

14. [**Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language Model Systems**](http://arxiv.org/pdf/2401.05778v1) by Tianyu Cui, Yanling Wang, Chuanpu Fu, Yong Xiao, Sijia Li, Xinhao Deng, Yunpeng Liu, Qinglin Zhang, Ziyi Qiu, Peiyang Li, Zhixing Tan, Junwu Xiong, Xinyu Kong, Zujie Wen, Ke Xu, Qi Li
	## Summary
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

----

15. [**TouchUp-G: Improving Feature Representation through Graph-Centric Finetuning**](http://arxiv.org/pdf/2309.13885v1) by Jing Zhu, Xiang Song, Vassilis N. Ioannidis, Danai Koutra, Christos Faloutsos
	## Summary
	How can we enhance the node features acquired from Pretrained Models (PMs) to
better suit downstream graph learning tasks? Graph Neural Networks (GNNs) have
become the state-of-the-art approach for many high-impact, real-world graph
applications. For feature-rich graphs, a prevalent practice involves utilizing
a PM directly to generate features, without incorporating any domain adaptation
techniques. Nevertheless, this practice is suboptimal because the node features
extracted from PM are graph-agnostic and prevent GNNs from fully utilizing the
potential correlations between the graph structure and node features, leading
to a decline in GNNs performance. In this work, we seek to improve the node
features obtained from a PM for downstream graph tasks and introduce TOUCHUP-G,
which has several advantages. It is (a) General: applicable to any downstream
graph task, including link prediction which is often employed in recommender
systems; (b) Multi-modal: able to improve raw features of any modality (e.g.
images, texts, audio); (c) Principled: it is closely related to a novel metric,
feature homophily, which we propose to quantify the potential correlations
between the graph structure and node features and we show that TOUCHUP-G can
effectively shrink the discrepancy between the graph structure and node
features; (d) Effective: achieving state-of-the-art results on four real-world
datasets spanning different tasks and modalities.

----

16. [**How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**](http://arxiv.org/pdf/2401.06373v1) by Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi
	## Summary
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

----

17. [**General-purpose foundation models for increased autonomy in robot-assisted surgery**](http://arxiv.org/pdf/2401.00678v1) by Samuel Schmidgall, Ji Woong Kim, Alan Kuntz, Ahmed Ezzat Ghazi, Axel Krieger
	## Summary
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

----

18. [**TinyLlama: An Open-Source Small Language Model**](http://arxiv.org/pdf/2401.02385v1) by Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu
	## Summary
	We present TinyLlama, a compact 1.1B language model pretrained on around 1
trillion tokens for approximately 3 epochs. Building on the architecture and
tokenizer of Llama 2, TinyLlama leverages various advances contributed by the
open-source community (e.g., FlashAttention), achieving better computational
efficiency. Despite its relatively small size, TinyLlama demonstrates
remarkable performance in a series of downstream tasks. It significantly
outperforms existing open-source language models with comparable sizes. Our
model checkpoints and code are publicly available on GitHub at
https://github.com/jzhang38/TinyLlama.

----

19. [**LLM Augmented LLMs: Expanding Capabilities through Composition**](http://arxiv.org/pdf/2401.02412v1) by Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, Partha Talukdar
	## Summary
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

----

20. [**Pushing Boundaries: Exploring Zero Shot Object Classification with Large Multimodal Models**](http://arxiv.org/pdf/2401.00127v1) by Ashhadul Islam, Md. Rafiul Biswas, Wajdi Zaghouani, Samir Brahim Belhaouari, Zubair Shah
	## Summary
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

----

21. [**Enhancing the medical foundation model with multi-scale and cross-modality feature learning**](http://arxiv.org/pdf/2401.01583v1) by Weijian Huang, Cheng Li, Hong-Yu Zhou, Jiarun Liu, Hao Yang, Yong Liang, Shanshan Wang
	## Summary
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

----

22. [**LLaMA Pro: Progressive LLaMA with Block Expansion**](http://arxiv.org/pdf/2401.02415v1) by Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, Ying Shan
	## Summary
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

----

23. [**AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction**](http://arxiv.org/pdf/2305.09620v2) by Junsol Kim, Byungkyu Lee
	## Summary
	Large language models (LLMs) that produce human-like responses have begun to
revolutionize research practices in the social sciences. This paper shows how
we can integrate LLMs and social surveys to accurately predict individual
responses to survey questions that were not asked before. We develop a novel
methodological framework to personalize LLMs by considering the meaning of
survey questions derived from their text, the latent beliefs of individuals
inferred from their response patterns, and the temporal contexts across
different survey periods through fine-tuning LLMs with survey data. Using the
General Social Survey from 1972 to 2021, we show that the fine-tuned model
based on Alpaca-7b can predict individual responses to survey questions that
are partially missing as well as entirely missing. The remarkable prediction
capabilities allow us to fill in missing trends with high confidence and
pinpoint when public attitudes changed, such as the rising support for same-sex
marriage. We discuss practical constraints, socio-demographic representation,
and ethical concerns regarding individual autonomy and privacy when using LLMs
for opinion prediction. This study demonstrates that LLMs and surveys can
mutually enhance each other's capabilities: LLMs broaden survey potential,
while surveys improve the alignment of LLMs.

----

24. [**LLaMA Beyond English: An Empirical Study on Language Capability Transfer**](http://arxiv.org/pdf/2401.01055v2) by Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, Xuanjing Huang
	## Summary
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

----

25. [**Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon**](http://arxiv.org/pdf/2401.03462v1) by Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, Zhicheng Dou
	## Summary
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

----

26. [**Assessing Knowledge Editing in Language Models via Relation Perspective**](http://arxiv.org/pdf/2311.09053v1) by Yifan Wei, Xiaoyan Yu, Huanhuan Ma, Fangyu Lei, Yixuan Weng, Ran Song, Kang Liu
	## Summary
	Knowledge Editing (KE) for modifying factual knowledge in Large Language
Models (LLMs) has been receiving increasing attention. However, existing
knowledge editing methods are entity-centric, and it is unclear whether this
approach is suitable for a relation-centric perspective. To address this gap,
this paper constructs a new benchmark named RaKE, which focuses on Relation
based Knowledge Editing. In this paper, we establish a suite of innovative
metrics for evaluation and conduct comprehensive experiments involving various
knowledge editing baselines. We notice that existing knowledge editing methods
exhibit the potential difficulty in their ability to edit relations. Therefore,
we further explore the role of relations in factual triplets within the
transformer. Our research results confirm that knowledge related to relations
is not only stored in the FFN network but also in the attention layers. This
provides experimental support for future relation-based knowledge editing
methods.

----

27. [**Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**](http://arxiv.org/pdf/2401.00905v1) by Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang
	## Summary
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

----

28. [**A Survey of Hallucination in Large Foundation Models**](http://arxiv.org/pdf/2309.05922v1) by Vipula Rawte, Amit Sheth, Amitava Das
	## Summary
	Hallucination in a foundation model (FM) refers to the generation of content
that strays from factual reality or includes fabricated information. This
survey paper provides an extensive overview of recent efforts that aim to
identify, elucidate, and tackle the problem of hallucination, with a particular
focus on ``Large'' Foundation Models (LFMs). The paper classifies various types
of hallucination phenomena that are specific to LFMs and establishes evaluation
criteria for assessing the extent of hallucination. It also examines existing
strategies for mitigating hallucination in LFMs and discusses potential
directions for future research in this area. Essentially, the paper offers a
comprehensive examination of the challenges and solutions related to
hallucination in LFMs.

----

29. [**LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning**](http://arxiv.org/pdf/2401.01325v1) by Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, Xia Hu
	## Summary
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

----

30. [**DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**](http://arxiv.org/pdf/2401.02954v1) by DeepSeek-AI, :, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, Yuheng Zou
	## Summary
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

----

31. [**If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**](http://arxiv.org/pdf/2401.00812v2) by Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R. Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, Heng Ji, Chengxiang Zhai
	## Summary
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

----

32. [**Large Language Models Relearn Removed Concepts**](http://arxiv.org/pdf/2401.01814v1) by Michelle Lo, Shay B. Cohen, Fazl Barez
	## Summary
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

----

33. [**Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review**](http://arxiv.org/pdf/2401.01519v2) by Luoma Ke, Song Tong, Peng Cheng, Kaiping Peng
	## Summary
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

----

34. [**Teach me with a Whisper: Enhancing Large Language Models for Analyzing Spoken Transcripts using Speech Embeddings**](http://arxiv.org/pdf/2311.07014v1) by Fatema Hasan, Yulong Li, James Foulds, Shimei Pan, Bishwaranjan Bhattacharjee
	## Summary
	Speech data has rich acoustic and paralinguistic information with important
cues for understanding a speaker's tone, emotion, and intent, yet traditional
large language models such as BERT do not incorporate this information. There
has been an increased interest in multi-modal language models leveraging audio
and/or visual information and text. However, current multi-modal language
models require both text and audio/visual data streams during inference/test
time. In this work, we propose a methodology for training language models
leveraging spoken language audio data but without requiring the audio stream
during prediction time. This leads to an improved language model for analyzing
spoken transcripts while avoiding an audio processing overhead at test time. We
achieve this via an audio-language knowledge distillation framework, where we
transfer acoustic and paralinguistic information from a pre-trained speech
embedding (OpenAI Whisper) teacher model to help train a student language model
on an audio-text dataset. In our experiments, the student model achieves
consistent improvement over traditional language models on tasks analyzing
spoken transcripts.

----

35. [**TrustLLM: Trustworthiness in Large Language Models**](http://arxiv.org/pdf/2401.05561v2) by Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, Willian Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, Yue Zhao
	## Summary
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

----

36. [**Mixtral of Experts**](http://arxiv.org/pdf/2401.04088v1) by Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
	## Summary
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

----

37. [**Correctness Comparison of ChatGPT-4, Bard, Claude-2, and Copilot for Spatial Tasks**](http://arxiv.org/pdf/2401.02404v2) by Hartwig H. Hochmair, Levente Juhasz, Takoda Kemp
	## Summary
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

----

38. [**The Earth is Flat? Unveiling Factual Errors in Large Language Models**](http://arxiv.org/pdf/2401.00761v1) by Wenxuan Wang, Juluan Shi, Zhaopeng Tu, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, Michael R. Lyu
	## Summary
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

----

39. [**LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model**](http://arxiv.org/pdf/2401.02330v2) by Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, Jian Tang
	## Summary
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

----

40. [**The Impact of Reasoning Step Length on Large Language Models**](http://arxiv.org/pdf/2401.04925v2) by Mingyu Jin, Qinkai Yu, Dong shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, Mengnan Du
	## Summary
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

----
