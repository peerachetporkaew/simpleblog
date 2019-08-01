---
title: "Research Found 2019-7-20"
comments : true
layout: post
published: true
categories: Research
---

### [UNIVERSAL TRANSFORMERS](https://arxiv.org/pdf/1807.03819.pdf)

### [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)

### [Modeling Recurrence for Transformer](https://arxiv.org/pdf/1904.03092.pdf)

Recently, the Transformer model that is based solely on attention mechanisms, has advanced the state-of-the-art on various machine translation tasks. However, recent studies reveal that the lack of recurrence hinders its further improvement of translation capacity. In response to this problem, we propose to directly model recurrence for Transformer with an additional recurrence encoder. In addition to the standard recurrent neural network, we introduce a novel attentive recurrent network to leverage the strengths of both attention and recurrent networks. Experimental results on the widely-used WMT14 English-German and WMT17 Chinese-English translation tasks demonstrate the effectiveness of the proposed approach. Our studies also reveal that the proposed model benefits from a short-cut that bridges the source and target sequences with a single recurrent layer, which outperforms its deep counterpart.

### [Modeling Localness for Self-Attention Networks](https://arxiv.org/pdf/1810.10182.pdf)

Self-attention networks have proven to be of profound value for its strength of capturing global dependencies. In this work, we propose to model localness for self-attention networks, which enhances the ability of capturing useful local context. We cast localness modeling as a learnable Gaussian bias, which indicates the central and scope of the local region to be paid more attention. The bias is then incorporated into the original attention distribution to form a revised distribution. To maintain the strength of capturing long distance dependencies and enhance the ability of capturing short-range dependencies, we only apply localness modeling to lower layers of self-attention networks. Quantitative and qualitative analyses on Chinese-English and English-German translation tasks demonstrate the effectiveness and universality of the proposed approach.





### [Brancher -A user-centered Python package for differentiable probabilistic inference](https://brancher.org/)

```python
a = Normal(0, 1, name="a")
b = Normal(0, 1, name="b")
c = Normal(a**2 + sin(b), exp(b), name="c")
```


### [Phrase-level Self-Attention Networks for Universal Sentence Encoding](https://aclweb.org/anthology/papers/D/D18/D18-1408/)

Universal sentence encoding is a hot topic in recent NLP research. Attention mechanism has been an integral part in many sentence encoding models, allowing the models to capture context dependencies regardless of the distance between the elements in the sequence. Fully attention-based models have recently attracted enormous interest due to their highly parallelizable computation and significantly less training time. However, the memory consumption of their models grows quadratically with the sentence length, and the syntactic information is neglected. To this end, we propose Phrase-level Self-Attention Networks (PSAN) that perform self-attention across words inside a phrase to capture context dependencies at the phrase level, and use the gated memory updating mechanism to refine each word’s representation hierarchically with longer-term context dependencies captured in a larger phrase. As a result, the memory consumption can be reduced because the self-attention is performed at the phrase level instead of the sentence level. At the same time, syntactic information can be easily integrated in the model. Experiment results show that PSAN can achieve the state-of-the-art performance across a plethora of NLP tasks including binary and multi-class classification, natural language inference and sentence similarity.



### [Predict then Propagate:](https://github.com/benedekrozemberczki/APPNP)

A PyTorch implementation of "Predict then Propagate: Graph Neural Networks meet Personalized PageRank" (ICLR 2019). APPNP is a node-level semi-supervised learning algorithm which has near state-of-the-art performance on most standard node classification datasets. It can be used for tasks such as document labelling, malware detection, churn prediction and so on.
https://github.com/benedekrozemberczki/APPNP



> Idea : 
Can this implement in Machine Translation (Ontology + NMT)

Great !

### [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)

https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255


### [Span-Based Constituency Parsing with a Structure-Label System and Provably Optimal Dynamic Oracles](http://web.engr.oregonstate.edu/~huanlian/papers/span.pdf)

http://web.engr.oregonstate.edu/~huanlian/papers/span.pdf
http://web.engr.oregonstate.edu/~huanlian/



### [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)

https://arxiv.org/pdf/1803.02155.pdf

https://medium.com/@_init_how-self-attention-with-relative-position-representations-works-28173b8c245a

### [(Hardcore) Weight Agnostic Neural Networks](https://arxiv.org/pdf/1906.04358.pdf)

https://weightagnostic.github.io

https://arxiv.org/pdf/1906.04358.pdf

### [Paper with CODE !!](https://paperswithcode.com/sota)

https://paperswithcode.com/sota


### [Input Switched Affine Networks: An RNN Architecture Designed for Interpretability (ICML 2017)](https://github.com/philipperemy/tensorflow-isan-rnn)

https://github.com/philipperemy/tensorflow-isan-rnn

ISAN

$$
    h_t = W_{x_t}h_{t-1} + b_{x_t}
$$

Simple RNN

$$
    h_t = W[h_{t-1};x] + b
$$


### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

### [Domain Adaptation of Neural Machine Translation by Lexicon Induction](https://arxiv.org/pdf/1906.00376.pdf)

It has been previously noted that neural machine translation (NMT) is very sensitive to
domain shift. In this paper, we argue that
this is a dual effect of the highly lexicalized
nature of NMT, resulting in failure for sentences with large numbers of unknown words,
and lack of supervision for domain-specific
words. To remedy this problem, we propose an
unsupervised adaptation method which finetunes a pre-trained out-of-domain NMT model
using a pseudo-in-domain corpus. Specifically, we perform lexicon induction to extract an in-domain lexicon, and construct a
pseudo-parallel in-domain corpus by performing word-for-word back-translation of monolingual in-domain target sentences. In five
domains over twenty pairwise adaptation settings and two model architectures, our method
achieves consistent improvements without using any in-domain parallel sentences, improving up to 14 BLEU over unadapted models,
and up to 2 BLEU over strong back-translation
baselines.

### [Neural Machine Translation for Query Construction and Composition](https://uclmr.github.io/nampi/extended_abstracts/soru.pdf)

Research on question answering with knowledge
base has recently seen an increasing use of deep
architectures. In this extended abstract, we study
the application of the neural machine translation
paradigm for question parsing. We employ a
sequence-to-sequence model to learn graph patterns in the SPARQL graph query language and
their compositions. Instead of inducing the programs through question-answer pairs, we expect
a semi-supervised approach, where alignments
between questions and queries are built through
templates. We argue that the coverage of language
utterances can be expanded using late notable
works in natural language generation.

### [WHAT DO YOU LEARN FROM CONTEXT? PROBING FOR SENTENCE STRUCTURE IN CONTEXTUALIZED WORD REPRESENTATIONS](https://openreview.net/pdf?id=SJzSgnRcKX)

Contextualized representation models such as ELMo (Peters et al., 2018a) and
BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a
diverse array of downstream NLP tasks. Building on recent token-level probing
work, we introduce a novel edge probing task design and construct a broad suite
of sub-sentence tasks derived from the traditional structured NLP pipeline. We
probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic,
local, and long-range phenomena. We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a
non-contextual baseline.

### [Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536)

Natural language is hierarchically structured: smaller units (e.g., phrases) are nested within larger units (e.g., clauses). When a larger constituent ends, all of the smaller constituents that are nested within it must also be closed. While the standard LSTM architecture allows different neurons to track information at different time scales, it does not have an explicit bias towards modeling a hierarchy of constituents. This paper proposes to add such an inductive bias by ordering the neurons; a vector of master input and forget gates ensures that when a given neuron is updated, all the neurons that follow it in the ordering are also updated. Our novel recurrent architecture, ordered neurons LSTM (ON-LSTM), achieves good performance on four different tasks: language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.

[CODE](https://github.com/yikangshen/Ordered-Neurons)

### [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch.

[CODE](https://github.com/kimiyoung/transformer-xl)

[CODE](https://github.com/huggingface/pytorch-transformers)



### [Hierarchical Multiscale Recurrent Neural Networks](https://www.semanticscholar.org/paper/Hierarchical-Multiscale-Recurrent-Neural-Networks-Chung-Ahn/0ca2bd0e40a8f0a57665535ae1c31561370ad183)

Learning both hierarchical and temporal representation has been among the long-standing challenges of recurrent neural networks. Multiscale recurrent neural networks have been considered as a promising approach to resolve this issue, yet there has been a lack of empirical evidence showing that this type of models can actually capture the temporal dependencies by discovering the latent hierarchical structure of the sequence. In this paper, we propose a novel multiscale approach, called the hierarchical multiscale recurrent neural networks, which can capture the latent hierarchical structure in the sequence by encoding the temporal dependencies with different timescales using a novel update mechanism. We show some evidence that our proposed multiscale architecture can discover underlying hierarchical structure in the sequences without using explicit boundary information. We evaluate our proposed model on character-level language modelling and handwriting sequence modelling.
