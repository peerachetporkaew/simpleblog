---
title: "Research Found 2019-8"
comments : true
layout: post
published: true
categories: Research
---

### [Towards Explainable NLP: A Generative Explanation Framework for Text Classification](https://arxiv.org/pdf/1811.00196.pdf)

Building explainable systems is a critical problem in the field of Natural Language Processing (NLP), since most machine learning models provide no explanations for the predictions. Existing approaches for explainable machine learning systems tend to focus on interpreting the outputs or the connections between inputs and outputs. However, the fine-grained information is often ignored, and the systems do not explicitly generate the human-readable explanations. To better alleviate this problem, we propose a novel generative explanation framework that learns to make classification decisions and generate fine-grained explanations at the same time. More specifically, we introduce the explainable factor and the minimum risk training approach that learn to generate more reasonable explanations. We construct two new datasets that contain summaries, rating scores, and fine-grained reasons. We conduct experiments on both datasets, comparing with several strong neural network baseline systems. Experimental results show that our method surpasses all baselines on both datasets, and is able to generate concise explanations at the same time. (Accepted to ACL 2019)


### [Gumble-Softmax](https://arxiv.org/pdf/1611.01144.pdf)

https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html

http://legacydirs.umiacs.umd.edu/~jbg/teaching/CMSC_726/18d.pdf

https://www.zhihu.com/question/62631725

https://www.youtube.com/watch?v=JFgXEbgcT7g
 
คำอธิบายจาก [เนิร์ด ML](https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2Fpermalink.php%3Fstory_fbid%3D876772579359148%26id%3D823059881397085&width=500) 

<iframe src="https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2Fpermalink.php%3Fstory_fbid%3D876772579359148%26id%3D823059881397085&width=500" width="500" height="851" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowTransparency="true" allow="encrypted-media"></iframe>

Categorical Reparameterization with Gumbel-Softmax (2016)

ในโจทย์ text generation เรามักจะใช้ softmax เป็น output layer โดยที่ softmax จะ model ความน่าจะเป็นที่จะ output token หนึ่ง ๆ ซึ่งอาจจะเป็น word หรือว่า character ก็ได้

โดยทั่วไปเราจะใช้ sequence model เช่น LSTM ในการค่อย ๆ พ่น token เหล่านั้นออกมาและประกอบเป็น sentence ในตอนท้ายสุด

เนื่องจากงานในด้าน NLP มีการค้นพบว่าการใช้ language model เป็น training signal นั้นช่วยให้คุณภาพของ text generation นั้นยิ่งดีขึ้นไปอีก

ดังนั้นเราจะสมมติไปอีกว่าหลังจากเรานำ token ที่ได้ generate นั้นโยนเข้าไปใน language model แล้ว เราจะต้องสามารถ backpropagate ผ่าน language model และ ผ่าน token กลับมาที่ LSTM ที่ generate token เหล่านั้นเพื่อเทรนให้มันเก่งขึ้นภายใต้ language model ได้

จะเห็นว่าปัญหาก็จะบังเกิดขึ้นระหว่าง "ข้อต่อ" ของ LSTM กับ language model นี่เอง เพราะว่าในขั้นตอนสุดท้ายของ LSTM เราได้ทำการ sampling เพื่อให้ได้ token ที่เป็น output ไปเรียบร้อยแล้ว และการ backpropagate ผ่าน sampling นั้นทำไม่ได้เพราะว่าฟังก์ชันนี้ไม่มี gradient

อย่างไรก็ตามเรายังสามารถหา gradient ทางอ้อมได้อยู่ด้วยการใช้เทคนิค likelihood ratio เช่น REINFORCE (Williams, 1992) ในการคะเน gradient ของชั้นการ sampling ได้ ซึ่งหลาย ๆ งานด้าน NLP ก็แก้ปัญหาดังกล่าวด้วยวิธีนี้

อย่างไรก็ตามวิธี REINFORCE นั้นแม้จะให้ gradient ที่ถูกต้อง (unbiased) แต่ว่ามันมี variance สูง ทำให้การเทรนนั้นไม่ได้รวดเร็วเท่าที่ควร และก็อาจจะทำให้การเทรนไม่ converge ได้เช่นกัน

ทางเลือกอีกทางหนึ่งหากเราไม่ต้องการใช้ REINFORCE ก็คือการใช้ straight-through แทน ซึ่งเทคนิคนี้ถือว่าขั้นตอนการ sampling นั้นมี gradient เป็น 1 ซึ่งทำให้เราคำนวณ gradient ผ่านไปได้ (แบบผิด ๆ) วิธีการนี้จึงให้ gradient ที่ไม่ถูกต้อง (biased) แต่ว่ามี variance ต่ำ ซึ่งทำให้เทรนได้รวดเร็วกว่า

ไม่ว่าจะเลือกวิธีการใดก็มีข้อเสียทั้งสิ้น นี่จึงเป็นที่มาของเปเปอร์นี้ที่จะเชื่อมสองวิธีนี้เข้าด้วยกัน ลักษณะของ Gumbel-Softmax คือ เราจะตอนเริ่มต้นเราจะใช้วิธีที่ variance ต่ำ แต่มี bias เพื่อให้เทรนได้เร็ว แต่เมื่อการเทรนใกล้สิ้นสุดเราจะลด bias ลงและเพิ่ม variance เข้าไปแทน เพื่อให้ได้คำตอบที่ดีขึ้น

เพื่อจะให้สามารถทำสิ่งดังกล่าวได้จึงมีความจำเป็นอย่างยิ่งที่จะทำให้ทุก ๆ ค่าที่ sample มานั้นยังสามารถ backpropagate ได้อยู่ ซึ่งต่างจากการ sampling ทั่วไปที่เราจะได้ one-hot vector ซึ่งไม่เหลือเค้าลางของการคำนวณที่ผ่านมาแล้ว

นั่นก็คือแทนที่แต่ละครั้งที่ sampling จะได้ one-hot vector เราจะได้อะไรที่ "smooth" กว่านั้นเล็กน้อย เป็นลักษณะของ softmax แทน (softmax คนละตัวกับ output layer) โดยที่ความ smooth สามารถควบคุมได้ด้วยการปรับ "temperature" ของ softmax นั่นเอง

โดยหาก softmax มี temperature สูง ๆ ก็คือ "ร้อน" entropy ก็จะมีมาก ความแน่นอนของคลาสจะมีน้อย ก็คือ sample จะมีความกระจายและการ backpropagate ผ่าน sample นี้ก็จะมี variance ต่ำ แต่จะมี bias เพราะว่าหน้าตาของมันไม่ได้เหมือนกับ one-hot (แต่เดิม) ซะทีเดียว

หาก softmax นี้มี temperature ต่ำ ๆ ก็คือ "เย็น" entropy ก็จะน้อย และก็จะมีความแน่นอนของคลาสมาก ก็คือจะมีหน้าตาคล้าย one-hot มากเข้าไปเรื่อย ๆ เมื่อ temperature = 0 มันก็จะกลายเป็น one-hot และการ backpropagate ผ่าน sample นี้ก็จะมี variance สูง แต่มี bias น้อยเพราะหน้าตาเหมือน one-hot มากกว่า

ในทางปฏิบัติเราก็จะค่อย ๆ "ลด" temperature ที่ว่านี้ลงจนน้อยมาก ๆ ใกล้ ๆ 0 (แต่ไม่ใช่ 0) เพื่อให้การเทรนของเราจบด้วยการมี bias น้อย

เพื่อให้ทุกสิ่งที่กล่าวมาสามารถทำได้จริงทางคณิตศาสตร์ การสุ่มค่าจาก categorical distribution (softmax ของ output layer) นั้นจะทำผ่าน Gumbel-Softmax trick ซึ่งจริง ๆ ก็คือการสุ่มค่าจาก Gumbel(0,1) ก่อน แล้วผ่านบวกลบคูณหารเล็กน้อยก็จะได้ค่า sample

ในมุมนี้ก็จะเห็นว่าจริง ๆ แล้วไม่ว่า categorical distribution ของเราจะมีหน้าตาอย่างไรก็ตาม สิ่งที่เราต้องทำคือการสุ่มจาก Gumbel(0,1) เสมอ ซึ่งทำให้เทคนิค Gumbel-Softmax นั้นเหมือนการใช้ Reparameterization trick ที่นำเสนอในการใช้เทรน variational auto encoder อย่างมาก แต่ว่า Gumbel-Softmax นั้นใช้สำหรับ categorical distribution ในขณะที่ Rep. trick ใช้กับ continuous distribution ที่มีหน้าตาคล้าย Gaussian

```python
#from https://github.com/vlievin/pytorch/blob/a29ba4cb3d882eab52f1755b6c0ced6008d27ceb/torch/nn/functional.py

@weak_script
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    gumbels = - (torch.empty_like(logits).exponential_() + eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
```

### [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (ICML2017)](https://arxiv.org/abs/1703.03400)

We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.

English :

Model-agnostic ==  ไม่ขึ้นกับ Model / Model-independent


### [interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

Good Ebook


### [N-shot Learning](https://blog.floydhub.com/n-shot-learning/?fbclid=IwAR0BXdLyBWdGZI-aqclgmWkg-1mQyviFG7SI980ks-IeSG-dHoKD9yrURA8)


### [ML Slides](https://m2dsupsdlclass.github.io/lectures-labs/?fbclid=IwAR0H-N2iOjeAAYjf0p11xVdkpeEF6_tQJN1DmX2sQE1GuvfzBP5P0r2fD20)


### MinimumRiskTraining for NMT

https://github.com/neubig/nmt-tips

http://www.phontron.com/paper/neubig16wat.pdf

https://www.groundai.com/project/on-the-weaknesses-of-reinforcement-learning-for-neural-machine-translation/1

### [Differentiable Dynamic Programming for Structured Prediction and Attention](http://proceedings.mlr.press/v80/mensch18a/mensch18a.pdf)

Dynamic programming (DP) solves a variety of
structured combinatorial problems by iteratively
breaking them down into smaller subproblems.
In spite of their versatility, many DP algorithms
are non-differentiable, which hampers their use
as a layer in neural networks trained by backpropagation. To address this issue, we propose
to smooth the max operator in the dynamic programming recursion, using a strongly convex
regularizer. This allows to relax both the optimal value and solution of the original combinatorial problem, and turns a broad class of DP algorithms into differentiable operators.



### A fresh look at some Machine Learning techniques from the perspective of Dempster-Shafer theory
http://bmei.cmu.ac.th/file.php?id=455

