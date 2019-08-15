---
title: "Minimum Risk Training for NMT"
comments : true
layout: post
published: true
categories: Paper
---

### Introduction

MRT เป็นเทคนิคที่ใช้ REINFORCE Algorithm ในการหา Policy Gradient ซึ่ง หลักการก็คือ เค้าเปลี่ยนวิธีการหา loss ใหม่ แต่เดิมหา loss จาก Groundtruth โดยใช้ Maximum Likelihood ซึ่งมันไม่สอดคล้องกับการใช้ BLEU score

REINFORCE Algorithm (http://www.scholarpedia.org/article/Policy_gradient_methods)


แต่ว่าจะเอา BLEU Score มา Optimize ตรงๆ ทำไม่ได้ เพราะว่าไม่สามารถทำ Backpropagation ได้ ดังนั้นจึงต้องใช้ REINFORCE Algorithm หลักการคือการหา Expected Risk loss ซึ่ง Risk loss คืออะไรก็ได้ ยกตัวอย่างเช่น (1-BLEU) ยิ่ง BLEU เยอะ Risk ก็จะน้อย

ก็หา Expectation ก็จะเริ่มจาก Sampling Y' ออกมาจาก X จำนวนหนึ่ง แล้วหา Risk Loss เอา Risk Loss (sentence-level) ไปคูณกับ cost((Y',X)) แล้วค่อยรวมทั้งหมดอีกที

อธิบายอีกอย่างหนึ่งคือ เราไม่รู้ว่า Policy ปัจจุบันดีหรือไม่ดี เราเลย Sampling ออกมาแล้วดู Risk เฉลี่ยทั้งหมดที่ Sampling ได้ เอาไปคูณกับ loss แบบ ML ที่เปลี่ยน target จาก Groundtruth ให้กลายเป็น target ที่ Sampling ออกมานั่นเอง ทำให้ อันไหนที่ดี มันจะ loss น้อย อันที่ไม่ดี จะ loss สูง 

ปัญหาของ วิธีการนี้อยู่ที่ตัว Risk ที่อาจจะมี variance สูง ดังนั้นจึงต้องทำการ Normalize เสียก่อน ซึ่งอาจมีการกำหนดค่า temperature (&alpha;) ไว้ด้วย

มาดูสมการกัน

![MRT](assets/mrt_equation.png)

ตัวอย่างการใช้ MRT และ temperature ของ Nematus

loss คือ  &#9651;(y,y^(s)) มันคือ loss ที่เกิดจาก Sampling ซึ่งเป็น Sentence-level vector วัดด้วยความแตกต่างแบบ BLEU score นั่นเอง

```python
def mrt_cost(cost, y_mask, options):

    """
        ค่า cost ที่เข้ามาตอนนี้ เป็น cost ของแต่ละประโยคแล้ว เช่น
        cost = [6.1, 3.2, 20.1] ซึ่งมาจาก -log P จึงมีค่าเป็น บวกเสมอ
        กรณีที่กำหนดค่า options['mrt_ml_mix'] > 0 นั่นหมายความว่าเค้าเอา target เฉลยใส่เข้าไปด้วยที่ตัวแรก (** ไม่ชัวร์)

        ค่า alpha เป็นค่า Temperature ซึ่งกำหนดให้น้อยมากๆๆๆ ทำให้ความแตกต่างของ Prob ของ
        แต่ละประโยค (หลังจาก normalize ด้วย softmax แล้ว) น้อยลงมากๆ เช่นนี้ทำให้การ Update Gradient ไม่ได้ถูกทำกับ Prob สูงเพียงอย่างเดียว ซึ่งใน Paper ใช้ 0.005 

    """

    loss = tensor.vector('loss', dtype=floatX) 
    alpha = theano.shared(numpy_floatX(options['mrt_alpha']))

    if options['mrt_ml_mix'] > 0:
        ml_cost = cost[0]

        # remove reference for MRT objective unless enabled
        if not options['mrt_reference']:
            cost = cost[1:]

    cost *= alpha #คือ (Negative Log prob ของแต่ละ Sample) * alpha

    # ณ ตอนนี้ ค่า cost จะมากกว่า 0 เพราะเป็น Neg Log Prob และคูณด้วย alpha ซึ่ง > 0

    #get normalized probability
    cost = tensor.nnet.softmax(-cost)[0] #eq. 13 ที่ใช้ -cost เพราะเราต้อง Normalize ตัว Prob จึงต้องแปลงจาก -log P ให้กลายเป็น log P ก่อน แล้วค่อย ทำ exp (log P) / sum( exp (log P )) ดังนั้นค่า Cost ที่เก็บคือ Normalized Prob จาก ทุกๆ sample ที่สุ่มออกมา 
    


    # risk: expected loss
    if options['mrt_ml_mix'] > 0 and not options['mrt_reference']:
        cost *= loss[1:]
    else:
        cost *= loss # คูณกับ Loss จาก MRT (คือ Baselin - BLEU)


    cost = cost.sum() #eq. 11-12

    if options['mrt_ml_mix'] > 0:
        #normalize ML by length (because MRT is length-invariant)
        ml_cost /= y_mask[:,0].sum(0)
        ml_cost *= options['mrt_ml_mix']
        cost += ml_cost

    return cost, loss
```

มาถึงตรงนี้เราจะพบว่าข้อมูลที่ต้องโหลดเข้า GPU จะเยอะมาก เพราะมันเท่ากับ Batch x Sampling size ดังนั้น จึงควรลดขนาด Batch ลงเพื่อให้ไม่เกิด Out-Of-Memory  ปกติ Batch Size สำหรับ MRT จะเท่ากับ 1

มาดูส่วนของการ Sampling ของ Nematus กันบ้าง 

(***จริงๆ ก็คือว่าถ้าเอา ประโยคเฉลยโยนเพิ่มเข้าไปเป็นส่วนหนึ่งใน Sampling ด้วยก็น่าจะดีไม่น้อย)

```python
if model_options['objective'] == 'MRT':
    xlen = len(x)
    n_samples += xlen

    assert maxlen is not None and maxlen > 0

    xy_pairs = [(x_i, y_i) for (x_i, y_i) in zip(x, y) if len(x_i) < maxlen and len(y_i) < maxlen]
    if not xy_pairs:
        training_progress.uidx -= 1
        continue

    for x_s, y_s in xy_pairs:

        # add EOS and prepare factored data
        x, _, _, _ = prepare_data([x_s], [y_s], maxlen=None,
                                    n_factors=factors,
                                    n_words_src=n_words_src, n_words=n_words)

        # draw independent samples to compute mean reward
        if model_options['mrt_samples_meanloss']:
            use_noise.set_value(0.)
            samples, _ = f_sampler(x, model_options['mrt_samples_meanloss'], maxlen)
            use_noise.set_value(1.)

            samples = [numpy.trim_zeros(item) for item in zip(*samples)]

            # map integers to words (for character-level metrics)
            samples = [seqs2words(sample, worddicts_r[-1]) for sample in samples]
            ref = seqs2words(y_s, worddicts_r[-1])

            #scorers expect tokenized hypotheses/references
            ref = ref.split(" ")
            samples = [sample.split(" ") for sample in samples]

            # get negative smoothed BLEU for samples
            scorer = ScorerProvider().get(model_options['mrt_loss'])
            scorer.set_reference(ref)
            mean_loss = numpy.array(scorer.score_matrix(samples), dtype=floatX).mean()
        else:
            mean_loss = 0.

        # create k samples
        use_noise.set_value(0.)
        samples, _ = f_sampler(x, model_options['mrt_samples'], maxlen)
        use_noise.set_value(1.)

        samples = [numpy.trim_zeros(item) for item in zip(*samples)]

        # remove duplicate samples
        samples.sort()
        samples = [s for s, _ in itertools.groupby(samples)]

        # add gold translation [always in first position]
        if model_options['mrt_reference'] or model_options['mrt_ml_mix']:
            samples = [y_s] + [s for s in samples if s != y_s]

        # create mini-batch with masking
        x, x_mask, y, y_mask = prepare_data([x_s for _ in xrange(len(samples))], samples,
                                                        maxlen=None,
                                                        n_factors=factors,
                                                        n_words_src=n_words_src,
                                                        n_words=n_words)

        cost_batches += 1
        last_disp_samples += xlen
        last_words += (numpy.sum(x_mask) + numpy.sum(y_mask))/2.0

        # map integers to words (for character-level metrics)
        samples = [seqs2words(sample, worddicts_r[-1]) for sample in samples]
        y_s = seqs2words(y_s, worddicts_r[-1])

        #scorers expect tokenized hypotheses/references
        y_s = y_s.split(" ")
        samples = [sample.split(" ") for sample in samples]

        # get negative smoothed BLEU for samples
        scorer = ScorerProvider().get(model_options['mrt_loss'])
        scorer.set_reference(y_s)
        loss = mean_loss - numpy.array(scorer.score_matrix(samples), dtype=floatX)

        # compute cost, grads and update parameters
        cost = f_update(lrate, x, x_mask, y, y_mask, loss)
        cost_sum += cost

```

จะเห็นว่า Nematus ใช้ mean_loss มาเป็น Baseline Reward สังเกต

```python
loss = mean_loss - numpy.array(scorer.score_matrix(samples), dtype=floatX)
```

ทำไมถึงเอา mean_loss มาลบ เพราะมันคือ

loss = - (loss - mean_loss)

เพราะ loss - mean_loss ==> reward แต่เราต้องการ loss ดังนั้น จึงต้องใส่ - เข้าไปข้างหน้าอีกทีหนึ่ง


```python
# build a training model
def build_model(tparams, options):

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy_floatX(0.))
    dropout = dropout_constr(options, use_noise, trng, sampling=False)

    x_mask = tensor.matrix('x_mask', dtype=floatX)
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype=floatX)
    # source text length 5; batch size 10
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype(floatX)
    # target text length 8; batch size 10
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype(floatX)

    x, ctx = build_encoder(tparams, options, dropout, x_mask, sampling=False)
    n_samples = x.shape[2]

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer_constr('ff')(tparams, ctx_mean, options, dropout,
                                    dropout_probability=options['dropout_hidden'],
                                    prefix='ff_state', activ='tanh')

    # every decoder RNN layer gets its own copy of the init state
    init_state = init_state.reshape([1, init_state.shape[0], init_state.shape[1]])
    if options['dec_depth'] > 1:
        init_state = tensor.tile(init_state, (options['dec_depth'], 1, 1))

    logit, opt_ret, _, _ = build_decoder(tparams, options, y, ctx, init_state, dropout, x_mask=x_mask, y_mask=y_mask, sampling=False)

    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])

    cost = (cost * y_mask).sum(0) # Cost ตรงนี้คือ cost ของแต่ละประโยคใน Batch ซึ่งเป็น negative log prob เรียบร้อยแล้ว เทียบเท่ากับ lprobs ใน fairseq ก่อน ติดลบ

    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost
```

มาดูโค้ดสำหรับ REINFORCE Algorithm แบบพื้นฐานที่สุดสำหรับการทำ Sequence Generation กันครับ

```python
"""
RNN Policy Gradient
"""

import torch
import torch.nn as nn

tgt_dict = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}

def n_grams(list_words):
    set_1gram, set_2gram, set_3gram, set_4gram = set(), set(), set(), set()
    count = {}
    l = len(list_words)
    for i in range(l):
        word = list_words[i]
        if word not in set_1gram:
            set_1gram.add(word)
            count[word] = 1
        else:
            set_1gram.add((word,count[word]))
            count[word] += 1
    count = {}

    for i in range(l-1):
        word = (list_words[i],list_words[i+1])
        if word not in set_2gram:
            set_2gram.add(word)
            count[word] = 1
        else:
            set_2gram.add((word,count[word]))
            count[word] += 1

    count = {}

    for i in range(l-2):
        word = (list_words[i],list_words[i+1], list_words[i+2])
        if word not in set_3gram:
            set_3gram.add(word)
            count[word] = 1
        else:
            set_3gram.add((word,count[word]))
            count[word] += 1
    count = {}

    for i in range(l-3):
        word = (list_words[i],list_words[i+1], list_words[i+2], list_words[i+3])
        if word not in set_4gram:
            set_4gram.add(word)
            count[word] = 1
        else:
            set_4gram.add((word,count[word]))
            count[word] += 1

    return set_1gram, set_2gram, set_3gram, set_4gram

def my_sentence_gleu(references, hypothesis):
    reference = references[0]
    ref_grams = n_grams(reference)
    hyp_grams = n_grams(hypothesis)
    match_grams = [x.intersection(y) for (x,y) in zip(ref_grams, hyp_grams)]
    ref_count = sum([len(x) for x in ref_grams])
    hyp_count = sum([len(x) for x in hyp_grams])
    match_count = sum([len(x) for x in match_grams])
    gleu = float(match_count) / float(max(ref_count,hyp_count))
    return float(gleu*100)

def letter2index(letter):
    global tgt_dict
    return tgt_dict[letter]

def index2letter(index):
    global tgt_dict
    for k in tgt_dict:
        if tgt_dict[k] == index:
            return k

def index2onehot(index,size):
    onehot = torch.zeros((size,))
    onehot[index] = 1
    return onehot

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)

n_letters = 4
n_categories = 4
n_hidden = 20
learning_rate = 0.01
rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)

def train_nll_step():
    global rnn, n_letters
    learning_rate = 0.01
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)
    hidden = rnn.initHidden()
    rnn.zero_grad()

    inputx = "A B C D".split(" ")
    input_index = [letter2index(x) for x in inputx]
    input_emb = [index2onehot(x,n_letters) for x in input_index]
    out_list = []

    outputy = torch.Tensor([[0],[1],[2],[3]]).long()

    for inp in input_emb:
        output, hidden = rnn(inp, hidden)
        out_list.append(output)
    
    out = torch.stack(out_list,dim=0)
    out = -torch.log(out)

    loss = torch.gather(out,1,outputy)
    loss = loss.sum()
    loss.backward()

    optimizer.step()

    return out, loss.item()

def sampling_out(prob):
    #print(prob)
    return torch.multinomial(prob,1)

def train_mrt_step():
    global rnn, n_letters
    hidden = rnn.initHidden()
    rnn.zero_grad()
    rnn.eval()
    inputx = "A B C D".split(" ")
    input_index = [letter2index(x) for x in inputx]
    input_emb = [index2onehot(x,n_letters) for x in input_index]
    

    outputy = torch.Tensor([[0],[1],[2],[3]]).long()
    
    out_list = []
    for inp in input_emb:
        output, hidden = rnn(inp, hidden)
        out_list.append(output)

    out = torch.stack(out_list,dim=0)

    #Sampling Baseline
    samples = []
    for k in range(0,4):
        sampling_x = sampling_out(out)
        samples.append(sampling_x.view(-1))
    
    samples = torch.stack(samples)
    #print(samples)

    samples = samples.data.tolist()
    bleu = []
    for sample in samples:
        score = my_sentence_gleu([[0,1,2,3]], sample)
        bleu.append(score)
    print("SAMPLE : ", sample)

    baseline = sum(bleu)/len(bleu)
    print(baseline)

    #Sampling in for optimization
    rnn.train()
    rnn.zero_grad()
    hidden = rnn.initHidden()
    out_list = []
    for inp in input_emb:
        output, hidden = rnn(inp, hidden)
        out_list.append(output)

    out = torch.stack(out_list,dim=0)


    samples = []
    for k in range(0,20):
        sampling_x = sampling_out(out)
        samples.append(sampling_x.view(-1))
    
    samples = torch.stack(samples)
    #print(samples)

    samples_L = samples.data.tolist()
    bleu = []
    for sample in samples_L:
        score = my_sentence_gleu([[0,1,2,3]], sample)
        bleu.append(score)
    
    bleu_diff = torch.Tensor(bleu) # Reward ที่ได้รับแปลผันตรงกับค่า BLEU Score

    print(bleu_diff)

    out = torch.stack(out_list,dim=0)
    out = -torch.log(out)

    sent_loss = []
    for sample in samples:
        loss = torch.gather(out,1,sample.view(-1,1))
        sent_loss.append(loss)
    
    sent_loss = torch.stack(sent_loss,dim=0)
    sent_loss = sent_loss.view(sent_loss.size(0),-1)
    
    alpha = 0.1
    sent_loss = sent_loss.sum(-1) * alpha # alpha คือ Temperature ทำบน Scale log เลยเป็นการคูณ แทนการยกกำลัง
    sent_loss = torch.softmax(-sent_loss, dim=0) #เพราะ sent_loss เป็น negative log มา เพราะจริงๆ คือ ใช้ normalized prob ไปคูณ Reward
    mrt_loss = sent_loss * bleu_diff #คูณกับ Reward เพื่อหา Expected Reward
    mrt_loss = mrt_loss.sum() # ได้ Expected Reward
    mrt_loss = -mrt_loss #ที่ต้องติดลบเพราะต้องการ Maximize แต่ optimizer.step() มันจะเป็นการ Minimize

    print(mrt_loss)
    mrt_loss.backward()
    optimizer.step()
    return out, mrt_loss.item()

if __name__ == "__main__":
    for i in range(0,500):
        print("ITER : ",i)
        out,loss = train_mrt_step()
        print(loss)
```


### Beyond

[Deep Reinforcement Learning For Sequence to Sequence Models](https://arxiv.org/abs/1805.09461)



