---
title: "Minimum Risk Training for NMT"
comments : true
layout: post
published: true
categories: Research
---

### Introduction

MRT เป็นเทคนิคที่ใช้ REINFORCE Algorithm ในการหา Policy Gradient ซึ่ง หลักการก็คือ เค้าเปลี่ยนวิธีการหา loss ใหม่ แต่เดิมหา loss จาก Groundtruth โดยใช้ Maximum Likelihood ซึ่งมันไม่สอดคล้องกับการใช้ BLEU score


แต่ว่าจะเอา BLEU Score มา Optimize ตรงๆ ทำไม่ได้ เพราะว่าไม่สามารถทำ Backpropagation ได้ ดังนั้นจึงต้องใช้ REINFORCE Algorithm หลักการคือการหา Expected Risk loss ซึ่ง Risk loss คืออะไรก็ได้ ยกตัวอย่างเช่น (1-BLEU) ยิ่ง BLEU เยอะ Risk ก็จะน้อย

ก็หา Expectation ก็จะเริ่มจาก Sampling Y' ออกมาจาก X จำนวนหนึ่ง แล้วหา Risk Loss เอา Risk Loss (sentence-level) ไปคูณกับ cost((Y',X)) แล้วค่อยรวมทั้งหมดอีกที 

ปัญหาของ วิธีการนี้อยู่ที่ตัว Risk ที่อาจจะมี variance สูง ดังนั้นจึงต้องทำการ Normalize เสียก่อน ซึ่งอาจมีการกำหนดค่า temperature (&alpha;) ไว้ด้วย

มาดูสมการกัน

![MRT](assets/mrt_equation.png)

ตัวอย่างการใช้ MRT และ temperature ของ Nematus

loss คือ  &#9651;(y,y^(s)) มันคือ loss ที่เกิดจาก Sampling นั่นเอง ซึ่งเป็น Sentence-level vector

```python
def mrt_cost(cost, y_mask, options):
    loss = tensor.vector('loss', dtype=floatX) 
    alpha = theano.shared(numpy_floatX(options['mrt_alpha']))

    if options['mrt_ml_mix'] > 0:
        ml_cost = cost[0]

        # remove reference for MRT objective unless enabled
        if not options['mrt_reference']:
            cost = cost[1:]

    cost *= alpha

    #get normalized probability
    cost = tensor.nnet.softmax(-cost)[0] #eq. 13

    # risk: expected loss
    if options['mrt_ml_mix'] > 0 and not options['mrt_reference']:
        cost *= loss[1:]
    else:
        cost *= loss


    cost = cost.sum() #eq. 11-12

    if options['mrt_ml_mix'] > 0:
        #normalize ML by length (because MRT is length-invariant)
        ml_cost /= y_mask[:,0].sum(0)
        ml_cost *= options['mrt_ml_mix']
        cost += ml_cost

    return cost, loss
```

มาถึงตรงนี้เราจะพบว่าข้อมูลที่ต้องโหลดเข้า GPU จะเยอะมาก เพราะมันเท่ากับ Batch x Sampling size ดังนั้น จึงควรลดขนาด Batch ลงเพื่อให้ไม่เกิด Out-Of-Memory 

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

