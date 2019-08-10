---
title: "Minimum Risk Training for NMT"
comments : true
layout: post
published: true
categories: Paper
---

### Introduction

MRT เป็นเทคนิคที่ใช้ REINFORCE Algorithm ในการหา Policy Gradient ซึ่ง หลักการก็คือ เค้าเปลี่ยนวิธีการหา loss ใหม่ แต่เดิมหา loss จาก Groundtruth โดยใช้ Maximum Likelihood ซึ่งมันไม่สอดคล้องกับการใช้ BLEU score


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
    cost = tensor.nnet.softmax(-cost)[0] #eq. 13 ที่ใช้ -cost เพราะว่ามันจะใช้ MRT_Loss เป็นตัวดันแทน Loss ดังนั้น ถ้า Cost ต่ำ แต่ Risk สูง เราจะต้องถือว่ามันมี MRT_Loss สูง

    #เช่น

    """
        A มี Risk เท่ากับ B = 1
        B cost = 2  ผ่าน softmax([-1 (A),-2 (B)]) = 0.26
        A cost = 1  ผ่าน softmax([-1 (A),-2 (B)]) = 0.73  
        หมายความว่ามันง่ายที่จะ Generate A ซึ่งมี Risk สูงพอๆ กับ B แต่ A มีโอกาสเกิดสูงกว่า ดังนั้น ควรให้ Loss กับ A มากกว่านั่นเอง

        จะเห็นว่า Risk กับ cost นั้นสัมพันธ์กัน ที่นี้ค่า Risk จะคำนวณอย่างไร ?

        เนื่องจากเราใช้ BLEU score ถ้า Risk สูง หมายถึง BLEU ต่ำๆ ดังนั้น ค่า Risk

        อาจจะเท่ากับ -BLEU ซึ่งพอไปคูณกับ cost ที่ผ่าน softmax แล้วจะทำให้ค่าติดลบไปอีก ดังนั้น เราจึงอาจจะให้ Risk = 1 - BLEU ซึ่งก็ Make sense ดี

        แต่วิธีการนี้เนื่องจากแต่ละประโยคให้ BLEU ที่มีความแตกต่างกันมากๆ เราจึงเรียกปัญหานี้ว่า High Variance วิธีการที่นิยมใช้คือ หา baseline โดย Sampling ออกมามากๆ แล้วค่อยหา BLEU เฉลี่ยนจาก Sampling นี้ จากนั้นค่อยหา Risk

        เพราะถ้าใช้ 1 - BLEU เลยจะไม่ยุติธรรมกับประโยคยากๆ ดังนั้นควรเอา 
        (BLEU ปัจจุบันหรือ baseline) - (BLEU ที่ Sampling ออกมา) ก็จะทำให้ได้ Risk ที่นิ่งขึ้น

        RISK = Baseline - BLEU sampling ซึ่ง Baseline ไม่ควรมีค่าน้อยเกินไป เพราะจะเท่ากับว่าไม่ได้อัพเดตเลย ดังนั้น ตัว RISK แบบนี้สามารถเป็น บวก หรือ ลบ ก็ได้

        ทีนี้ก็มีคำถามอีกว่า Scale ของ BLEU ควรอยู่ใน 0 - 1 หรือ 0 - 100
    """ 

    #me

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


ยังรันไม่ได้เลยต้องมาดูว่าหา loss ผิดตรงไหน

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

### Beyond

[Deep Reinforcement Learning For Sequence to Sequence Models](https://arxiv.org/abs/1805.09461)



