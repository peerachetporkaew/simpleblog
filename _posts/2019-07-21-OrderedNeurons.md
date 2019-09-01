---
title: "Ordered Neurons"
comments : true
layout: post
published: true
categories: Paper
---

### อธิบายเรื่อง Ordered Neurons : Integrating Tree Structures into Recurrent Neural Networks

Natural language is hierarchically structured: smaller units (e.g., phrases) are nested within larger units (e.g., clauses). When a larger constituent ends, all of the smaller constituents that are nested within it must also be closed. While the standard LSTM architecture allows different neurons to track information at different time scales, it does not have an explicit bias towards modeling a hierarchy of constituents. This paper proposes to add such an inductive bias by ordering the neurons; a vector of master input and forget gates ensures that when a given neuron is updated, all the neurons that follow it in the ordering are also updated. Our novel recurrent architecture, ordered neurons LSTM (ON-LSTM), achieves good performance on four different tasks: language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.

ภาษาธรรมชาตินั้นถือว่าเป็นข้อมูลที่มีลำดับชั้น: หน่วยที่เล็กกว่า (เช่น นามวลี กริยาวลี) ประกอบกันขึ้นเป็นหน่วยที่ใหญ่ (เช่น อนุประโยค) เมื่อสิ้นสุดอนุประโยค แน่นอนว่าส่วนประกอบภายในที่เล็กกว่าก็จะต้องจบในอนุประโยคนั้น ถึงแม้ว่า LSTM แบบมาตรฐานสามารถที่จะติดตามข้อมูล ณ เวลาต่างๆ กันได้ แต่มันไม่มีการจดจำที่ชัดเจนเพื่อที่จะสร้างลำดับชั้นของส่วนประกอบภายใน (constituent) ได้ งานวิจัยนี้ได้เพิ่มส่วนขยายที่ทำให้เซลประสาทนั้นมีการทำงานที่เป็นลำดับขั้นไว้ภายใน กล่าวคือ เมื่อเซลประสาทหนึ่งใน input gate และ forget gate ถูกแก้ไข ก็จะทำให้เซลประสาทอื่นๆ ที่อยู่ภายใต้ (มีลำดับที่ต่ำกว่า) เซลประสาทนั้นได้รับการอัพเดตไปด้วย เราตั้งชื่อสถาปัตยกรรมของโครงข่ายประสาทเทียมแบบป้อนกลับลักษณะนี้ว่า Ordered Neurons LSTM (ON-LSTM) ซึ่งผลการทดสอบพบว่าให้ค่าความถูกต้องที่น่าพอใจในทั้ง 4 งาน ได้แก่ Language Modeling, Unsupervised Parsing, Targeted Syntactic Evaluation และ Logical Inference


เป้าหมายหลักของงานนี้ คือ constituency parsing ซึ่งทำยังไงให้ hidden state ของ memory cell นั้นสามารถ interpret ออกมาเป็น tree ได้ เราลองมาดูภาพประกอบต่อไปนี้กับครับ

(Figure 2 ใน Paper ON-LSTM)

ขออนุญาตข้ามมาส่วน Methodology กันเลย ซึ่งต่อจากนี้จะอธิบายตามแบบฉบับของผมเองนะครับ 

### ทบทวน LSTM แบบปกติกันก่อน

ปกติเมื่อพูดถึง Neural Network เราจะสนใจ Input vector และ Output vector ซึ่งนั่นก็เป็นหลักการโดยปกติ แต่พอกล่าวถึง Recurrent Neural Network แล้วนั่นคือ Output vector จะมาเป็น Input อีกอันหนึ่ง (จึงถูกเรียกว่า hidden state) คู่กับ Input ของข้อมูลด้วย ดังนั้น จึงเรียกว่า Recurrent ซึ่งหากมี Recurrent Cell หนึ่งชั้นก็คือ มีพารามิเตอร์ของ Recurrent เพียง 1 ชุดเท่านั้น และใช้ พารามิเตอร์นี้กับทุกๆ time step นี่ก็คือหลักการของ Shared weight ครับ

ทีนี้พอมาเป็น LSTM มันไม่ได้มีแค่ input , hidden แล้ว แต่ยังมี memory cell เพิ่มเข้ามาด้วย ซึ่งแต่ละ time step ในการประมวลผล มันจะอัพเดตตัว Memory cell ด้วยการลบ และใส่ ข้อมูลเข้าไปใหม่ ดังนั้น LSTM จึงถูกเรียกว่า Long Short-Term Memory ( ซึ่งผมคิดว่ามันน่าจะแปลว่า Memory แบบ Short-Term ที่ Long, ไม่ใช่ Long-Term and Short-Term Memory ! แต่ไม่แน่ใจว่าผู้พัฒนามีเจตนาเลือกใช้อันไหนนะครับ เป็น คหสต.)

> สามารถอ่านรายละเอียดแบบเต็มๆ ได้ที่ [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

ดังนั้นเวลาที่เราจะดู LSTM เราต้องดู input (x) , hidden (h) และ memory cell (C) สมการการอัพเดต h และ C เป็นดังนี้ครับ

![LSTM](./assets/lstm.jfif)

ซึ่งตัว ON-LSTM นั้นจุดหลักๆ เลย คือ การสร้าง Order ให้กับตัว Memory Cell ครับ กล่าวคือ LSTM ปกติในแต่ละครั้งของการอัพเดตค่า C แต่ละ Neuron สามารถถูกอัพเดตได้โดยอิสระต่อกัน แต่ใน ON-LSTM เขาอยากให้เป็นแบบนี้ คือ หากจะอัพเดต Neuron บนๆ จะต้องให้ Neuron ล่างๆ ถูกอัพเดตไปด้วย จึงเรียกว่ามีลำดับ （Order นั่นเอง) ทีนี้เขาทำยังไงให้ Gate ตัวใหม่นี้มันอัพเดตแบบมีลำดับชั้นได้ พระเอกของงานนี้คือ activation function ที่ชื่อ cumax

### Activatoin Function cumax()

ผู้วิจัยได้นำเสนอ activation function แบบใหม่ เพื่อให้การอัพเดตเป็นไปตามลำดับ ซึ่งนิยามของ cumax คือ

$$
\hat{g} = \mathrm{cumax}(...) = \mathrm{cumsum}(\mathrm{softmax}(...)),
$$

โดยที่ cumsum นั้นหมายถึง cumulative sum เราจะแสดงให้เห็นว่า vector g^ (hat) นั้นสามารถมองให้เป็น expectation ของ binary gate g = (0, ..., 0, 1, ..., 1) ได้ (ซึ่งในการคำนวณจริง g^ ไม่ใช่ binary gate เป็น vector จำนวนจริงที่บริเวณที่ด้านล่างจะเข้าใกล้ศูนย์ ด้านบนจะเข้าใกล้ 1 นั่นคือ ส่วนที่เป็น 0 จะเป็นเลขน้อยๆ ส่วนที่เป็น 1 จะมีค่าใกล้ 1 แต่เพื่อให้เข้าใจหลักการทำงานได้ง่ายจึงอธิบายด้วย binary gate) ซึ่ง binary gate นี้จะแบ่ง vector ออกเป็นสองส่วนคือ 0-segment และ 1-segment  

ทีนี้เราจะสมมติให้ตัวแปร d คือ ตำแหน่งที่เปลี่ยนจาก 0 เป็น 1 ตัวแรก ดังนั้นเราสามารถนิยามได้ว่า

$$
p(g_k=1) = p(d \leq k) = \sum_{i \leq k}p(d=i)
$$

ซึ่ง p(d=i) นั้น ก็คือ ที่มาจาก softmax ดังนั้น p(g_k=1) เท่ากับ cumulative sum ของ softmax(...) วิธีการนี้ทำให้ computation graph สามารถหา gradient ได้



 
[CODE](https://github.com/yikangshen/Ordered-Neurons)


[//]: #(![torch.cumsum](./assets/torch.cumsum.PNG))

วันนี้เราจะมาดูว่าส่วนประกอบภายในของ Ordered Neurons ทำงานอย่างไรกันนะครับ

self.ih คือ ทุก weight ของ Input เขาเอามา concat กันเพื่อให้ประหยัด operation ในการคำนวณ

self.hh คือ ทุก weight ของ Hidden ซึ่งใช้แบบมี dropout ด้วย ซึ่งก็ concat ทั้งหมดมาเหมือนกับของ Input



```python
class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]
```

n_chunk คือ จำนวน Chunk ซึ่ง chunk เดียวกันจะถูกคุมด้วย master gate เดียวกันครับ

มาดูที่ตัว LinearDropConnect ซึ่งพระเอกของตัว LinearDropConnect นี้ก็คือ dropout ถ้า dropout เป็น 0 นั่นแสดงว่าไม่ใช้ dropout ถ้าไม่ใช่ 0 มันจะสร้างตัว mask ขึ้นมาเพื่อปิด weight ของตัวมันเองบางตัว (LinearDropConnect extend มาจาก nn.Linear) เทคนิคการทำ sample_mask เป็นเทคนิคที่น่าสนใจครับ เพราะว่าระหว่างนี้มันไม่ใช้ in_place operation เลย (หากมี in_place operation ก็จะทำให้ไม่สามารถ backward ได้ครับ)

```python
def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)
```

ในส่วนของ forward ก็มีเทคนิคสำหรับจัดการการทำ dropout เช่นกัน ซึ่งระหว่าง Train เราจะต้อง sample mask ไปเรื่อยๆ แต่เวลาที่ Test เราเปิดหมด ดังนั้นเราจะต้องชดเชยด้วยการลดขนาดของ weight ลงไปตามสัดส่วนของ dropout เพื่อไม่ให้มัน expose มาเกินไป

```python
return F.linear(input, self.weight * (1 - self.dropout), self.bias)
```

เอาหล่ะที่นี้มาดูตัว foward ของ Order Neurons กันครับ

```python
def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        # เริ่มจากการ Transform input
        if transformed_input is None:
            transformed_input = self.ih(input)

        # ส่วนนี้คือการ Transform ส่วน hidden
        gates = transformed_input + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:].view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        # คำนวณ master input gate และ forget gate
        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        # ค่า d คือ Expectation ของ g_k ซึ่งก็คือ จุดแบ่งระหว่าง 0 กับ 1 นั่นเอง self.n_chunk ความจริง คือ cforgetgate.size(-1) นั่นเอง แต่เอาค่าคงที่มาใช้ เพื่อให้ประหยัดการคำนวณไม่ต้องคำนวณแบบ Tensor
        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * F.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)
```

หากเรา print ค่าออกมาจะพบว่าค่าของ Gate นั้นไม่ใช่ 0, 1 อย่างที่ใน paper อธิบาย แต่ผู้เขียนใช้ 0,1 อธิบายเพื่อให้เห็นภาพและเข้าใจง่ายเท่านั้นเอง จริงๆ แล้วเป็นจำนวนจริงทั้งหมด และสุดท้ายเอาค่า d ซึ่งเป็น expectation ของ g_k มาทำต่อเท่านั้นเอง


ระหว่างนี้ผมได้ไปลองรันระบบจนได้โมเดลที่ดีระดับนึงออกมา 

ทีนี้มาส่วนของการ Parse กันบ้าง สมมติว่า Train จนได้โมเดลเสร็จเรียบร้อยแล้ว ซึ่ง Structure ของ ON-LSTM ที่ใช้ทำ Language Modeling นั้นเป็นแบบ 3 Layer ดังนั้น การหา distance สำหรับการ Parse จึงสามารเลือกได้ว่าจะมาจากชั้นใด

กรณีนี้ใช้ชั้นที่ 2 ดังนั้นจะได้ดังนี้ 

```python
def test_one(model,corpus,sen,cuda,prt):
    model.eval()
    sen = sen.split(" ")
    word2idx = corpus.dictionary.word2idx
    x = numpy.array([word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen])
    input = Variable(torch.LongTensor(x[:, None]))
    
    if cuda:
        input = input.cuda()

    hidden = model.init_hidden(1)
    _, hidden = model(input, hidden)

    distance = model.distance[0].squeeze().data.cpu().numpy()
    distance_in = model.distance[1].squeeze().data.cpu().numpy()

    gates = distance[1]
    depth = gates[1:-1]
    parse_tree = build_tree(depth, sen_cut)

    return MRG(parse_tree).strip()
```

จะเห็นว่าตัวแปร gates นั้นเลือกใช้จาก distance[1] เพื่อนำมาสร้าง depth และ parse ออกมาเป็น Tree อีกที


### English

some sort of 

conversely

downstream tasks

Interestingly

observed data

open question

learning the underlying grammar

More recently

attempts to perform

is controlled by a learnt

the task of ... is known as ...

Our work is more closely related to ..., which propose to ...

Given these requirements, we introduce

In our model, 

In other words, ...

The ... is further controlled by ...


