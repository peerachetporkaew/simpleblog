---
title: "Ordered Neurons"
comments : true
layout: post
published: true
categories: Paper
---

### Ordered Neurons : Code Hacking

Natural language is hierarchically structured: smaller units (e.g., phrases) are nested within larger units (e.g., clauses). When a larger constituent ends, all of the smaller constituents that are nested within it must also be closed. While the standard LSTM architecture allows different neurons to track information at different time scales, it does not have an explicit bias towards modeling a hierarchy of constituents. This paper proposes to add such an inductive bias by ordering the neurons; a vector of master input and forget gates ensures that when a given neuron is updated, all the neurons that follow it in the ordering are also updated. Our novel recurrent architecture, ordered neurons LSTM (ON-LSTM), achieves good performance on four different tasks: language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.

[CODE](https://github.com/yikangshen/Ordered-Neurons)


![torch.cumsum](./assets/torch.cumsum.PNG)

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


