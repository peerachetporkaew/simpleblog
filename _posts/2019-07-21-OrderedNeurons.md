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


