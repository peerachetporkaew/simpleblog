---
title: "Training with Sentence Oracle"
comments : true
layout: post
published: true
categories: Paper
---

**Training with Sentence Oracle**

โพสต์นี้จะมาแนะนำเปเปอร์ "Bridging the Gap between Training and Inference for Neural Machine Translation" ซึ่งจัดการกับปัญหา Exposure Bias ในการ Generate seqeunce ที่ยาวๆ เขาบอกว่าปัญหามันอยู่ที่ ตอนเทรนนั้นเราเทรนจาก y_{t-1} ที่ไม่เคยมี noise เลย ทำให้เวลา Inference หากเลือกไม่ถูกสักตัวหนึ่งแล้วก็จะทำให้ token ถัดๆ ไปเพี้ยนได้ ดังนั้น เวลาเทรนจึงควรเพิ่ม Noise เข้าไปด้วย 

(มีเปเปอร์ที่อาจจะมีประโยชน์ทำนองนี้อีกอันคือ **Soft Contextual Augmentation for Neural Machine Translation**)

ดังนั้นวิธีการที่เขาทำคือ 

> To select the sentence-level oracles, we first perform beam search for all sentences in each batch, as- suming beam size is k, and get k-best candidate translations. In the process of beam search, we also could apply the Gumbel noise for each word generation. We then evaluate each translation by calculating its BLEU score with the ground truth sequence, and use the translation with the highest BLEU score as the oracle sentence.

แทนที่จะใช้ Ground-truth ในการเทรนเพื่อหา Loss ปกติ เขาเปลี่ยนมาใช้ Sentence-level Oracle แทน โดยการทำ Beam search + Gumbel Noise  ทั้ง Batch แล้วเลือก Sentence ที่ค่า BLEU สูงสุดของแต่ละประโยคมาเป็นตัวแทน ของ y_{t-1} แต่ ตอนเทรน y* ยังคงใช้ Ground-truth อยู่ดี ทำให้เกิดปัญหาขึ้นว่า ถ้าความยาวประโยคไม่เท่ากันจะทำอย่างไร วิธีการที่เขาแก้ก็คือ Force Decoding เอาเลย

ตรงนี้มีประเด็นที่น่าสนใจอันหนึ่งคือ y_{t-1} ที่มาจาก Sentence Oracle นั้นเขาใช้ Embedding ตรงๆ เลย จริงๆ แล้วเราสามารถใช้ Soft Embedding ได้ โดยเอา softmax prob * Embedding Weight ไปตรงๆ ได้เลย วิธีนี้จะทำให้เราไม่ต้องทำ การ Force Decoding. สามารถอ้างอิง เปเปอร์ **Neural Machine Translation with Gumbel-Greedy Decoding** ได้ ซึ่งเราอาจจะไม่ต้องใช้ Generator / Discriminator ก็ได้ ไอเดียนี้น่าจะไปลองกับ MLE / MRT / OR-NMT ด้วย



