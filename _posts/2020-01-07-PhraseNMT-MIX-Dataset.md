---
title: "Implementation of Fairseq for mix datasets"
comments : true
layout: post
published: true
categories: Code
---

### Introduction

เนื่องจากช่วงนี้จะต้องสร้างตัว PhraseNMT version 2 ที่ใช้หลักการของการ Masking แทนการใช้ sequence ไปตรงๆ ซึ่งตอนนี้ได้พัฒนาตัว PhraseNMT version 2 เรียบร้อยแล้ว และคิดว่าจะทำให้ระบบนี้สามารถเทรนเพื่อแปลแบบ Phrase และ Sentence ไปพร้อมๆ กันได้ในทีเดียว เพื่อลดจำนวน Parameter และอาจจะได้ Phrase Alignment มาเป็นของแถม ดังนั้นจึงต้องออกแบบ Fairseq ตัวใหม่ให้รองรับมากกว่า 1 dataset



### Specification

ปกติ data-bin จะมีชุดข้อมูลอยู่เพียงอันเดียว แต่เราจะเพิ่มเป็นสองชุด คือ data-bin/ds01 และ data-bin/ds02 โดย ds01 คือชุดข้อมูลแบบประโยค และ ds02 คือ ชุดข้อมูลแบบ phrase

### Implementation

สามารถค้นหาดูได้จากเครื่อง XiaoMi : /home/peerachet/Desktop/Code/PhraseNMTV2_MIX/MIXPhraseNMTV2.odp



