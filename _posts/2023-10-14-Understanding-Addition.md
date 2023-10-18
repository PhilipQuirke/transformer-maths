---
title: "Understanding Addition in Transformers"
date: 2023-10-14
---
# Introduction
If you type the question “When Mary and John went to the store, John gave a drink to” into Chat GPT, it will answer “Mary”.  
Chat GPT is a “Transformer” model that considers each word in the question and generates a predicted answer. 
You can also type in “12345+86764=” and it will give the right answer. How does it do this?

This blog explains how a toy (1-layer, 3-head) transformer model answers integer addition questions like:
![AdditionQuestionAndAnswer](/static/AdditionQuestionAnswer.svg?raw=true "AdditionQuestionAndAnswer")

This blog is written as an introduction to Mechanistic Interpretability and Transformer models for novices. 
It covers our investigation, testing and results of integer addition in transformers, building up section by section, and finally explaining this diagram:
![AdditionStaircaseA3Summary](/static/StaircaseA3_Summary.svg?raw=true "AdditionStaircaseA3Summary")

A CoLab notepad is provided here. 
It contains all the code needed to train the model and use the trained model, create graphs, etc. You can alter the code to test out other approaches.

# Humans vs Model Learning
When we were learning to do addition, we likely memorized some facts (e.g. 1+1=2) but quickly learnt this was not scalable and then learnt the standard way to do addition. 

Transformer models are trained by us providing them with many example questions and scoring the correctness of their answer. Initially their answers are random, but over the training they discover (by themselves) ways to do addition accurately (aka with low loss).

We might expect the model to use the human approach to addition: adding first the lowest-value digit-pair (D0 + D0’), noting whether this sum generated a “Carry 1”, then adding D1 + D1’ + D0’s Carry 1 (if any), etc. This approach is the easiest for us, but it is very sequential (D0 then D1 etc) with a strong time-ordering.

As our model train, it tries many possibly-useful approaches to many addition sub-tasks (e.g. D3 + D3’) in parallel. There is no overall coordination or time-ordering. The learning is more like evolution - using the answer scoring to prefer one approach over another. The below graph (from the CoLab Part 5) shows our model training over time - getting better at adding each of D0 to D4 digits - learning at a different speed for each digit. After training, the model has learnt a different (accurate) way to do addition than humans
