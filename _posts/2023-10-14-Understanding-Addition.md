---
title: "Understanding Addition in Transformers"
date: 2023-10-14
---
# Introduction
If you type the question “When Mary and John went to the store, John gave a drink to” into Chat GPT, it will answer “Mary”.  
Chat GPT is a “Transformer” model that considers each word in the question and generates a predicted answer. 
You can also type in “12345+86764=” and it will give the right answer. How does it do this?

This blog explains how a toy (1-layer, 3-head) transformer model answers integer addition questions like:

