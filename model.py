#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:04:49 2023

@author: yusufsahin
"""

import torch
import torch.utils.data
import math

class RBM():
    
    def __init__(self, number_visible, number_hidden):
        self.number_visible = number_visible
        self.number_hidden = number_hidden
        
        #initialize weights
        self.weights = torch.randn((number_hidden,number_visible), dtype=torch.float64)
        self.a = torch.randn(size=(1, number_hidden), dtype=torch.float64)# two dimensional, first dim corresponds to the batch
        self.b = torch.randn(size=(1, number_visible), dtype=torch.float64)
    
    def sample_hidden(self, x):
        logit = torch.mm(x, self.weights.t())
        logit += self.a.expand_as(logit)
        probabilities = torch.sigmoid(logit)
        sampled_hidden = torch.bernoulli(probabilities)
        return probabilities, sampled_hidden
    
    def sample_visible(self, y):
        logit = torch.mm(y, self.weights)
        logit += self.b.expand_as(logit)
        probabilities = torch.sigmoid(logit)
        sampled_visible = torch.bernoulli(probabilities)
        return probabilities, sampled_visible
    
    def update_weights(self, v0, vk, probs_h_0, probs_h_k):
        self.weights += torch.mm(probs_h_0.t(), v0) - torch.mm(probs_h_k.t(), vk)
        self.b += torch.sum(v0 - vk, 0)
        self.a += torch.sum((probs_h_0 - probs_h_k), 0)
        
    def train(self, dataset, batch_size, num_cycles, epochs):
        losses = []
        num_batches = math.floor(len(dataset/batch_size))
        for epoch in range(epochs):
            for batch in range(0, num_batches, batch_size):
                data = dataset[batch:batch+batch_size,:]
                
                probs_h_0, _ = self.sample_hidden(data)
                data_k = data
                
                
                for cycle in range(1, num_cycles):
                    _, hidden = self.sample_hidden(data_k)
                    _, data_k = self.sample_visible(hidden)
                    data_k[data < 0] = data[data < 0]
                
                probs_h_k, _ = self.sample_hidden(data_k)
                
                
                self.update_weights(data, data_k, probs_h_0, probs_h_k)
                loss =  torch.mean(torch.abs(data[data >= 0] - data_k[data >= 0]))
                print(loss)
                losses.append(loss.item())
        return losses
    
    def test(self, trainset, testset):
        test_loss = 0
        num_tested_samples = 0
        for i in range(len(trainset)):
            train_sample = trainset[i:i+1]
            test_sample = testset[i:i+1]
            if len(test_sample[test_sample >= 0]) > 0:
                _, hidden = self.sample_hidden(train_sample)
                _, train_sample = self.sample_visible(hidden)
                test_loss += torch.mean(torch.abs(train_sample[test_sample >= 0] - test_sample[test_sample >= 0]))
                num_tested_samples += 1
        return test_loss/num_tested_samples
                
            
                
                    
        
        
        
            
    
    
        
        