-- Written by Darian Hadjiabadi --


require 'nn'
require "batches"
require "util"

function CNN2()
    local cnn = nn.Sequential()
    cnn:add(nn.SpatialConvolutionMM(1,32,5,5))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2,2))

    cnn:add(nn.SpatialConvolutionMM(32,64,5,5))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2,2))

    cnn:add(nn.Reshape(64*5*5))
    cnn:add(nn.Linear(64*5*5,10))
    return cnn
end

function CNN1()
    local cnn = nn.Sequential()
    cnn:add(nn.SpatialConvolutionMM(1,32,5,5))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2,2))
    
    cnn:add(nn.Reshape(32*14*14))
    cnn:add(nn.Linear(32*14*14,10))
    return cnn
end

function LinClass()
    local lc = nn.Sequential()
    lc:add(nn.Reshape(1*32*32))
    lc:add(nn.Linear(1*32*32,10))
    return lc
end
