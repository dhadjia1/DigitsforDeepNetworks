-- Written by Darian Hadjiabadi --

require 'nn'
require "networks"
require "util"
require "batches"

function train(network, numSteps, trainBatch, learningRate, trainingData, testingData, testBatch)
	
    local mnistTrain = torch.load(trainingData,'b64')
    local mnistTest = torch.load(testingData,'b64')
    local accuracyTable = {}
    local iterationTable = {}
    for i = 1,numSteps do
        if i % 50 == 0 then
            local testImages, testLabels = mnistTest:getNextBatch(testBatch)
            local preds = network:forward(testImages)
			table.insert(accuracyTable,accuracy(preds,testLabels))
			table.insert(iterationTable,i)
		end
		local image,labels = mnistTrain:getNextBatch(trainBatch)
		local scores = network:forward(image)
		local crit = nn.CrossEntropyCriterion()
		local loss = crit:forward(scores,labels)
		local dScores = crit:backward(scores,labels)
		network:backward(image,dScores)
		network:updateParameters(learningRate)
		network:zeroGradParameters()
	end
	return accuracyTable, iterationTable
end

function test(network, fileName, bSize)
	local mnistTest = torch.load(fileName, 'b64')
	local testImages, testLabels = mnistTest:getNextBatch(bSize)
	local preds = network:forward(testImages)
	return accuracy(preds,testLabels)

end

function setContains(table,element)
	for _,value in pairs(table) do
		if value == element then
			return true
		end
	end
	return false
end


local trainingData = "./data/trainingData.t7"
local testData = "./data/testData.t7"

-- check that files exists --
if not fileExists(trainingData) then
	print('Training data does not exist...exiting')
	return
end
if not fileExists(testData) then
	print('Testing data does not exist...exiting')
	return
end


local trainingSteps = 1000
local trainBatch = 100
local learningRate = 0.05
local testBatch = 1000
local finalTestingBatch = 1000

local arguments = {}
if #arg == 0 then
	print('No command line arguments entered, will proceed to train all three models')
	arguments = {"LinClass","CNN1","CNN2"}
else
	arguments = arg
end

if setContains(arguments,"LinClass") then
	-- Linear classifier training and testing
	local linClass = LinClass()
	local preAccuracyLinClass = test(linClass, testData, finalTestingBatch)
	print(string.format('LinClass accuracy before training: %f percent', preAccuracyLinClass*100))
	local accuracyTableLinClass, iterationTable = train(linClass,trainingSteps,trainBatch,learningRate,trainingData,testData,testBatch)
	torch.save('LinClassTrained.dat',linClass)
	local aTLinClass = torch.Tensor(accuracyTableLinClass)
	torch.save('LinClassAccuracy.dat',aTLinClass)
	local postAccuracyLinClass = test(linClass, testData, finalTestingBatch)
	print(string.format('LinClass accuracy after training: %f percent', postAccuracyLinClass*100))
	linClass = nil
end

if setContains(arguments,"CNN1") then
	-- 1-CNN training and testing --
	local cnn1 = CNN1()
	local preAccuracyCNN1 = test(cnn1, testData, finalTestingBatch)
	print(string.format('1-CNN accuracy before training: %f percent', preAccuracyCNN1*100))
	local accuracyTableCNN1, iterationTable = train(cnn1,trainingSteps,trainBatch,learningRate,trainingData,testData,testBatch)
	torch.save('cnn1Trained.dat', cnn1)
	local aTCNN1 = torch.Tensor(accuracyTableCNN1)
	torch.save('cnn1Accuracy.dat',aTCNN1)
	local postAccuracyCNN1 = test(cnn1, testData, finalTestingBatch)
	print(string.format('1-CNN accuracy after training: %f percent', postAccuracyCNN1*100))
	cnn1 = nil
end

if setContains(arguments,"CNN2") then
	-- 2-CNN training and testing --
	local cnn2 = CNN2()
	local preAccuracyCNN2 = test(cnn2, testData, finalTestingBatch)
	print(string.format('2-CNN accuracy before training: %f percent', preAccuracyCNN2*100))
	local accuracyTableCNN2, iterationTable = train(cnn2,trainingSteps,trainBatch,learningRate,trainingData,testData,testBatch)
	torch.save('cnn2Trained.dat', cnn2)
	local aTCNN2 = torch.Tensor(accuracyTableCNN2)
	local iT = torch.Tensor(iterationTable)
	torch.save('cnn2Accuracy.dat',aTCNN2)
	torch.save('cnn2Iterations.dat',iT)
	local postAccuracyCNN2 = test(cnn2, testData, finalTestingBatch)
	print(string.format('2-CNN accuracy after training: %f percent', postAccuracyCNN2*100))
	cnn2 = nil
end
