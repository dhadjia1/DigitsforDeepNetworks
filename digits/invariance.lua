-- Written by Darian Hadjiabadi --

require 'nn'
require "util"


-- Get V_center for all images -- 
function centerPredictions(network, fileName)
	local predictionTable = {}
    local images = torch.load(fileName,'b64')
    local preds = network:forward(images):clone()
	table.insert(predictionTable,normalize(preds))
    return predictionTable
end

-- Acquire V_shift for each shift and for each image, then calculate average distance --
function batchDistance(network,fileName,central)
    local imageCell = torch.load(fileName,'b64')
	local distanceVector = {}
    for i = 1,#imageCell do
        local images = imageCell[i]
		local preds = network:forward(images)
		table.insert(distanceVector,avgDistance(central[1],normalize(preds)))
    end
    return distanceVector
end


function acquireDifferenceVector(cD,lD,rD,mF)
	local network = torch.load(mF)
	local centerPreds = centerPredictions(network,cD)
	local leftDistanceVector = batchDistance(network,lD,centerPreds)
	local rightDistanceVector = batchDistance(network,rD,centerPreds)
	local centerDistance = avgDistance(centerPreds[1],centerPreds[1])	
	
	local completeDistanceVector = {}
	for i = 1,#leftDistanceVector do
	    table.insert(completeDistanceVector,leftDistanceVector[#leftDistanceVector-i+1])
	end
	table.insert(completeDistanceVector,centerDistance)
	for i = 1,#rightDistanceVector do
		table.insert(completeDistanceVector, rightDistanceVector[i])
	end
    return completeDistanceVector
end

function checkData(data)
    for i =1,#data do
		if not fileExists(data[i]) then
			return false
		end
	end
	return true
end



local centerData = "./data/translations/center.t7"
local leftData = "./data/translations/leftShifts.t7"
local rightData = "./data/translations/rightShifts.t7"
local linModelFile = 'LinClassTrained.dat'
local cnn2ModelFile = 'cnn2Trained.dat'
local cnn1ModelFile = 'cnn1Trained.dat'

-- check that files exists --

if not checkData({centerData, leftData, rightData, linModelFile, cnn1ModelFile, cnn2ModelFile}) then
    print('One of the data files names does not exist... exiting')
	return
else
	print('All files present')
end

local completeDistanceVector = {}
local distance = {-5,-4,-3,-2,-1,0,1,2,3,4,5}
torch.save('DistanceVector.dat',distance)

-- Linear Classifier --
local linearCompleteDistanceVector = acquireDifferenceVector(centerData,leftData,rightData,linModelFile)
torch.save('LinearClassifierDistance.dat',linearCompleteDistanceVector)

-- 1-CNN --
local cnn1CompleteDistanceVector = acquireDifferenceVector(centerData,leftData,rightData,cnn1ModelFile)
torch.save('CNN1Distance.dat',cnn1CompleteDistanceVector)

-- 2-CNN --
cnn2CompleteDistanceVector = acquireDifferenceVector(centerData,leftData,rightData,cnn2ModelFile)
torch.save('CNN2Distance.dat',cnn2CompleteDistanceVector) 


