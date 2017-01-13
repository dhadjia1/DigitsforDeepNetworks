require "util"

local Dataset = torch.class('Dataset')

function Dataset:__init(images, labels, imSideLengths)
   self.images = images
   self.labels = labels 
   self.imSideLengths = imSideLengths
   self.currInd = 1
end

function Dataset:getNextBatch(batchSz)
   -- Pull next batch-sized chunk of data starting at self.currInd
   -- increment self.currInd so that the next batch starts at the right 
   -- place. If you reach the end of the dataset, circle back around.
   local images = torch.Tensor(batchSz,1,self.imSideLengths[1],self.imSideLengths[2])
   local labels = torch.IntTensor(batchSz)
   local counter = 1
   while counter <= batchSz do
      images[counter] = self.images[self.currInd]
      labels[counter] = self.labels[self.currInd] 
      -- print(self.labels[self.currInd]) 
      if self.currInd == self.labels:size()[1] then
         self.currInd = 1
      else
         self.currInd = self.currInd+1
      end
      counter = counter+1
   end
   return images, labels
end

if not(fileExists("./data/trainingData.t7")) or not(fileExists("./data/testData.t7")) then
   trainImages = torch.load('./data/trainImages.t7')
   trainLabels = torch.load('./data/trainLabels.t7')
   testImages = torch.load('./data/testImages.t7')
   testLabels = torch.load('./data/testLabels.t7')
   local imSideLengths = {32, 32}
   local mnistTrain = Dataset.new(trainImages, trainLabels, imSideLengths)
   local mnistTest = Dataset.new(testImages, testLabels, imSideLengths)

   torch.save("./data/trainingData.t7", mnistTrain)
   torch.save("./data/testData.t7", mnistTest)
end

