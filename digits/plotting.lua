-- Written by Darian Hadjiabadi --

require 'gnuplot'


-- 2.2 --

local iterations = torch.load('cnn2Iterations.dat')
local cnn2TrainingAccuracy = torch.load('cnn2Accuracy.dat')
local cnn1TrainingAccuracy = torch.load('cnn1Accuracy.dat')
local linClassTrainingAccuracy = torch.load('LinClassAccuracy.dat')


gnuplot.pngfigure('InitialTrainingReport.png')
gnuplot.plot(
	{'Lin-Class', iterations, linClassTrainingAccuracy, '-'},
	{'2-CNN', iterations, cnn2TrainingAccuracy, '-'},
    {'1-CNN', iterations, cnn1TrainingAccuracy, '-'}
)
gnuplot.xlabel('Training Iteration')
gnuplot.ylabel('Accuracy')
gnuplot.movelegend('right','bottom')
gnuplot.plotflush()

-- 2.3 --

local distances = torch.Tensor(torch.load('DistanceVector.dat'))
local linDistance = torch.Tensor(torch.load('LinearClassifierDistance.dat'))
local cnn2Distance = torch.Tensor(torch.load('CNN2Distance.dat'))
local cnn1Distance = torch.Tensor(torch.load('CNN1Distance.dat'))
gnuplot.pngfigure('InvarianceMeasurements.png')
gnuplot.plot(
	{'Lin-Class',distances,linDistance,'-'},
	{'2-CNN', distances, cnn2Distance, '-'},
	{'1-CNN', distances,cnn1Distance,'-'}
)
gnuplot.xlabel('Shift (pixels)')
gnuplot.ylabel('Averaged Score Error')
gnuplot.plotflush()

