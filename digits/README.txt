
Darian Hadjiabadi
=======================================================

The three networks are initialized in 'networks.lua'
Plots are provided in plots/ but can be recreated with program 'plotting.lua'

File execution order: 
	(1) $ th training.lua <args>
		This file can take in a combination of valid arguments from the set {"LinClass", "CNN1", "CNN2"}. This will train the appropriate models. If no argument is entered, all three models will be trained. CNN2 is intensive to train
		so I reccomend just training one at a time. Furthermore, this program will save the trained models and data for plotting in .dat files.		
	(2) $ th invariance.lua
		Program will run through the invariance analysis routine for all three models. The program will load in .dat model files that were created in (1), and output the results (.dat) so that it may be plotted.
	(3) $ th plotting.lua. 
		Takes .dat outputs from (1) and (2) to create and automatically save png plots.
	

I did not include the trained model save files because they are too large. The data necessary for plotting is provided, however, so plotting.lua can be called without having to first train.


