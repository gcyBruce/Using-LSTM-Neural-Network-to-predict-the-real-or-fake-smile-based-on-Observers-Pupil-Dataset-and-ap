Python code:
It is a LSTM neural network with Distinctiveness technical to prune the network.

Requirements: python 3, pytorch.   numpy, torch, sklearn.preprocessing.
 
Dataset:
It includes two excel documents for left eye pupil dilation and right eye pupil dilation (normalised to between 0 and 1). And each excel includes 19 sheets, L1, L3, L4, L5 and H1-H5 sheets are data for real smile. A1-A10 sheets are fake smile. And in each sheet, p3 participant’s data is not from Asia. however, the project focuses on Asia participants. There is 10 sec at 60 Hz = 600, but there is a few less due to data loss


