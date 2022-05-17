# Can you predict the tide?

## Authors :
- [Akiyo Worou](mailto:akiyo.worou@student-cs.fr>)
- [Théophile Louvet](mailto:theophile.louvet@student-cs.fr>)
- [Tiago Teixeira ](mailto:tiago.teixeira@student-cs.fr>)


## Description

This challenge is proposed as a part of the [Challenge Data](https://challengedata.ens.fr/challenges/67).

The objective is to predict the sea surge (difference between predicted and real sea level) for 2 european coastal cities. The input available are the surge values in the past 5 days (measured every 12 hours), as well as the sea level pressure field (measured every 3 hours) which contains the data for 1681 points spanned over the north west atlantic. The model must predict the surge every 12 hours for the next 5 days.

## Installation

The files X_train, X_test and Y_train shared by the organizers shall be placed in the data folder. These files are not on the repository because of their large size (~2GB in total)

Python is needed, along with basic packages (numpy, scikit-learn...)

## Utilization

Using the code is straightforward: place yourself in the code directory, enter: `python main.py`
will train the best models of multiregression and display their results, entering:`python RNN.py <GRU|LSTM>`
will train a basic recurrent neural network with GRU or LSTM units, type: `python RNN.py --help`

to see information about the additionnal parameters
