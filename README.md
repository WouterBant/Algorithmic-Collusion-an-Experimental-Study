<h1 align="center">Algorithmic Collusion: An Experimental Study of Firms' CSR Investment Decisions</h1>

## About
This repo contains the code used to obtain the results from the paper: Algorithmic Collusion: An Experimental Study of Firms' CSR Investment Decisions. This paper researches algorithmic collusion of Artificial Intelligence Algorithms in a duopoly setting where firms let these algorithms completely decide their production quantity and investment in Corporate Social Responsibility.

## Structure

* All code can be found in [/src](src):

  * [/analysis_util](src/analysis_util): contains a [cycle classifier](src/analysis_util/cylcle_classifier.py) and functions used for [plotting](src/analysis_util/visualize.py) and to get the [statistics](src/analysis_util/statistics.py) presented in the paper.

  * [/classes](src/classes): contains the [Q-learning](src/classes/Qlearning.py), [DQN](src/classes/DQN.py) and [regulator](src/classes/regulator.py) agents, as well as the [economic environment](src/classes/environment.py) and the [action](src/classes/action.py) class.

  * [/runs](src/runs): contains the analysis of different runs and shows the figures presented in the paper.

  * [algorithms.py](src/algorithms.py): contains the algorithms used to simulate episodes in the different settings discussed in the paper.


## Getting started
### Requirements

Install the dependencies with the following command:

```

pip install -r requirements.txt

```

### Usage
To use an algorithm just navigate to one of the files in [/runs](src/runs) and try the example or something else. Note that most plots from the paper require the simulation data available at [Google Drive](https://drive.google.com/file/d/1snpfTBvhV1hD44AK2DQ8_w3vwsQILzYP/view?usp=sharing). When you use simulate_episodes an h5 file called 'simulation_data' should be created or placed inside the [/data](data) directory.

## License
MIT   
