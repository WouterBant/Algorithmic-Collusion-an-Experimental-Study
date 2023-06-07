# BSc Thesis, Algorithmic Collusion: A Computational Study of Firms' CSR Investment Decisions

In progress


## Getting started
### Requirements

Install the dependencies with the following command:

```

pip install -r requirements.txt

```

## Structure

* All code can be found in [/src](src):

  * [/analysis_util](src/analysis_util): contains a [cycle classifier](src/analysis_util/cylcle_classifier.py) and functions used for [plotting](src/analysis_util/visualize.py).

  * [/classes](src/classes): contains the [Q-learning](src/classes/Qlearning.py), [DQN](src/classes/DQN.py) and [regulator](src/classes/regulator.py) agents, as well as the [economic environment](src/classes/environment.py) and the [action](src/classes/action.py) class.

  * [/runs](src/runs): contains the analysis of different runs and shows the figures presented in the paper.

  * [algorithms.py](src/algorithms.py): contains the algorithms used to simulate episodes in the different settings discussed in the papers.
