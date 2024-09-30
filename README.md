# catanProj

For the project we use Bryan Collazo's implementation of catan, which can be found at 

https://github.com/bcollazo/catanatron


## Our Files

Our Files Can be found at 
 catanProj/catanatron_experimental/catanatron_experimental/MichaelFiles

- MyHeuristic.py  - implementations of our heuristic agent

- HeuristicFeatures.py - helper functions for our heuristic player

- MyNNPlayer.py - implementations of our NN agent

- Net.py - implementations of the NN we used for MyNNPlayer

- TrainNN.py - the code we used to train the NN agent

- TrainNNHelpers.py - helper functions for TrainNN

- Feaatures.py - features used by the NN agent

- GenerateGameData.py - code used to generate data for supervised learning

- model_weights - directory containing the weights for the NN agent

--

--


Instructions to run our project with our agents:

## Installation

Clone this repository and install dependencies. This will include the Catanatron bot implementations and the `catanatron-play` simulator and our implementations for the heauristic agent and the NN agent.

```
git clone https://github.com/MichaelHuji/catanAI
cd catanProj/
```

Create a virtual environment with Python3.8 or higher. Then:

```
pip install -r requirements.txt
```

## Usage

Run simulations via the CLI:
Our heuristic search agent against catanatron's weighted random player
```
catanatron-play --players=W,MYVF --num=100
```

Our heuristic search agent against catanatron's value function player
```
catanatron-play --players=F,MYVF --num=100
```

Our Deep NN against catanatron's weighted random player
```
catanatron-play --players=W,NN --num=100
```

Our Deep NN against catanatron's value function player
```
catanatron-play --players=F,NN --num=100
```

Our single layer NN against catanatron's value function player
```
catanatron-play --players=F,NN91 --num=100
```

Our single layer NN against catanatron's Alpha-Beta
```
catanatron-play --players=AB,NN91 --num=50
```

Alpha-Beta with our single layer NN utility function against catanatron's Alpha-Beta
```
catanatron-play --players=AB,MYAB --num=50
```

2 of our single layer NN against catanatron's value function player
```
catanatron-play --players=F,NN91,NN91 --num=100
```

Our single layer NN against 3 of catanatron's value function player
```
catanatron-play --players=F,NN91,F,F --num=100
```

Our 3 agents against each other
```
catanatron-play --players=MYVF,NN91,NN --num=100
```


# Bot Legend

### Catanatron's bots

- R  - Catanatron's RandomPlayer  -  

- W  - Catanatron's WeightedRandom  -  

- VP - Catanatron's Greedy Victory Point Maximizer  -  

- M  - Catanatron's Monte Carlo Tree Search  -  

- F  - Catanatron's Value Function Agent  -  wins against MCTS with close to 100%

- AB - Catanatron's Value Function Agent  -  wins against F with about 60%
  

### Our bots

- MYVF - Our Heuristic Agent  -  wins against F with about 50%, wins against AB with about 40%

- NN - Our Deep NN Agent  -  wins against MCTS with about 95%, wins against F with about 12%

- NN91 - Our Single Layer NN Agent  -  wins against F with about 58%, wins against AB with about 55%
 
- MYAB - Our Single Layer NN with depth 2 search  -  wins against F with about 80%, wins against AB with about 70%

--


--
## Help

### (make sure you did cd catanProj and pip install -r all-requirements.txt)

See more information with `catanatron-play --help`.








--

--

--

