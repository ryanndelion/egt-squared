# Evolutionary Game Theory Squared - Evolving Agents in Endogenously Evolving Zero Sum Games

This repository contains supplementary code and material that was used to generate the simulations presented in our paper, 'Evolutionary Game Theory Squared - Evolving Agents in Endogenously Evolving Zero Sum Games'.

## Abstract
The predominant paradigm in evolutionary game theory and more generally online learning in games is based on a clear distinction between a population of *dynamic agents* that interact given a *fixed, static game*. In this paper, we move away from the artificial divide between dynamic agents and static games, to introduce and analyze a large class of competitive settings where both the agents and the games they play evolve strategically over time. We focus on arguably the most archetypal game-theoretic setting -- zero-sum games (as well as network generalizations) -- and the most studied evolutionary learning dynamic -- replicator, the continuous-time analogue of multiplicative weights. Populations of agents compete against each other in a zero-sum competition that itself evolves adversarially to the current population mixture. Remarkably, despite the chaotic coevolution of agents and games, we prove that the system exhibits a number of regularities. First, the system has *conservation laws* of an information-theoretic flavor that couple the behavior of all agents and games. Secondly, the system is *Poincare recurrent*, with effectively all possible initializations of agents and games lying on recurrent orbits that come arbitrarily close to their initial conditions infinitely often. Thirdly, the *time-average agent behavior and utility converge* to the Nash equilibrium values of the *time-average* game. Finally, we provide a polynomial time algorithm to efficiently predict this time-average behavior for any such coevolving network game.

## Codebase
All code is written in Python 3.6, and only requires basic scientific computing packages such as NumPy and SciPy as well as data visualization packages such as Matplotlib and Plotly to run. Most of the code has been edited so that it can easily be executed on any standard computer in a matter of minutes. Note that in this repository we have a version of the Jupyter Notebook without figures (due to space constraints), as well as a python file that contains our complete code (which can be converted to .ipynb locally) and a .html file that presents the visualizations.
