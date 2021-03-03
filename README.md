# continuous control
This project is a solution for collaboration and competition problem, where an unity environment is controled by artificial brain.

This environment have 8 continuous states and 2 continuous actions.

The states corresponds to the position and velocity of the ball and racket.

The actions corresponds to movement and jumping of the racket.

The environment is considered solved if the mean of last 100 episodes is more than or equal 0.5 points.

The task is episodic and in each episode the maximum score of the two agents are taken.

The agent will receive +0.1 points if hits the ball over the net and receive -0.01 if hits the ball out of bounds or ball hit the ground.


![plot image](tennis.png)


A requirements file is disponibilized for install necessary libs, just use pip install -r requirements.txt
download unity agent and paste in navigation directory.

Run Tennis.ipynb on jupyter notebook and follow instructs.

For solve this a reinforcement learning networks was implemented, the hiperameters was seted by comparing score graph.
After train is possible to see an smart agent.
