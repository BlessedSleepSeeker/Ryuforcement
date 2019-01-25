# Ryuforcement : Fighting Games Reinforcement/Q-Learning AI

## Introduction

Ryuforcement is a fighting game AI.
Our mains objective are :
- Develop a Reinforcement/Q-Learning algorithm.<br>
- Train Ryuforcement, so he can become the Strongest Warrior !

Ryuforcement will learn to play Street Fighter II at first.
We're using Gym Retro to emulate Street Fighter II : Special Champion Edition, a Genesis game.

Ryuforcement plays Ryu. __**SHORYUKEN!!**__

A Note on "importing 1 potential game" "imported 0 games" combo.
When you try to import a game, Gym Retro check the **sha**'s rom.

You can find the required **sha** for the supported games here :

https://github.com/openai/retro/tree/master/retro/data/stable/ NAMEOFYOURGAME/rom.sha

Just find a rom with the same sha, and you're good to go !

The Street Fighter II rom we used : https://edgeemu.net/details-38235.htm

## Installation

Clone the repository where you want.
Check if

### Requirements

* Openai Gym Retro (https://github.com/openai/retro)

### PIP Package

```pip install virtualenv```

### Manual Installation

Comming soon...

## Quick Start
After installation,

```. Ryuforcement/bin/activate```

```./Ryuforcement.py```

## Documentation / Explanation
env.step() doc:
```_obs, _rew, done, _info = env.step(action)```
* Variable taken:
	* *action*: ```action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]```

	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Actions are defined with an array of 12.

	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Each row correspond a button pressed or not.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[0] is Medium Kick.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[1] is Light Kick.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[2] is ??.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[3] is ??.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[4] is ??.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[5] is Up.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[6] is Down.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[7] is Left.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[8] is Right.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[9] is Medium Punch.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[10] is Light Punch.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array[11] is Heavy Punch.

![alt text](https://raw.githubusercontent.com/Camille-Gouneau/Ryuforcement/master/img/InputManette.png)

* Variables returned:
	* *_obs*: array that represents the current screen : 
	
	<div style="text-align:left"><img src ="https://raw.githubusercontent.com/Camille-Gouneau/Ryuforcement/master/img/_obsTranformations.png" /></div>
	
	
	* *_rew*: Reward of the AI
	* *done*: bool who break the loop when the game is over
	* *_info*: ```{'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,'health': 176}```

## Key Features

Comming soon...

## Results

### Training Details

Comming soon...

### Copyright & licensing

Comming soon...

### Contact

- camille.gouneau@epitech.eu
- pierre-eloy.sylvestre@epitech.eu

### Known bugs

Comming soon...

### Credits & acknowledgments

Comming soon...

### Changelog & news

0.0.1 : Environnement is functionnal, thanks to Gym Retro

Look at the git logs for further details !
