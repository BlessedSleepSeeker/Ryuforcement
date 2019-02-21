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

```pip install retro-gym```

### Manual Installation

Comming soon...

## Quick Start
After installation,

```python3 -m retro.import /path/to/your/ROMs/directory/```

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

Possible actions are :

- Movements :
Neutral = 		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Right = 		[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Left = 			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Crouch = 		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Crouch Right = 	[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
Crouch Left = 	[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
Jump Neutral = 	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Jump Right =	[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
Jump Left =		[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

- Normals :

Standing Low Punch =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Standing Medium Punch =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
Standing High Punch =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Standing Low Kick =		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Standing Medium Kick =	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Standing High Kick =	[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Doesn't seems to work

Crouching Low Punch =		[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
Crouching Medium Punch =	[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
Crouching High Punch =		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
Crouching Low Kick =		[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Crouching Medium Kick =		[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Crouching High Kick =		[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] # Doesn't seems to work

Specials :

This still need research

Slow Hadoken =		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Medium Hadoken =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Fast Hadoken =		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Slow Shoryuken =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Medium Shoryuken =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Fast Shoryuken =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Slow Tatsumaki =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Medium Tatsumaki =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Fast Tatsumaki =	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Throw :

Left Throw
Right Throw

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

<div style="text-align:left"><img src ="https://raw.githubusercontent.com/Camille-Gouneau/Ryuforcement/master/img/TBagFramePerfect.gif" /></div>
