# Ryuforcement : Street Fighter II Reinforcement AI

## Introduction

Ryuforcement is a project of reinforcement learning AI designed for playing Street Fighter

ROM used : https://edgeemu.net/details-38235.htm

## Installation

### Requirements

* Openai Gym Retro (https://github.com/openai/retro)

### PIP Package

Comming soon...

### Manual Installation

Comming soon...

## Quick Start
After installation,

```. Ryuforcement/bin/activate```

```./Ryuforcement.py```

## Documentation / Explanation
env.step() doc:
```_obs, _rew, done, _info = env.step(yeet)```
* Variable taken:
	* *yeet*: ```yeet = [0,0,0,0,0,0,0,0,0,0,0,0]```
	
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

![alt text](https://raw.githubusercontent.com/Camille-Gouneau/Ryuforcement/master/temp/InputManette.png)

* Variables returned:
	* *_obs*:
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

Comming soon...

### Known bugs

Comming soon...

### Credits & acknowledgments

Comming soon...

### Changelog & news

Comming soon...
