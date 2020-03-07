# PyPlayetest

A library for implementing framework to do virtual playtesting for
boardgames.

* Provide basic components to make it easy to create new prototypes

* Wraps an API around the game you created to integrate with [OpenAI gym](https://gym.openai.com/)
  or a default integration harness.


# A Quick Demo

To see a demo of this game, clone this repo and run:

```
example/play.py mapt_Blackjack-v0
```

And you will be able to use our interactive harness to play the game.

You can observe the code that generates the game (in a few lines of python!)
in [pt-blackjack/game.py](pt-blackjack/game.py).


# Getting started

To get started, read the docs at [here](#todo).

```
pip install playtest
```

Then can just `import playtest` and get started in creating your prototype.
