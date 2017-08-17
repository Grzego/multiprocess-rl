### Variation of Asynchronous RL in Keras (Theano backend) + OpenAI gym [1-step Q-learning, n-step Q-learning, A3C]
This is a simple variation of [asynchronous reinforcement learning](http://arxiv.org/pdf/1602.01783v1.pdf) written in Python with Keras (Theano backend). Instead of many threads training at the same time there are many processes generating experience for a single agent to learn from. 

### Explanation
There are many processes (tested with 4, it should work better with more in case of Q-learning methods) which are creating experience and sending it to the shared queue. Queue is limited in length (tested with 256) to stop individual processes from excessively generating experience with old weights. Learning process draws from queue samples in batches and learns on them. In A3C network weights are swapped relatively fast to keep them updated.

### Currently implemented and working methods
* [1-step Q-learning](https://github.com/Grzego/async-rl/tree/master/q-learning-1-step)
* [n-step Q-learning](https://github.com/Grzego/async-rl/tree/master/q-learning-n-step)
* [A3C](https://github.com/Grzego/async-rl/tree/master/a3c)

### Requirements
* [Python 3.4/Python 3.5](https://www.python.org/downloads/)
* [Keras](http://keras.io/)
* [Theano](http://deeplearning.net/software/theano/) ([Tensorflow](https://www.tensorflow.org/) would probably work too)
* [OpenAI (atari-py)](https://gym.openai.com/)
* `pip3 install scikit-image h5py scipy`

### Sample game (A3C)
![](https://github.com/Grzego/async-rl/blob/master/a3c/resources/sample-game.gif?raw=true)

#### Feedback
Because I'm newbie in Reinforcement Learning and Deep Learning, feedback is very welcome :)

### Note
* Weights were learned in Theano, so loading them in Tensorflow may be a little problematic due to Convolutional Layers.
* If training halts after few seconds, don't worry, its probably because Keras lazily compiles Theano function, it should resume quickly.
* Each process sets its own compilation directory for Theano so compilation can take very long time at the beginning (can be disabled with `--th_comp_fix=False`)

### Useful resources
* [Asyncronous RL in Tensorflow + Keras + OpenAI's Gym](https://github.com/coreylynch/async-rl)
* [Replicating "Asynchronous Methods for Deep Reinforcement Learning"](https://github.com/muupan/async-rl)
* [David Silver's "Deep Reinforcement Learning" lecture](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/pdf/1602.01783v1.pdf)
* [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/pdf/1312.5602v1.pdf)

