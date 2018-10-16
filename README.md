# Youtube-Code-Repository
Repository for all the code from my youtube channel
You can find me at https://youtube.com/MachineLearningWithPhil <br>

<h2> Kaggle/Venus-Volcanoes </h2>

My crude implementation of a convolutional neural network to perform image classification on data gathered <br>
by the Magellan spacecraft. The data is horribly skewed, as most images do not contain a volcano. <br>
This means we'll have to do some creative data engineering for our model training. <br>
Please note that in the test set, 84.1% of the data is "no volcano", and our model returns <br>
an accuracy of around 88%, which is better than a model that outputs straight 0s for predictions. <br>

You can check out the video for this at https://youtu.be/Ki-xOKydQrY <br>
You can find the data for this project at https://www.kaggle.com/fmena14/volcanoesvenus/home
<h2> ReinforcementLearning/DeepQLearning </h2>

My implementation of the Deep Q learning algorithm in PyTorch. Here we teach the algorithm to play the game of space invaders. I haven't had enough time to train this model yet, as it takes quite some time even on my 1080Ti / i7 7820k @ 4.4 GHz. I'll train
longer and provide a video on how well it does, at a later time.

The blog post talking about how Deep Q learning works can be found at http://www.neuralnet.ai/coding-a-deep-q-network-in-pytorch/ <br>
Video for this is at https://www.youtube.com/watch?v=RfNxXlO6BiA&t=2s



<h2> CNN.py </h2>

Simple implementation of a convolutional neural network in TensorFlow, version 1.5. <br>
Video tutorial on this code can be found here https://youtu.be/azFyHS0odcM <br>
Achieves accuracy of 98% after 10 epochs of training <br>
Requires data from http://yann.lecun.com/exdb/mnist/ <br>

<h2> ReinforcementLearning/blackJack-no-es.py </h2>

Implementation of Monte Carlo control without exploring starts in the blackjack environment from the OpenAI gym. <br>
Video tutorial on this code can be found at https://youtu.be/e8ofon3sg8E <br>
Algorithm trains for 1,000,000 games and produces a win rate of around 42%, loss rate of 52% and draw rate of 6% <br>

<h2> ReinforcementLearning/blackJack-off-policy.py </h2>

Implementation of off policy Monte Carlo control in the blackjack environment from the OpenAI gym. <br>
Video tutorial on this code can be found at https://youtu.be/TvO0Sa-6UVc <br>
Algorithm trains for 1,000,000 games and produces a win rate of around 29%, loss rate of 66% and draw rate of 5% <br>

<h2> ReinforcementLearning/cartpole_qlearning.py </h2>

Implementation of the Q learning algorithm for the cart pole problem. Code is based on the course by lazy programmer,  <br>
which you can find here <a href="https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/q_learning.py"> here </a>  <br>
Video tutorial on this code can be found at https://youtu.be/ViwBAK8Hd7Q <br>

<h2> ReinforcementLearning/doubleQLearning.py </h2>

Implementation of the double Q learning algorithm in the cart pole environment. This is based on my course on  <br>
reinforcement learning, which you can find at <a href="https://github.com/philtabor/Reinforcement-Learning-In-Motion/tree/master/Unit-8-The-Mountaincar"> this repo </a> <br>
Video tutorial on this code can be found https://youtu.be/Q99bEPStnxk <br>

<h2> ReinforcementLearning/sarsa.py </h2>

Implementation of the SARSA algorithm in the cart pole environment. This is based on my course on reinforcement learning,  
which can be found <a href="https://github.com/philtabor/Reinforcement-Learning-In-Motion/tree/master/Unit-7-The-Cartpole"> here </a> <br>
Video tutorial on this code can be found at https://youtu.be/P9XezMuPfLE <br>
