import random
import numpy as np

all_entrants = ['Henry Magregor', 'Charly B', 'Arjun H', 'Pete O', 'Lolis F',
                'Kaan A', 'Inosiro', 'Brian B', 'Ben C', 'Jorge B', 'Jesse G',
                'Hauke H', 'Pas D', 'Aditya C', 'Marc C', 'Logan G', 'Brian C',
                'Antemasq', 'Alex D', 'Bibek P', 'Andrew S', 'Gonzalo B',
                'Martin P', 'Bikash S', 'William P', 'Daniel A', 'Naomi G',
                'Alex V', 'Chris G', 'Steve L', 'Felix G', 'Greg K', 'x g',
                ]

random.seed(2023)

gpu_winner = random.choice(all_entrants)

all_entrants.remove(gpu_winner)

nnai_winner = random.choice(all_entrants)

all_entrants.remove(nnai_winner)

dli_winners = [random.choice(all_entrants) for _ in range(5)]

# Make sure there are no duplicate names, so there is no ambiguity in who won
assert len(np.unique(all_entrants)) == len(all_entrants)

print('GPU Winner:', gpu_winner)

print('NeuralNet.ai Subscription Winner:', nnai_winner)

print('Deep Learning Institute winners:', dli_winners)
