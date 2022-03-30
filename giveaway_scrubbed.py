import random
import numpy as np

# These people all indicated they are located in the US/CA or had a means to
# ship to a US/CA address for forwarding.
us_ca_entrants = ["Rintze", "Hasan", "Keith", "Joseph", "Asceptt", "Brian",
                  "Xiaoyu", "Anik", "Devshank", "Jeremy", "Amin", "Brenton",
                  "Remi", "Howard", "Michael", "Khizr", "Jay", "Ricardo",
                  "Matt", "Chris", "Tanner", "Paul", "Pang", "Jose", "David",
                  "Kurt", "Jesse"]

# These people indicated they were in a foreign country and did not indicate
# they had the means to ship to a foreign address.
intl_entrants = ["Harsh"]

# These people did not indicate where they were or their means to forward mail
unknown_entrants = ["Gareth", "Dan", "Dileep", "Zeeshan", "Romin", "Dellan",
                    "Marcin", "Wouter", "Cecil", "Jamal", "Gabriel", "ATV",
                    "Violet", "Waqas", "Joy", "Tianqi", "Thomas"]

random.seed(2022)

gpu_winner = random.choice(us_ca_entrants)

all_entrants = us_ca_entrants + intl_entrants + unknown_entrants

nnai_winner = random.choice(all_entrants)

dli_winners = [random.choice(all_entrants) for _ in range(5)]

# Make sure there are no duplicate names, so there is no ambiguity in who won
assert len(np.unique(us_ca_entrants)) == len(us_ca_entrants)

assert len(np.unique(all_entrants)) == len(all_entrants)

print('GPU Winner:', gpu_winner)

print('NeuralNet.ai Subscription Winner:', nnai_winner)

print('Deep Learning Institute winners:', dli_winners)
