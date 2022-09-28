import random
import numpy as np

# This time I didn't specify to indicate where you are located and if you had
# the means to ship abroad. Silly oversight on my part, but that's life.
# This means I'll do the drawing and email everyone first. If someone overseas
# wins the GPU but can't get it shipped, then I'll subtract their name and draw
# again.

all_entrants = ['xiaoyu', 'sunil', 'kelvin', 'jacob', 'sean', 'dilith',
                'noctildon', 'lukas_k', 'alex', 'matt_t', 'inosiro',
                'f1datadiver', 'sambaran', 'dean_v_a', 'balaji', 'aditya',
                'brian_cu', 'sim', 'philip', 'antonio', 'roumen', 'marc',
                'william_p', 'michael_f', 'behnood', 'lucas_p', 'ahmed_k',
                'jamal_c', 'luciano_d', 'amir-ul', 'kinal', 'sidhanath',
                'lorenzo', 'michael_w', 'ravi_j', 'brigliano', 'hrovje',
                'daniel_b', 'terry_w', 'jun', 'kurt_b', 'hauke', 'super_dave',
                'george', 'lukas_d', 'waleed', 'clark', 'frak', 'ravi_c',
                'sawaiz', 'ferran', 'jack-ziad', 'christian_g', 'zxavier',
                'daniel_k', 'akash', 'jbene', 'hause', 'jack', 'cristiano',
                'nguyen_q_d', 'tatonata', 'dennis_f', 'till_z', 'dusan',
                'abdennacer', 'antonio_p', 'dilan', 'adam_b', 'brian_co',
                'k_ali', 'matt_r', 'navoda', 'doyun', 'william_s', 'jed_j',
                'bijay', 'bruno', 'shivam', 'arjun_h', 'emil', 'abdulla_m',
                'nick', 'joyce_w', 'abhinav', 'alex_v', 'ruturaj_s']

random.seed(2022)

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
