import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions   # Making a shortcut
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # The first day has an 80% chance of being cold
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],   # A cold day has a 50% chance of being followed by a hot day
                                                 [0.2, 0.8]])  # A hot day has a 20% chance of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0.,15.], scale=[5., 10.]) # Temperature normal distributions
# loc is mean and scale in sd


model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=20)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())