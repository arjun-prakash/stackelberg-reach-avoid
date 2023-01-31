import gym
from gym import spaces
import numpy as np
from gym_examples.envs.dubins_car import TwoPlayerDubinsCarEnv


import jax
import jax.numpy as jnp
import haiku as hk
import optax


#generate data

env = TwoPlayerDubinsCarEnv()
# state = env.reset()
# X = []
# y = []
# done = False
# while not done:
#     for agent in env.players:

#         action = env.action_space[agent].sample() 
#         state, reward, done, info = env.step(action, agent)
#         X.append(state)
#         y.append(reward)
#         env.render()
# env.make_gif()



state = env.reset()
X = []
y = []
for i in range(100):
    for player in env.players:
        state = env.reset()
        action = env.action_space[player].sample() 
        X.append(np.hstack([state['attacker'], state['defender']]))
        r = env.sample(state, action, player,0.9)
        y.append(r)

X = np.array(X)
y = np.array(y)

print(X[:5])
print(y[:5])


#params are defined *implicitly* in haiku
def forward(X):
    l1 = hk.Linear(10)(X)
    l2 = hk.Linear(2)(l1)

    return l2


# a transformed haiku function consists of an 'init' and an 'apply' function
forward = hk.without_apply_rng(hk.transform(forward))

# initialize parameters
rng = jax.random.PRNGKey(seed=13)
params = forward.init(rng, X)

# redefine 'forward' as the 'apply' function
forward = forward.apply


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)


def update(params, grads):
    return jax.tree_map(lambda p, g: p - 0.05 * g, params, grads)


optimizer = optax.sgd(learning_rate=1e-2)

opt_state = optimizer.init(params)
for epoch in range(1000):
    loss, grads = jax.value_and_grad(loss_fn)(params,X=X,y=y)
    print("progress:", "epoch:", epoch, "loss",loss)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
# After training
print("estimation of the parameters:")
print(params)

test_state = env.reset()
test_state = np.hstack([test_state['attacker'], test_state['defender']])

estimate  = forward(X=test_state, params=params)
print("estimate", estimate)


state = env.reset()
done = False
max_iter = 1000
counter = 0
#max attacker, min defender

while (not done) and (counter < max_iter):
    counter+=1

    for player in env.players: #attacker, defender
        possible_actions = []
        for a in range(env.action_space[player].n):
            input = np.hstack([state['attacker'], state['defender']])

            estimate = forward(X=input, params=params)
            possible_actions.append(estimate)
        action = np.argmax(np.array(possible_actions))

    state, reward, done, _ = env.step(action)
    env.render()
    print(counter)
    
env.make_gif()
