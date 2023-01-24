import gym
from gym import spaces
import numpy as np
from gym_examples.envs.dubins_car import DubinsCarEnv


import jax
import jax.numpy as jnp
import haiku as hk
import optax


#generate data

env = DubinsCarEnv()
state = env.reset()
X = []
y = []
for i in range(10000):
    print(i)
    state = env.reset()
    for action in range(env.action_space.n):
        X.append(state)
        r = env.sample(state, action, 0.9)
        y.append(r)

X = np.array(X)
y = np.array(y)

#params are defined *implicitly* in haiku
def forward(X):
    l1 = hk.Linear(10)(X)
    l2 = hk.Linear(20)(l1)

    l3 = hk.Linear(1)(l2)

    return l3.ravel()


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


optimizer = optax.adam(learning_rate=1e-2)

opt_state = optimizer.init(params)
for epoch in range(500):
    loss, grads = jax.value_and_grad(loss_fn)(params,X=X,y=y)
    print("progress:", "epoch:", epoch, "loss",loss)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
# After training
print("estimation of the parameters:")
print(params)

estimate  = forward(X=env.reset(), params=params)
print("estimate", estimate)




# # the main training loop
# for _ in range(50):
#     loss = loss_fn(params, X_test, y_test)
#     print(loss)

#     grads = grad_fn(params, X, y)
#     params = update(params, grads)




env = DubinsCarEnv()
state = env.reset()
done = False
max_iter = 100
counter = 0
while (not done) and (counter < max_iter):
    counter+=1
    possible_actions = []
    for a in range(env.action_space.n):
        next_state, _, done, _ = env.state_action_step(state, a)
        estimate = forward(X=next_state, params=params)
        possible_actions.append(estimate[0])
    action = np.argmax(np.array(possible_actions))
    print(action, possible_actions )

    state, reward, done, _ = env.step(action)
    env.render()
    print(counter)
    
env.make_gif()