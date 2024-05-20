import d3rlpy

# get CartPole dataset
dataset, env = d3rlpy.datasets.get_cartpole()  

# setup CQL algorithm
cql = d3rlpy.algos.DiscreteCQLConfig().create(device='cpu')

# start training
cql.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    evaluators={
        'environment': d3rlpy.metrics.EnvironmentEvaluator(env), # evaluate with CartPole-v1 environment
    },
)


import gym
from gym.wrappers import RecordVideo

# start virtual display
# d3rlpy.notebook_utils.start_virtual_display()

# wrap RecordVideo wrapper
env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), './video')

# evaluate
d3rlpy.metrics.evaluate_qlearning_with_environment(cql, env)
d3rlpy.notebook_utils.render_video("video/rl-video-episode-0.mp4")

