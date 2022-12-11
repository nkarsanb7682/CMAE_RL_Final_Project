import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from env.env_factory import get_env


ray.init()
ray.init(ignore_reinit_error=True, local_mode=True)
# registering custom environment
select_env = "push_box"
register_env(select_env, lambda config: get_env(select_env)())
config = ppo.DEFAULT_CONFIG.copy()
config["env"] = select_env
config["log_level"] = "WARN"
config["horizon"] = 250
config["num_workers"] = 0
config["exploration_config"] = {
    "type": "Curiosity",
    "eta": 0.1,
    "lr": 0.0003,
    "feature_dim": 64,
    "feature_net_config": {
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
    },
    "sub_exploration": {
        "type": "StochasticSampling",
    },
}
config["multiagent"] = {
    "count_steps_by": "env_steps",
}
agent = ppo.PPOTrainer(config, env=select_env)
#agent.restore("tmp/icm/checkpoint_000001/rllib_checkpoint.json")
agent.restore("tmp/icm/checkpoint_000001")

n_iter = 2000
min_iter_to_goal = float('inf')

for i in range(25, n_iter):
    result = agent.train()

    if i % 5 == 0:
        chkpt = agent.save('tmp/icm')
        print("checkpoint saved at", chkpt)
    if result["episode_reward_max"] > 0.0:
        print("Reached goal after {} iters!".format(i))
        min_iter_to_goal = min(min_iter_to_goal, i)


print(f"REACHED GOAL IN {min_iter_to_goal} ITERATIONS !!!")
policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())