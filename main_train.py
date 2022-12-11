import ray

trainer_config = {
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
}


def main():
    print("Training with Ray Tune")
    ray.init()