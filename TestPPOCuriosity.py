import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune import register_env
from env.env_factory import get_env
import shutil
import csv
import os


class TestPPOCuriosity:

    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=3)

    def envCreator(self, env="push_box"):
        register_env(env, lambda config: get_env(env)())

    def test_curiosity_on_push_box(self):

        config = (
            ppo.PPOConfig()
            # Limit horizon to make it really hard for non-curious agent to reach
            # the goal state.
            .rollouts(num_rollout_workers=0)
            .training(lr=0.001)
            .exploration(
                exploration_config={
                    "type": "Curiosity",
                    "eta": 0.2,
                    "lr": 0.001,
                    "feature_dim": 128,
                    "feature_net_config": {
                        "fcnet_hiddens": [],
                        "fcnet_activation": "relu",
                    },
                    "sub_exploration": {
                        "type": "StochasticSampling",
                    },
                }
            )
        )

        num_iterations = 1000
        chkpt_root = "tmp/icm"
        shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
        if os.path.exists('ppo_icm_results.csv'): os.remove('ppo_icm_results.csv')
        for _ in framework_iterator(config, frameworks="torch"):
            # W/ Curiosity
            self.envCreator(env="push_box")
            algo = config.build(env="push_box")
            with open('ppo_icm_results.csv', 'w') as file:
                writer = csv.writer(file)
                for i in range(num_iterations):
                    result = algo.train()
                    max_reward = result["episode_reward_max"]
                    done = max_reward > 0
                    loss = result["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]
                    steps = result["info"]["num_env_steps_trained"]
                    curr_res = [i, max_reward, loss, steps, done]
                    writer.writerow(curr_res)
                    if i % 25 == 0:
                        chkpt_file = algo.save(chkpt_root)
                        print("checkpoint saved at: ", chkpt_file)
                    if max_reward > 0.0:
                        print("Reached goal after {} iters!".format(i))
            algo.stop()


driver = TestPPOCuriosity()
driver.setUpClass()
driver.test_curiosity_on_push_box()
