import random
from utils.convert2base import s_to_sp, int_to_obs
import numpy as np
import collections
from scipy.stats import entropy
import scipy
from utils import pq
import bisect

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm
from utils.convert2base import obs_to_int_pi

from HER_DQN import HER_DQN

class Storage(object):
    def __init__(self, max_size=50000):
        self.state = np.zeros(max_size, dtype=np.int32)
        self.next_state = np.zeros(max_size, dtype=np.int32)
        self.action = np.zeros(max_size, dtype=np.int32)
        self.reward = np.zeros(max_size)
        self.done = np.zeros(max_size)
        self.time_stamp = np.zeros(max_size, dtype=np.int32)
        self.size = 0
        self.iterator = 0
        self.t = 0
        self.max_size = max_size

    def insert(self, s, a, r, ns, done):
        self.time_stamp[self.iterator] = self.t
        self.state[self.iterator] = s
        self.next_state[self.iterator] = ns
        self.action[self.iterator] = a
        self.reward[self.iterator] = r
        self.done[self.iterator] = done
        self.size += 1
        self.t += 1
        self.iterator += 1
        self.iterator = self.iterator % self.max_size
        self.size = min(self.max_size, self.size)
        self.t = 0 if done else self.t


    def get_all_data(self):
        inds = reversed(range(self.size))
        for i in inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

    def get_prioritized_batch(self, bz=512): # 512 for pass/secret/push
        data_inds = []
        # For DEBUG
        pos_rew_inds = np.where(self.reward >= 1)
        if pos_rew_inds[0].size == 0:
            return
        for ind in pos_rew_inds:
            ind = ind[0]
            t = self.time_stamp[ind]
            data_inds.extend(list(range(ind - t, ind + 1)))
        data_inds.reverse()
        random_inds = random.choices(list(range(self.size)), k=bz - len(data_inds))
        data_inds.extend(random_inds)
        for i in data_inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

    def get_random_batch(self, bz=512): # 512 for pass/secret/push
        random_inds = random.choices(list(range(self.size)), k=bz)
        for i in random_inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]



    def reset(self):
        self.state[:] = 0
        self.next_state[:] = 0
        self.action[:] = 0
        self.reward[:] = 0
        self.done[:] = 0
        self.time_stamp[:] = 0
        self.size = 0
        self.t = 0
        self.iterator = 0

    def batch_insert(self, s, a, r, ns, done):
        """
        Batch insert data. Don't keep track of time stamp. Assume storage will not be full.
        Should only be used for goal_replay_buffer
        """
        bz = len(s)
        if self.iterator + bz < self.max_size:
            self.state[self.iterator : self.iterator + bz] = s
            self.next_state[self.iterator : self.iterator + bz] = ns
            self.action[self.iterator : self.iterator + bz] = a
            self.reward[self.iterator : self.iterator + bz] = r
            self.done[self.iterator : self.iterator + bz] = done
            self.iterator += bz
            self.size += bz
        else:
            right_len = self.max_size - self.iterator
            left_len = bz - right_len
            self.state[self.iterator: self.iterator + right_len] = s[: right_len]
            self.next_state[self.iterator: self.iterator + right_len] = ns[: right_len]
            self.action[self.iterator: self.iterator + right_len] = a[: right_len]
            self.reward[self.iterator: self.iterator + right_len] = r[: right_len]
            self.done[self.iterator: self.iterator + right_len] = done[: right_len]

            self.state[:left_len] = s[right_len:]
            self.next_state[:left_len] = ns[right_len:]
            self.action[:left_len] = a[right_len:]
            self.reward[:left_len] = r[right_len:]
            self.done[:left_len] = done[right_len:]
            self.size = self.max_size
            self.iterator = left_len


    def get_goal_tractories(self, goal: (int, int), goal_replay_buffer: 'Storage', base: int):
        goal_s, goal_a = goal
        goal_s_inds = np.where(self.state == goal_s)
        if goal_s_inds[0].size == 0:
            return False
        if goal_s_inds[0].size == 0:
            goal_s = s_to_sp(goal_s, base=base)
            goal_s_inds = np.where(self.state == goal_s)

        for ind in goal_s_inds:
            ind = ind[0]
            t = self.time_stamp[ind]
            goal_replay_buffer.batch_insert(self.state[ind - t: ind], self.action[ind - t: ind],
                                            self.reward[ind - t: ind], self.next_state[ind - t: ind],
                                            self.done[ind - t: ind])
            goal_replay_buffer.insert(goal_s, goal_a, 1, goal_s, 1)
        return True

    def sample_states(self, bz):
        inds = np.random.choice(range(self.size), size=min(bz, self.size), replace=False)
        return self.state[inds], inds
        #return np.random.choice(self.state[:self.size], size=min(bz, self.size), replace=False)


class QLearning(object):
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2,
                 all_subspace=False, tree_subspace=False, subspace_q_size=10):
        self.q_table = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.count = np.zeros((n_states, n_actions))
        nvec = observation_space.nvec
        self.n_vars = len(observation_space.nvec)
        self.nvec = nvec
        if all_subspace:
            self.counts = [[np.zeros((var_range, n_actions)) for var_range in nvec],  # 1 var
                           [np.zeros((nvec[0] * nvec[1], n_actions)) for _ in
                            range(scipy.special.comb(len(nvec), 2, exact=True))]  # 2 var
                           ]
        if tree_subspace:
            self.counts = [[np.zeros((var_range, n_actions)) for var_range in nvec]]  # 1 var
            for i in range(1, self.n_vars):
                self.counts.append([None for _ in range(scipy.special.comb(self.n_vars, i + 1, exact=True))])

            self.subspace_q = pq.PQ()
            norm_h, level = 0, 0
            for count_id in range(self.n_vars):
                self.subspace_q.push((norm_h, level, count_id))

            def create_counter_space_mapping():
                from itertools import combinations
                counter_space_mapping = {i: {} for i in range(self.n_vars)}
                space_counter_mapping = {}
                vars = list(range(self.n_vars))
                for level in range(self.n_vars):
                    combs = list(combinations(vars, level + 1))
                    for counter_id, comb in enumerate(combs):
                        counter_space_mapping[level][counter_id] = comb
                        space_counter_mapping[comb] = counter_id
                return counter_space_mapping, space_counter_mapping
            # self.counter_space_mapping[level][counter_id] --> list of variable indices
            self.counter_space_mapping, self.space_counter_mapping = create_counter_space_mapping()
            self.subspace_q_size = subspace_q_size
            self.subspaces = None
        else:
            self.counts = [
                [np.zeros((var_range, n_actions)) for var_range in nvec],  # level 0
                [np.zeros((nvec[0] * nvec[1], n_actions)), np.zeros((np.prod(nvec[4:]), n_actions))],  # level 1
                [np.zeros((np.prod(nvec[:4]), n_actions)), np.zeros((nvec[0] * nvec[1] * np.prod(nvec[4:]), n_actions))],  # level 2
                [np.zeros((n_states, n_actions))]  # level 3
            ]
        self.base, self.raw_dim = base, raw_dim

    def update_q(self, s, a, r, s_next, done):
        if not done:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] \
                             + self.alpha * (r + self.gamma * np.max(self.q_table[s_next]))
        else:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] + self.alpha * r

    def select_action(self, s, eps, other_q_table=None, alpha=None):
        if np.random.rand() < eps:
            return np.random.choice(self.n_actions)
        else:
            # break tie uniformly
            if other_q_table is None:
                return np.random.choice(np.flatnonzero(self.q_table[s] == self.q_table[s].max()))
            else:
                q_table = (1 - alpha) * self.q_table[s] + alpha * other_q_table[s]
                return np.random.choice(np.flatnonzero(q_table == q_table.max()))

    def update_count(self, s, a, multilevel=False, all_subspace=False, tree_subspace=False):
        self.count[s, a] += 1
        if multilevel:
            self.update_counts(s, a, all_subspace, tree_subspace)

    def update_counts(self, s, a, all_subspace, tree_subspace):
        """
        Take joint state int (level 3) and update all counts in self.counts
        """
        # convert to raw obs
        raw_s = int_to_obs(s, self.base, self.raw_dim)
        if tree_subspace:
            if self.subspaces is None:
                self.subspaces = self.subspace_q.nsmallest(self.subspace_q_size)
            for subspace in self.subspaces:
                _, level, counter_id = subspace
                self.update_one_counter(raw_s, a, level, counter_id)
        else:
            for level in range(len(self.counts)):
                self.update_level_count(raw_s, a, level, all_subspace)

    def update_one_counter(self, raw_s, a, level, counter_id):
        rep = self.obs_to_subspace_reps(raw_s, level, counter_id)
        self.counts[level][counter_id][rep.item()][a] += 1

    def obs_to_subspace_reps(self, raw_s, level, counter_id):
        subspace_vars = self.counter_space_mapping[level][counter_id]
        rep = 0
        for i, var in enumerate(subspace_vars):
            rep += raw_s[:, var] * self.base**i
        return rep

    def update_level_count(self, raw_s, a, level, all_subspace):
        level_reps = obs_to_level_int(raw_s, self.base, self.raw_dim, level=level, all_subspace=all_subspace)
        for i, level_rep in enumerate(level_reps):
            assert level_rep.size == 1
            self.counts[level][i][level_rep.item()][a] += 1


class ExpBonusQLearning(QLearning):
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2, bonus_coef=1):
        super(ExpBonusQLearning, self).__init__(n_states, n_actions, base, raw_dim, observation_space, gamma, alpha)
        self.bonus_coef = bonus_coef

    def update_q(self, s, a, r, s_next, done):
        assert self.count[s, a] > 0
        bonus = 1 / np.sqrt(self.count[s, a])
        r = r + self.bonus_coef * bonus
        if not done:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] \
                                 + self.alpha * (r + self.gamma * np.max(self.q_table[s_next]))
        else:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] + self.alpha * r


class ActiveQLearning(QLearning):
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2, goal_q_len=300,
                 all_subspace=False, no_range_info=False, stochastic_select_subspace=False, tree_subspace=False,
                 recip_t=50, subspace_q_size=10, replay_size=50000, level_penalty=False, priority_sample=True):
        super(ActiveQLearning, self).__init__(n_states, n_actions, base, raw_dim, observation_space, gamma, alpha,
                                              all_subspace=all_subspace, tree_subspace=tree_subspace,
                                              subspace_q_size=subspace_q_size)
        self.replay_buffer = Storage(replay_size)
        self.goal_replay_buffer = Storage(replay_size)  # store trajectories that related to the goal
        self.base = base
        self.goal_q = collections.deque(maxlen=goal_q_len)
        self.n_updates = 0
        self.level = 0
        self.count_id = 0
        self.compute_ent_every = 20
        self.no_range_info = no_range_info
        self.stochastic_select_subspace = stochastic_select_subspace
        self.tree_subspace = tree_subspace
        self.recip_t = recip_t
        self.level_penalty = level_penalty
        self.priority_sample = priority_sample

    def compute_ent_subspaces(self, subspaces):
        """
            compute ent of subspaces
        """
        norm_ents = []
        for subspace in subspaces:
            _, level, count_id = subspace
            count = self.counts[level][count_id]
            if self.no_range_info:
                range = np.count_nonzero(count)
            else:
                range = count.size
            ent = entropy(count.reshape(-1)) / np.log(range)
            norm_ents.append(ent)
        return norm_ents

    def compute_ent(self):
        """
        compute all counts' ent
        :return: all counts' ent (1d list), selected projected space (level, count_id)
        """
        norm_ents = []
        selected_level, selected_count_id = None, None
        min_ent = 10000000
        for level, level_counts in enumerate(self.counts):
            for count_id, count in enumerate(level_counts):
                if self.no_range_info:
                    range = np.count_nonzero(count)
                else:
                    range = count.size
                ent = entropy(count.reshape(-1)) / np.log(range)
                norm_ents.append(ent)
                if ent < min_ent:
                    min_ent = ent
                    selected_level = level
                    selected_count_id = count_id
        return norm_ents, (selected_level, selected_count_id)

    def compute_ent_all(self):
        """
        compute all counts' ent
        :return: all counts' ent (1d list), selected projected space (level, count_id)
        """
        norm_ents = []
        for level, level_counts in enumerate(self.counts):
            for count_id, count in enumerate(level_counts):
                if count is None:
                    ent = None
                else:
                    if self.no_range_info:
                        range = np.count_nonzero(count)
                    else:
                        range = count.size
                    ent = entropy(count.reshape(-1)) / np.log(range)
                norm_ents.append(ent)
        return norm_ents

    def select_projectd_space(self, ents):
        """
        :param ents: normalized entropy of all counts
        :return: selected projected space (level, count_id)
        """
        raise NotImplemented

    def get_goal_proj(self, level, count_id, bz=1024, all_subspace=False):
        """
        Goal is the (s, a) with c(s) > 0 and c(s, a) is least of proj(level, count_id) in a sampled batch.
        TODO: Extend to returing a list of goals
        :param level:
        :param count_id:
        :return: goal
        """
        count = self.counts[level][count_id]
        states, state_inds = self.replay_buffer.sample_states(bz=bz)
        raw_s = int_to_obs(states, self.base, self.raw_dim)
        proj_ints = self.obs_to_subspace_reps(raw_s, level, count_id)
        count = count[proj_ints]
        # select the states/action with least count
        goal_s_ids, goal_as = np.where(count == np.min(count))
        goal_ss = states[goal_s_ids]
        goal_idx = np.random.randint(0, len(goal_ss))
        goal_s = goal_ss[goal_idx]
        goal_a = goal_as[goal_idx]
        return goal_s, goal_a

    def insert_data(self, s, a, r, s_next, done, th=10000):
        # print('s', s)
        # print('a', a)
        #print('self.count', self.count)
        if self.count[s, a] < th:
            self.replay_buffer.insert(s, a, r, s_next, done)

    def get_goal(self, multilevel=False, all_subspace=False):
        self.n_updates += 1
        if multilevel:
            if self.n_updates % self.compute_ent_every == 0:
                if self.tree_subspace:
                    self.subspaces = []
                    for _ in range(min(self.subspace_q_size, len(self.subspace_q))):
                        self.subspaces.append(self.subspace_q.pop())
                    norm_ents = self.compute_ent_subspaces(self.subspaces)

                    for i in range(len(self.subspaces)):
                        self.subspaces[i] = (norm_ents[i], self.subspaces[i][1], self.subspaces[i][2])
                        self.subspace_q.push(self.subspaces[i])
                    # print("self.stochastic_select_subspace", self.stochastic_select_subspace)
                    self.stochastic_select_subspace = True
                    if self.stochastic_select_subspace:
                        level_penalty = np.array([space[1] * self.level_penalty for space in self.subspaces])
                        exp_ent = np.exp(-self.recip_t * np.array(norm_ents) + level_penalty)
                        #print('exp_ent', exp_ent / np.sum(exp_ent))
                        selected_space_ids = np.random.multinomial(1, exp_ent / np.sum(exp_ent))
                        selected_space = self.subspaces[selected_space_ids.nonzero()[0].item()]
                        self.level, self.count_id = selected_space[1], selected_space[2]
                    else:
                        raise NotImplementedError

                    # Grow the search tree
                    if self.level < self.n_vars - 1:
                        subspace = self.counter_space_mapping[self.level][self.count_id]
                        for var in range(self.n_vars):
                            if var in subspace:
                                continue
                            subspace_new = list(subspace)
                            bisect.insort(subspace_new, var)
                            count_id = self.space_counter_mapping[tuple(subspace_new)]
                            if self.counts[self.level + 1][count_id] is None:
                                subspace_var_dims = self.nvec[np.array(subspace_new, dtype=np.int32)]
                                self.counts[self.level + 1][count_id] = \
                                    np.zeros((np.prod(subspace_var_dims), self.n_actions))
                                self._init_counter_from_storage(self.level + 1, count_id)
                                self.subspaces.append((0, self.level + 1, count_id))
                                self.subspace_q.push(self.subspaces[-1])
                else:
                    norm_ents, (self.level, self.count_id) = self.compute_ent()

                    if self.stochastic_select_subspace:
                        exp_ent = np.exp(-self.recip_t * np.array(norm_ents))
                        selected_space = np.random.multinomial(1, exp_ent / np.sum(exp_ent))
                        selected_space = selected_space.nonzero()[0].item()

                        if selected_space >= len(self.counts[0]):
                            self.level, self.count_id = 1, selected_space - len(self.counts[0])
                        else:
                            self.level, self.count_id = 0, selected_space

            goal_s, goal_a = self.get_goal_proj(self.level, self.count_id, all_subspace=all_subspace)
        else:
            states, state_inds = self.replay_buffer.sample_states(bz=1024)
            count = self.count[states]
            goal_s_ids, goal_as = np.where(count == np.min(count))
            goal_ss = states[goal_s_ids]
            goal_idx = np.random.randint(0, len(goal_ss))
            goal_s = goal_ss[goal_idx]
            goal_a = goal_as[goal_idx]

        return goal_s, goal_a

    def _init_counter_from_storage(self, level, count_id):
        for i in range(self.replay_buffer.size):
            a = self.replay_buffer.action[i]
            raw_s = int_to_obs(self.replay_buffer.state[i], self.base, self.raw_dim)
            self.update_one_counter(raw_s, a, level, count_id)

    def _shape_replay_buffer(self, goal=None):
        self.replay_buffer.reward[:] = 0
        if goal is None:
            state_count = np.sum(self.count, axis=1)
            state_count = np.tile(state_count, (self.count.shape[1], 1)).transpose()
            count = np.where(state_count != 0, self.count, np.full_like(self.count, np.inf))
            goal_s, goal_a = np.where(count == np.min(count))

            goal_idx = np.random.randint(0, len(goal_s))

            goal_s = goal_s[goal_idx]
            goal_a = goal_a[goal_idx]
        else:
            goal_s, goal_a = goal
        # add hallucinated dummy terminal state
        self.replay_buffer.insert(goal_s, goal_a, 1, goal_s, 1)

    def update_q_from_D(self, epoch=1, reset_q=True, goal=None):
        goal_s = goal[0] if goal is not None else -1
        goal_a = goal[1] if goal is not None else -1
        success = True
        if reset_q:
            self.reset_q()
            self.goal_replay_buffer.reset()
            success = self.replay_buffer.get_goal_tractories(goal, self.goal_replay_buffer, self.base)
        if success:
            for _ in range(epoch):
                if goal is not None:
                    data = self.goal_replay_buffer.get_all_data()
                else:
                    if self.priority_sample:
                        data = self.replay_buffer.get_prioritized_batch()
                    else:
                        data = self.replay_buffer.get_random_batch()
                for s, a, r, ns, done in data:
                    if goal is not None:
                        if s == goal_s and a == goal_a:
                            r = 1
                        else:  # zero external reward
                            r = 0
                    self.update_q(s, a, r, ns, done)

    def _prioritize_pos(self):
        inds = np.where(self.replay_buffer.reward > 0)
        for ind in inds:
            ind = ind[0]
            t = self.replay_buffer.time_stamp[ind]
            for _ in range(5):
                self.replay_buffer.batch_insert(self.replay_buffer.state[ind - t: ind + 1], self.replay_buffer.action[ind - t: ind + 1],
                                            self.replay_buffer.reward[ind - t: ind + 1], self.replay_buffer.next_state[ind - t: ind + 1],
                                            self.replay_buffer.done[ind - t: ind + 1])

    def reset_q(self):
        self.q_table[:] = 0

class DeepQLearning(ActiveQLearning):
    # Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2, goal_q_len=300,
                 all_subspace=False, no_range_info=False, stochastic_select_subspace=False, tree_subspace=False,
                 recip_t=50, subspace_q_size=10, replay_size=50000, level_penalty=False, priority_sample=True):
        super(DeepQLearning, self).__init__(n_states, n_actions, base, raw_dim, observation_space, gamma, alpha,
                                              all_subspace=all_subspace, tree_subspace=tree_subspace,
                                              subspace_q_size=subspace_q_size)
        #print("all_subspace", all_subspace)            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = HER_DQN().to(self.device)
        self.target_net = HER_DQN().to(self.device)
        #print(summary(self.target_net, (1, 6)))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.episode_durations = []
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.BATCH_SIZE = 300
        self.GAMMA = 0.85
        self.TARGET_UPDATE = 10

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def trainDQN(self, env, eps, agentNum):
        num_episodes = 20
        raw_obs_dim = env.observation_space.nvec.size
        agentNum = agentNum - 1
        for i_episode in tqdm(range(num_episodes)):
            #print(f"Episode {i_episode}/{num_episodes}")
            # Initialize the environment and state
            state = env.reset() # format (agent1_x, agent1_y, agent2_x, agent2_y, box_x, box_y)
            state, _ = obs_to_int_pi(state, base=env.grid_size, raw_dim=raw_obs_dim)

            for t in count():
                
                action = self.select_action(state, eps)
                if torch.is_tensor(action[0]):
                    action = [x.item() for x in action]
                #print(action)
                next_state, reward, done = env.step(action, True) # Problem. step function assumes there are multiple agents
                #print("next_state", next_state)
                reward = torch.tensor([reward], device=self.device) 

                if done:
                    next_state = None
                
                # Store the transition in memory
                if not done:
                    #print("TESTEST", state, action[agentNum], reward, next_state, done)
                    next_state, _ = obs_to_int_pi(next_state, base=env.grid_size, raw_dim=raw_obs_dim)
                    #print(next_state)
                    self.insert_data(state, action[agentNum], reward, next_state, done) # Use actions for agent 1
                else:
                    self.insert_data(state, action[agentNum], reward, 1, done)

                # Move to the next state
                #print("pre", state)
                state = next_state
                #print("post", state)


                # Perform one step of the optimization (on the policy network)
                self.optimize_model(agentNum)
                if done:
                    #print("HTCFUYCUYCUYCUYCUYCUYCUYC")
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

                # Update the target network, copying all weights and biases in DQN
                if t % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        eps = eps - (eps * (eps/num_episodes)) # Decay epsilon
        #env.render()
        #env.close()
        plt.ioff()
        plt.show()

    def optimize_model(self, agent_id):
        # if self.replay_buffer.max_size < self.BATCH_SIZE:
        #     return



        # transitions = memory.sample(BATCH_SIZE)
        # full_buffer = [s for s in self.replay_buffer.get_all_data()]
        # if len(full_buffer) < self.BATCH_SIZE:
        #     batch = full_buffer
        # else:
        batch = self.replay_buffer.get_random_batch(self.BATCH_SIZE)
        #batch = [random.randint(0, self.BATCH_SIZE)]#self.replay_buffer.get_random_batch(self.BATCH_SIZE) # *zip(*transitions) # Need to format to replay_buffer format

        outputs = []
        targets = []
        for _ in range(self.BATCH_SIZE):
            transitionTuple = next(batch)
            s = torch.from_numpy(np.array([transitionTuple[0]]).astype(np.float32))
            r = torch.from_numpy(np.array([transitionTuple[2]]).astype(np.float32))
            output = self.policy_net(s)[agent_id + 2]
            target = self.target_net(s)[2]#[agent_id] #torch.from_numpy(np.array(self.target_net(s)))
            # print("output", output)
            # print("target", target)
            target = (target * self.GAMMA) + r
            #print("target", target)
            outputs.append(output)
            targets.append(target)

            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
        return loss
            #print(f"loss: {loss:>7f}")


        # print(outputs)
        # print("_____")
        # print(targets)

        # #print(outputs, targets)
        # outputs = torch.stack(outputs)
        # targets = torch.stack(targets)
        # #print("outputs, targets", outputs, targets)
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(outputs, targets)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # loss = loss.item()
        # print(f"loss: {loss:>7f}")


        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                     batch)), device=self.device, dtype=torch.bool)

        # list_non_final_next_states = []
        # list_state_batch = []
        # list_action_batch = []
        # list_reward_batch = []
        # for s, a, r, ns, done in batch:
        #     if ns is not None:
        #         print(s, a, r, ns, done)
        #         list_non_final_next_states.append(torch.LongTensor(ns))
        #         list_state_batch.append(torch.LongTensor(s))
        #         list_action_batch.append(torch.LongTensor(a))
        #         list_reward_batch.append(torch.LongTensor(int(r)))

        # non_final_next_states = torch.cat(list_non_final_next_states) # index 3 of tuple is next state
        # state_batch = torch.cat(list_state_batch)
        # action_batch = torch.cat(list_action_batch)
        # reward_batch = torch.cat(list_reward_batch)
        # # for s in state_batch:
        # #     print(torch.from_numpy(np.array([s.unsqueeze(dim=0).item()]).astype(np.float32)))

        # # state_action_values = [self.policy_net(torch.from_numpy(np.array([s.unsqueeze(dim=0).item()]).astype(np.float32))).gather(1, action_batch) for s in state_batch]
        # state_action_values = [self.policy_net(torch.from_numpy(np.array([s.unsqueeze(dim=0).item()]).astype(np.float32))) for s in state_batch]

        # # print("state_batch", len(state_batch))
        # # state_action_values = [self.policy_net(s.unsqueeze(dim=0)) for s in state_batch]

        # print("TEST0")
        # next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # # Compute the expected Q values
        # expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # # Optimize the model
        # print("TEST1")
        # self.optimizer.zero_grad()
        # print("TEST2")
        # loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()


    def select_action(self, s, eps):
        global steps_done

        if np.random.rand() < eps:
            #return np.random.choice(self.n_actions)
            return random.sample([0, 1, 2, 3], 2)
            #return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                s = np.array([s])
                s = torch.from_numpy(s)
                # s = s.to(torch.long)
                # s = s.type(torch.LongTensor)
                s = s.float()
                actions = []
                for actionLogits in self.policy_net(s):
                    # print("actionLogits", actionLogits)
                    # print("torch.argmax(actionLogits)", torch.argmax(actionLogits))
                    # print("actionLogits.unsqueeze(1).max(1)", actionLogits.unsqueeze(1).max(1))
                    # print("actionLogits.unsqueeze(1).max(1)[0]", actionLogits.unsqueeze(1).max(1)[0])
                    #actions.append(actionLogits.unsqueeze(1).max(1)[1].view(1, 1))
                    actions.append(torch.argmax(actionLogits))
                return actions

            # break tie uniformly
            # if other_q_table is None:
            #     return np.random.choice(np.flatnonzero(self.q_table[s] == self.q_table[s].max()))
            # else:
            #     q_table = (1 - alpha) * self.q_table[s] + alpha * other_q_table[s]
            #     return np.random.choice(np.flatnonzero(q_table == q_table.max()))
    
    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
    
    # def update_count():
    #     raise NotImplementedError
        
    # def update_q():
    #     raise NotImplementedError
    
    # def update_q_from_D():
    #     raise NotImplementedError










