from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions
import re

class DQNAgent:

    def __init__(self, state_size, constants):

        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)

        self.rule_request_set = rule_requests

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()

        self._load_weights()

        self.reset()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def reset(self):

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = self._map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self._rule_action()
            else:
                return self._dqn_action(state)

    def _rule_action(self):

        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)
        return index, rule_response

    def _map_action_to_index(self, response):

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def _dqn_action(self, state):

        index = np.argmax(self._dqn_predict_one(state))
        action = self._map_index_to_action(index)
        return index, action

    def _map_index_to_action(self, index):

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def _dqn_predict_one(self, state, target=False):

        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):

        if target:
            return self.tar_model.predict(states, verbose=0)
        else:
            return self.beh_model.predict(states, verbose=0)

    def add_experience(self, state, action, reward, next_state, done):

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):

        return len(self.memory) == self.max_memory_size

    def train(self):

        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(self.memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict(states)
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict(next_states)
            tar_next_state_preds = self._dqn_predict(next_states, target=True)

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = beh_state_preds[i]
                if not self.vanilla:
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def copy(self):

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
