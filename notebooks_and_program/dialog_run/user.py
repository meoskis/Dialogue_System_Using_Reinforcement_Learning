from dialogue_config import FAIL, SUCCESS, usersim_intents, all_slots
from utils import reward_function


class User:
    def __init__(self, constants):
        self.max_round = constants['run']['max_round_num']

    def reset(self):

        return self._return_response()

    def _return_response(self):

        response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}
        while True:
            input_string = input('Response: ')
            chunks = input_string.split('/')

            intent_correct = True
            if chunks[0] not in usersim_intents:
                intent_correct = False
            response['intent'] = chunks[0]

            informs_correct = True
            if len(chunks[1]) > 0:
                informs_items_list = chunks[1].split(', ')
                for inf in informs_items_list:
                    inf = inf.split(': ')
                    if inf[0] not in all_slots:
                        informs_correct = False
                        break
                    response['inform_slots'][inf[0]] = inf[1]

            requests_correct = True
            if len(chunks[2]) > 0:
                requests_key_list = chunks[2].split(', ')
                for req in requests_key_list:
                    if req not in all_slots:
                        requests_correct = False
                        break
                    response['request_slots'][req] = 'UNK'

            if intent_correct and informs_correct and requests_correct:
                break

        return response

    def _return_success(self):

        success = -2
        while success not in (-1, 0, 1):
            success = int(input('Success?: '))
        return success

    def step(self, agent_action):

        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'

        print('Agent Action: {}'.format(agent_action))

        done = False
        user_response = {'intent': '', 'request_slots': {}, 'inform_slots': {}}

        if agent_action['round'] == self.max_round:
            success = FAIL
            user_response['intent'] = 'done'
        else:
            user_response = self._return_response()
            success = self._return_success()

        if success == FAIL or success == SUCCESS:
            done = True

        assert 'UNK' not in user_response['inform_slots'].values()
        assert 'PLACEHOLDER' not in user_response['request_slots'].values()

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success is 1 else False
