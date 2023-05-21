usersim_intents = ['inform', 'request', 'thanks', 'reject', 'done']

usersim_default_key = 'бронь'

usersim_required_init_inform_keys = ['кухня']

agent_inform_slots = ['город', 'район', 'область', 'кухня', 'название', 'яндекс_карты', 'гугл_карты', 'дата', 'время', usersim_default_key]
                      
agent_request_slots = ['город', 'район', 'область', 'кухня', 'название', 'яндекс_карты', 'гугл_карты', 'дата', 'время',
                      'количество_детей', 'количество_человек']
                      
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]

for slot in agent_inform_slots:
    if slot == usersim_default_key:
        continue
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})

for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

rule_requests = ['название', 'время', 'дата', 'город', 'район', 'количество_человек']

no_query_keys = ['количество_детей', usersim_default_key]

FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

all_intents = ['inform', 'request', 'done', 'match_found', 'thanks', 'reject']

all_slots = ['город', 'район', 'область', 'кухня', 'название', 'яндекс_карты', 'гугл_карты', 'дата', 'время',
            'количество_детей', 'количество_человек', usersim_default_key]