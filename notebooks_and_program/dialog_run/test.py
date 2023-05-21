from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json
from user import User
from utils import remove_empty_slots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_TEST = run_dict['num_ep_run']
    MAX_ROUND_NUM = run_dict['max_round_num']

    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    remove_empty_slots(database)

    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def test_run():

    print('Тестирование началось...')
    episode = 0
    while episode < NUM_EP_TEST:
        episode_reset()
        episode += 1
        ep_reward = 0
        done = False
        state = state_tracker.get_state()
        while not done:
            agent_action_index, agent_action = dqn_agent.get_action(state)
            state_tracker.update_state_agent(agent_action)
            user_action, reward, done, success = user.step(agent_action)
            ep_reward += reward
            if not done:
                emc.infuse_error(user_action)
            state_tracker.update_state_user(user_action)
            state = state_tracker.get_state(done)
        print('Эпизод: {} Успех: {} Награда: {}'.format(episode, success, ep_reward))
    print('...Тестирование завершено')


def episode_reset():

    state_tracker.reset()
    user_action = user.reset()
    emc.infuse_error(user_action)
    state_tracker.update_state_user(user_action)
    dqn_agent.reset()


test_run()
