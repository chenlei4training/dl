import pandas as pd
import numpy as np
import time

np.random.seed(1)

n_state = 10
actions = ["left", "right"]


q_table = pd.DataFrame(np.zeros(shape=(n_state, len(actions))), columns=actions)


def get_env_feedback(current_state, a):
    reward = 0
    is_teminate = False
    if current_state == (n_state - 1):
        is_teminate = True
        reward = 1

    return reward, is_teminate


def take_actoion(state):
    complete_random=False #if true means never use the greedy appoche

    changce_random_jump = 0.1

    if complete_random or ((q_table.loc[state]==0).all()):
        return np.random.choice(actions)
    else:
        if np.random.uniform() <= changce_random_jump:
            return np.random.choice(actions)
        else:
            # print('q_table.loc[state].idxmax()',q_table.loc[state].idxmax())
            return q_table.loc[state].idxmax() #greedy appoche

def q_learning():
    episode = 5
   
    for epi in range(episode):
        current_state = 0
        current_action = take_actoion(current_state)
        n_jump = 0
        _, is_terminate = get_env_feedback(current_state, current_action)
        while not is_terminate:
            # Loop the core algorithm to update the q_table
            show(current_state,n_jump)
            next_state = -100
            if current_state == 0 and current_action == "left":
                next_state = 0
            elif current_action == "left":
                next_state = current_state - 1

            if current_action == "right":
                next_state = current_state + 1

            next_action = take_actoion(next_state)
            n_jump += 1
            next_state_reward, is_terminate = get_env_feedback(next_state, next_action)

            if is_terminate:
                q_table.loc[current_state, current_action] += 1
            else:
                q_table.loc[current_state, current_action] += (
                    0.1 * (0.5 * (next_state_reward + q_table.loc[next_state].max() - q_table.loc[current_state, current_action]))
                )
            current_state = next_state
            current_action = next_action
        show(current_state,n_jump)
        print("\nfinish epi:", epi + 1,'  n_jump:',n_jump)
        # print('last action',current_action)



def show(state,jump):
    str_list = (n_state - 1) * ["-"] + ["T"]
    str_list[state] = "o"
    print("\r", "%s --- %d"%(''.join(str_list),jump), end="")  # \r 也就是打印头归位，回到某一行的开头。
    pause = 0.5

    time.sleep(pause)

def testP():
    is_teminate = False

    while not is_teminate:
        str = ""
        for _ in range(n_state):
            str += np.random.choice(actions)

        print(str)
        if str == "rightrightrightright":
            is_teminate = True


if __name__ == "__main__":
    # testP()
    q_learning()
    print(q_table)

