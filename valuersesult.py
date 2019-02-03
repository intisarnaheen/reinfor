import numpy as np

def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.

    @param v the value vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)


    #Change as you want
tot_states = 12
gamma = 0.999 #Discount factor
iteration = 0 #Iteration counter
epsilon = 0.01 #Stopping criteria small value
cum_reward = 0
policy_reward = []

    #List containing the data for each iteation
graph_list = list()

    #Transition matrix loaded from file (It is too big to write here)
T = np.load("T.npy")

    #Reward vector
r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    #Utility vectors
u = np.array([0.0, 0.0, 0.0,  0.0,
              0.0, 0.0, 0.0,  0.0,
              0.0, 0.0, 0.0,  0.0])
u1 = np.array([0.0, 0.0, 0.0,  0.0,
               0.0, 0.0, 0.0,  0.0,
               0.0, 0.0, 0.0,  0.0])
c_count=0

for c_count in range (30):
    u = u1.copy()
    iteration += 1
    graph_list.append(u)
    for s in range(tot_states):
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            cum_reward = cum_reward + reward
            policy_reward.append(cum_reward)
            v = np.zeros((1, tot_states))
            v[0, s] = 1.0
            u1[s] = return_state_utility(v, T, u, reward, gamma)

            reward = r[s]
            cum_reward = cum_reward + reward
            policy_reward.append(cum_reward)
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            #delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria

print("total final reward", policy_reward)
