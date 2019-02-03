import numpy as np
import random
reward_count = 0
moved_right_count = 0 #the number of time the state moves to the right
moved_left_count = 0 #the number of time the state moves to the start
stayed_count = 0 #the number of times the state stays the same
p1 = [50, 50, 50, 50, 50] #initial probability of moving right
p2 = 80 #probability of moving right given p1
p3 = 20
state = 1
count = 1000 #number of total iterations
ipochs = 200
reward = 0
s=[]
a=[]
s_new=[]
for i in range(count):
    r1 = random.randint(1,100)
    r2 = random.randint(1,100)
    #print((r1,r2))
    #print(a)
    if r1 >= p1[state-1]: #if the random number is <= our first probability, we check for the second one
        #we again pick a random number for the probability p2
        #print("T1 Right action")
        action = 1
        a.append(action)
        if (r2 > p3 and state < 5):
            s.append(state)
             #if it checks and we are not at the last state, we move right
            #s[i]=state
            #print(s)
            state = state + 1
            s_new.append(state)
            #action =1
            #a.append(action)

            #print ("Moved right, and the current state is: ", state, "\tIteration number: ", i+1)
            moved_right_count = moved_right_count + 1
            #s_new[i]=state
            #print(s)
            reward = 0
            #a[i]=1
            reward_count = reward_count+reward
            #print ("Reward in this state is :", reward , "\tIteration number:", i+1)
        elif (r2 > p3 and state ==5):
            #print ("Reached the final state ", state, "\tIteration number: ", i+1)
            stayed_count = stayed_count + 1
            #print(state)
            s.append(state)
            s_new.append(state)

            reward = +10
            reward_count = reward_count+reward
            #print ("Reward in the final state is :", reward , "\tIteration number:", i+1)
        else:
            #print("Left action")
            s.append(state)
            state = 1
            s_new.append(state)

           # action =2
            #a.append(action)
            #s[i]= state
            #s_new[i] =state +1
            #a[i]= 2
            #print ("Went back to initial state, iteration number: ", i+1)
            moved_left_count = moved_left_count + 1
            reward = +2
            reward_count = reward_count+reward
    else:
        #print("T1 Left action")
        action = 2
        a.append(action)
        #print ("Went back to initial state, so Reward in this state is :", reward , "\tIteration number:", i+1)
        if (r2 <= p3 and state < 5): #if it checks and we are not at the last state, we move right
            #s[i]=state
            s.append(state)
            state = state + 1
            s_new.append(state)

           # action =1
            #a.append(action)

            #s_new[i]=state
            #print ("Moved right, and the current state is: ", state, "\tIteration number: ", i+1)
            moved_right_count = moved_right_count + 1
            #a[i]=1
            reward = 0
            reward_count = reward_count+reward
            #print ("Reward in this state is :", reward , "\tIteration number:", i+1)
        elif (r2 <= p3 and state ==5):
            #print ("Reached the final state ", state, "\tIteration number: ", i+1)
            #s[i] = 5
            #s_new[i]= 5
            #a[i]=1
            s.append(state)
            s_new.append(state)

           # action =1
            #a.append(action)
            stayed_count = stayed_count + 1
            reward = +10
            reward_count = reward_count+reward
            #print ("Reward in the final state is :", reward , "\tIteration number:", i+1)
        else:
            #print("Left action")
            s.append(state)
            state = 1
            #s[i]= state
            #s_new[i] =state +1
            #a[i]= 2
            #print ("Went back to initial state, iteration number: ", i+1)
            s_new.append(state)

            moved_left_count = moved_left_count + 1
            reward = +2
            reward_count = reward_count+reward
            #print ("Went back to initial state, so Reward in this state is :", reward , "\tIteration number:", i+1)
        #state = 1
        #print ("Went back to initial state, iteration number: ", i+1)
        #stayed_count = stayed_count + 1
print("Total reward:", reward_count )
#print ("Stayed in the same state: ", stayed_count, " times")

#print ("Moved right: ", moved_right_count, " times")
#print ("Moved left: ", moved_left_count, " times")

tuple = []
for i in range(count):
    tuple.append((s[i],s_new[i], a[i]))

matrix = np.ones((5,5,2))

for currState in range(5):
    for nextState in range(5):
        for currAction in range(2):
            nume =  len([item for item in tuple if item == (currState+1,nextState+1,currAction+1)])
            dino =  len([item for item in tuple if (item[0],item[2]) == (currState+1,currAction+1)])
            matrix[currState,nextState,currAction] =  (0 if dino == 0 else nume/dino)

#print(matrix)
def return_state_utility(v, t, u, reward, gamma):

    action_array = np.zeros(2)
    for action in range(0, 2):
        #prob * (reward + discount_factor * V[next_state])
        action_array = np.sum(np.multiply(np.dot(v, t[:,:,action]),reward)+ gamma * np.multiply(u, np.dot(v, t[:,:,action])))
        #action_array[action] = np.sum(np.multiply(np.dot(v, T[:,:,action]),reward)+ gamma * np.multiply(u, np.dot(v, T[:,:,action])))
        #action_array = np.sum(reward+ gamma * np.multiply(u, np.dot(v, T[:,:,action])))
        #action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        #a_s= action_array.shape
        #print(a_s)
        #a_dim = action_array.ndim
        #print(a_dim)
    return np.max(action_array)
    #return reward + gamma * np.max(action_array)
def main():
    #Change as you want
    tot_states = 5
    gamma = 0.8 #Discount factor
    iteration = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value

    #List containing the data for each iteation
    graph_list = list()

    #Transition matrix loaded from file (It is too big to write here)
    #transitionMatrix1 = np.array([[0.2,0.8,0,0,0],[0.2,0,0.8,0,0],[0.2,0,0,0.8,0],[0.2,0,0,0,0.8],[0.2,0,0,0,0.8]])
    #t1 = transitionMatrix1.transpose()
    #print()
    #transitionMatrix2 = np.array([[0.8,0.2,0,0,0],[0.8,0,0.2,0,0],[0.8,0,0,0.2,0],[0.8,0,0,0,0.2],[0.8,0,0,0,0.2]])
    #t2 = transitionMatrix2.transpose()
    #d = np.concatenate((t1, t2), axis =1)
    #d  = np.concatenate((transitionMatrix2, transitionMatrix1),axis=1)
    #b= np.dstack((transitionMatrix2,transitionMatrix1))
    #print(b)
    t=matrix
    print(t)

    #xxx= b[:,:,1]
    #print("transitio for action",xxx)




    #t_s= T.shape
    #t_dim = T.ndim
    #print(t_s)
    #print(t_dim)
    #np.
    #concatenate((a, b), axis=0)
    #print(T)
    #Reward vector

    r = np.array([2, 0, 0, 0, 10])

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,0.0,0.0])
    u1 = np.array([0.0, 0.0, 0.0,0.0,0.0])

    while True:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u1[s] = return_state_utility(v, t, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria
            #new_action= return_state_utility(v, t, u, reward, gamma)

        if delta < epsilon * (1 - gamma) / gamma:
                print("=================== FINAL RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                print(u[0:5])
                print("===================================================")
                break

if __name__ == "__main__":
    main()

