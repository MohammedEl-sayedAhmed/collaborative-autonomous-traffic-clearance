#!/usr/bin/env python
import rospy
import numpy as np
import random
import os
from std_msgs.msg import Header
from racecar_clear_ev_route.srv import RLPolicyActionService, RLPolicyActionServiceRequest, RLPolicyActionServiceResponse
from racecar_rl_environments.srv import states, statesRequest, statesResponse
        
class SingleAgentQlearning:
    def __init__(self, environment_init=dict(), algo_params=dict(), load_q_table = False, test_mode_on = False):

        rospy.loginfo("Initializing Single Agent Qlearning.")

        self.test_mode_on = test_mode_on
        
        #Constants for the environment
        self.environment_init = environment_init
        
        self.rel_amb_y_min = self.environment_init['rel_amb_y_min']
        self.rel_amb_y_max = self.environment_init['rel_amb_y_max']
        self.amb_vel_min = self.environment_init['amb_vel_min']
        self.amb_vel_max = self.environment_init['amb_vel_max']
        self.agent_vel_min = self.environment_init['agent_vel_min']
        self.agent_vel_max = self.environment_init['agent_vel_max']
        self.amb_acc = self.environment_init['amb_acc']
        self.agent_acc = self.environment_init['agent_acc']

        self.amb_lane_count = 3 #default
        self.agent_lane_count = 3 #default
        self.action_space = 5 #default

        #RL Actions
        self.Actions = ["change_left", "change_right", "acc", "no_acc", "dec"]
        self.action_string_to_index_dict = {
            "change_left": 0,
            "change_right": 1,
            "acc": 2,
            "no_acc": 3,
            "dec": 4
        }  # Must maintain order in Actions
        self.actions_indices = []   
        for act in self.Actions:
            self.actions_indices.append(self.action_string_to_index_dict[act]) # 0 1 2 3 4
        
        self.action_chosing_method = None  # To be asssigned: Exploration or Exploitation based on exp_exp_tradeoff and epsilon

        #Q table Initialization 
        if(load_q_table or self.test_mode_on): # load_q_table if we are testing or we want to load it
            self.q_table = self.load_q_table()
        else:
            self.init_q_table(-1000)
            
        #Setting Algorithm parameters 
        self.exp_exp_tradeoff = algo_params['exp_exp_tradeoff'] #TODO: Ask why
        self.epsilon = algo_params['epsilon']
        self.gamma = algo_params['gamma']
        self.learning_rate = algo_params['learning_rate']
        self.max_epsilon = algo_params['max_epsilon']
        self.min_epsilon = algo_params['min_epsilon']
        self.decay_rate = algo_params['decay_rate']

        # Fix toggle (default False = shipped behaviour): when True the agent may actually
        # perform lane changes instead of the change_left->dec / change_right->no_acc remap
        # below. Lets a real training campaign compare baseline vs. fixed via a ROS param.
        self.enable_lane_changes = rospy.get_param('enable_lane_changes', False)
        rospy.loginfo("enable_lane_changes = %s", self.enable_lane_changes)
        
        #new and last observed states parameters 
        self.new_observed_state_for_this_agent = statesResponse()
        self.last_observed_state_for_this_agent = statesResponse()

        #discretized new and last obesrved states (INDICES) parameters
        self.new_observed_state_INDEX_for_this_agent = statesResponse()
        self.last_observed_state_INDEX_for_this_agent = statesResponse()
        
        #RL engagement and disengagement
        self.RLdisengage = False  

        rospy.loginfo("Finished initializing Single Agent Qlearning.")
    
    #Define a uniformly-spaced grid that can be used to discretize a space, works for a single dimension
    def create_uniform_grid(self, low, high, bins, step):
        """
        Parameters
        ----------
        low: Lower bound of the continuous space.
        high: Upper bound of the continuous space.
        bin: Number of bins (segments)
    
        Returns
        -------
        grid:a list of arrays containing split points for each dimension.
        """
        grid = []
        grid.append(np.linspace(low + step, high ,endpoint=False, num=bins-1))
    
        return grid
    
    #Discretize a sample as per given grid, works for higher dimensions
    def discretize(self,sample, grid):
        """
        Parameters
        ----------
        sample : It has to be an array, A single sample from the (original) continuous space --> eg. velocity = 0.23
        grid: An array containing split points for each dimension.
    
        Returns
        -------
        discretized_sample: A sequence of integers with the same number of dimensions as sample.
        """
        dim = len(np.transpose(sample))
        discrete = np.zeros((dim,), dtype=int)
        for j in range(dim):
            discrete[j] = int(np.digitize(sample[j], grid[j]))
            
        return discrete
    
    #updates the new observed state parameters  
    def update_new_observed_state_for_this_agent(self, new_observed_state_for_this_agent):
        #first, update last observed state for this agent
        self.last_observed_state_for_this_agent = self.new_observed_state_for_this_agent
        self.last_observed_state_INDEX_for_this_agent = self.new_observed_state_INDEX_for_this_agent

        self.new_observed_state_for_this_agent = new_observed_state_for_this_agent
        rospy.loginfo("New Observed State is %f,%d,%f,%d,%f",self.new_observed_state_for_this_agent.agent_vel, self.new_observed_state_for_this_agent.agent_lane, self.new_observed_state_for_this_agent.amb_vel,self.new_observed_state_for_this_agent.amb_lane,self.new_observed_state_for_this_agent.rel_amb_y)

        #Discretization
        self.new_observed_state_INDEX_for_this_agent.agent_vel = int(self.discretize(np.array([self.new_observed_state_for_this_agent.agent_vel]), self.agent_vel_state_grid))  # [0,1,2,3,4,5] 
        self.new_observed_state_INDEX_for_this_agent.agent_lane = int(self.new_observed_state_for_this_agent.agent_lane)
        self.new_observed_state_INDEX_for_this_agent.amb_vel =int(self.discretize(np.array([self.new_observed_state_for_this_agent.amb_vel]), self.amb_vel_state_grid))  # [0,1,2,3,4,5,6,7,8,9,10] 
        self.new_observed_state_INDEX_for_this_agent.amb_lane = int(self.new_observed_state_for_this_agent.amb_lane)
        self.new_observed_state_INDEX_for_this_agent.rel_amb_y = int(np.round(self.new_observed_state_for_this_agent.rel_amb_y) + abs(self.rel_amb_y_min))

        rospy.loginfo("New Observed State INDEX is %f,%d,%f,%d,%f",self.new_observed_state_INDEX_for_this_agent.agent_vel, self.new_observed_state_INDEX_for_this_agent.agent_lane, self.new_observed_state_INDEX_for_this_agent.amb_vel,self.new_observed_state_INDEX_for_this_agent.amb_lane,self.new_observed_state_INDEX_for_this_agent.rel_amb_y)

        if ((self.new_observed_state_INDEX_for_this_agent.agent_vel >= self.agent_vel_dimension) or (self.new_observed_state_INDEX_for_this_agent.agent_lane >= self.agent_lane_dimension) or (self.new_observed_state_INDEX_for_this_agent.amb_vel >= self.amb_vel_dimension) or (self.new_observed_state_INDEX_for_this_agent.amb_lane>= self.amb_lane_dimension) or (self.new_observed_state_INDEX_for_this_agent.rel_amb_y >= self.rel_amb_y_dimension) or (self.new_observed_state_INDEX_for_this_agent.rel_amb_y < 0)):
            rospy.loginfo("\n\n\n\nSTATE IS OUTSIDE Q TABLE!!!!!!\n\n\n\n\n")
            return False
        else:
            return True
    
    #picks an action by exploitation or exploration 
    def pick_action(self, feasible_actions_indices):
        
        if (self.action_chosing_method == 'expLOIT'):  # test_mode_on will force the algorithm to choose exploitation.
            
            max_value_index = np.argmax(self.q_table[self.new_observed_state_INDEX_for_this_agent.agent_vel, 
                    self.new_observed_state_INDEX_for_this_agent.agent_lane,
                    self.new_observed_state_INDEX_for_this_agent.amb_vel, 
                    self.new_observed_state_INDEX_for_this_agent.amb_lane, 
                    self.new_observed_state_INDEX_for_this_agent.rel_amb_y, feasible_actions_indices]) 
            rospy.loginfo("maximum value index in exploitation is %d\n", max_value_index)  

            desired_action_index = feasible_actions_indices[max_value_index]

            rospy.loginfo("desired action index in exploitation is %d\n", desired_action_index) 
            desired_action_string = self.Actions[desired_action_index]
            rospy.loginfo("desired action string in exploitation is %s\n", desired_action_string) 
            self.Action = desired_action_string
        
        elif(self.action_chosing_method == 'expLORE'):
            
            action_index = random.choice(feasible_actions_indices)  #action's number, not index in the array 
            rospy.loginfo(" action in exploration is %d\n", action_index)
            desired_action_string = self.Actions[action_index]
            rospy.loginfo(" desired_action_string in exploration is %s\n", desired_action_string)
            self.Action = desired_action_string
    
    #action execution, return its feasibility and taken time if feasible
    #also, it deactivates RL
    def execute_action(self):
        action_feasibility = False
        action_taken_time = 0

        # Lane changes were remapped away "for testing purposes only", which disabled the
        # agent's core move-aside maneuver. Keep that as the default, but allow enabling real
        # lane changes via the enable_lane_changes param (see __init__).
        if not self.enable_lane_changes:
            if self.Action == 'change_left':
                self.Action = 'dec'
            if self.Action == 'change_right':
                self.Action = 'no_acc'

        #update request parameters 
        if (self.Action == 'change_left'):
            action_request_control_action = 1
            action_request_acc = 0
        elif(self.Action == 'change_right'):
            action_request_control_action = 2
            action_request_acc = 0
        elif(self.Action == 'acc'):
            action_request_control_action = 0
            action_request_acc = self.agent_acc
        elif(self.Action == 'no_acc'):
            action_request_control_action = 0
            action_request_acc = 0
        elif(self.Action == 'dec'):
            action_request_control_action = 0
            action_request_acc = -self.agent_acc
        
        action_request_header = Header()
        action_request_header.stamp = rospy.Time.now()
        deactivateRL = False
        
        try:
            rospy.wait_for_service('move_car/RL/RLPolicyActionService')
            rospy.loginfo("Sending a Request to the move car action client") 
            sendAction = rospy.ServiceProxy('move_car/RL/RLPolicyActionService',RLPolicyActionService)
            resp = sendAction(action_request_header, action_request_control_action, action_request_acc, deactivateRL)
            action_feasibility = resp.RLActionresult
            action_taken_time = resp.RLActionTime
            rospy.loginfo("response is %d", action_feasibility)
    
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            rospy.wait_for_service('move_car/RL/RLPolicyActionService')
            rospy.loginfo("move_car/RL/RLPolicyActionService is re-established, however, action is not executed")
                
        
        return action_feasibility, action_taken_time

    
    
    #final method to be used by RL master client, it takes the new observed state and returns the
    #action with the time taken for the it to be executed 
    def take_action(self, new_observed_state_for_this_agent):
        rospy.loginfo("\n\n\n\n\n\n") 
        rospy.loginfo("///////// Taking New Action /////////") 
        
                
        #update the new observed state for the action
        toTakeAction = self.update_new_observed_state_for_this_agent(new_observed_state_for_this_agent)
        if toTakeAction == False:
            return "NOACTION", 1
        else:
            #exploration or exploitation choice 
            self.exp_exp_tradeoff = random.uniform(0,1) #TODO
            
            if (self.exp_exp_tradeoff > self.epsilon or self.test_mode_on):
                self.action_chosing_method = 'expLOIT'
                rospy.loginfo("Action choosing method: Exploit\n") #TODO:Comment
            else:
                self.action_chosing_method = 'expLORE'
                rospy.loginfo("Action choosing method: Explore\n") #TODO:Comment

            #initially all actions are feasible #TODO:old
            self.actions_indices = []  
            for act in self.Actions:
                self.actions_indices.append(self.action_string_to_index_dict[act]) # 0 1 2 3 4
            
            feasible_actions_indices = self.actions_indices
            found_feasible_action = False

            iterations = 0  #TODO:new
            #keep trying to pick an action until it's feasible
            #remove the infeasible actions from feasible_actions_indices
            while(not found_feasible_action):
                iterations = iterations + 1
                rospy.loginfo("Trying To find an action, iteration is %d",iterations)

                self.pick_action(feasible_actions_indices)
                rospy.loginfo("Picked action is %s\n", self.Action) #TODO:Comment
                action_feasibility, action_taken_time = self.execute_action()
                rospy.loginfo("Action Feasibility is %d, and time taken is %f\n", action_feasibility, action_taken_time) #TODO:Comment
                
                if (action_feasibility == True and action_taken_time > 0):  #If action was feasible and executed 
                    found_feasible_action = True
                else:
                    found_feasible_action = False
                    feasible_actions_indices.remove(self.action_string_to_index_dict[self.Action])

            return self.Action, action_taken_time  #TODO: return a flag to indicate that there is no feasible action, if happens

    

    #sends a flag to move car action client to disengage RL and activate navigation
    def disengage(self):
        self.RLdisengage = True
        action_request_header = Header()
        action_request_header.stamp = rospy.Time.now()

        try:
            rospy.wait_for_service('move_car/RL/RLPolicyActionService')
            sendAction = rospy.ServiceProxy('move_car/RL/RLPolicyActionService',RLPolicyActionService)
            resp = sendAction(action_request_header, -1, -1, self.RLdisengage)
    
        except rospy.ServiceException, e:
                rospy.wait_for_service('move_car/RL/RLPolicyActionService')
                print "Service call failed: %s"%e
                
    def init_q_table(self,initial_value):
        '''
        #Q_table. Multi-dimensional np.ndarray, each dimension: either state partial assignment or action (string action -> integer)
        transformation is defined via action_to_string_dict
        
        #Access order for the Q_table is [agent_vel][agent_lane][amb_vel][amb_lane][rel_amb_y][action]
        
        #Values are kept as integers by rounding and casting as int. Values are clipped using np.clip() function
        # agent_vel  e.g. (6): [0,1,2,3,4,5] #Clipped before applying velocity
        # agent_lane e.g. (3): [0,1,2]
        # amb_vel    e.g. (11): [0,1,2,3,4,5,6,7,8,9,10]
        # amb_lane   e.g. (3):  [0,1,2]
        # rel_amb_y  e.g. (16+1+41 = 58) [-41,-40,-39,.....,0,...13,14,15,16]
        #since window is designed to be:
            #<--4 time steps *10 cells/step  = 40 steps  behind -- . agent . -- 3 steps * 5 cells/sec -->
        # action     e.g. (5) : [Change left, Change right, acc +, acc -, acc 0]
        Q_table size, is therefore = 6 * 3 * 11 *3 * 58 * 5 = 172260 ~ 170K .
        Note that some Q(s,a) pairs will be infeasible and hence will not be trained/updated.
        '''
        #Number of segments (bins) for agent velocity discretization
        agent_vel_bins = int(np.ceil(((self.agent_vel_max - self.agent_vel_min)/self.agent_acc)))
        self.agent_vel_state_grid = self.create_uniform_grid(self.agent_vel_min, self.agent_vel_max, agent_vel_bins ,self.agent_acc)
        
        #Number of segments (bins) for ambulance velocity discretization
        amb_vel_bins = int(np.ceil(((self.amb_vel_max - self.amb_vel_min)/self.amb_acc)))
        self.amb_vel_state_grid = self.create_uniform_grid(self.amb_vel_min, self.amb_vel_max, amb_vel_bins, self.amb_acc)
        
        self.agent_vel_dimension = agent_vel_bins
        self.agent_lane_dimension = self.agent_lane_count
        self.amb_vel_dimension = amb_vel_bins
        self.amb_lane_dimension = self.amb_lane_count
        self.rel_amb_y_dimension = abs(self.rel_amb_y_max) + 1 + abs(self.rel_amb_y_min)
        self.actions_dimension = self.action_space
        rospy.loginfo("\n\n\n\n\n\n\n\n\n")
        rospy.loginfo("init_table_dimensions")
        rospy.loginfo("\nagent_vel_dimension is %d, agent_lane_dimension is %d, amb_vel_dimension is %d, amb_lane_dimension is %d, rel_amb_y_dimension is %d, actions_dimension is %d", self.agent_vel_dimension, self.agent_lane_dimension, self.amb_vel_dimension, self.amb_lane_dimension, self.rel_amb_y_dimension, self.actions_dimension)
        self.q_table = np.zeros((self.agent_vel_dimension, self.agent_lane_dimension, self.amb_vel_dimension, self.amb_lane_dimension, self.rel_amb_y_dimension, self.actions_dimension))
        
        #Initialize all the Q table with -1000 as a flag for unvisited action
        self.q_table.fill(initial_value)
    
    def display_q_table(self):
        rospy.loginfo("\n\n\nQ table is\n")
        rospy.loginfo(self.q_table)
        rospy.loginfo("\n\n\n")
        
    def update_q_table(self, reward, new_observed_state_for_this_agent):
        
        if(self.test_mode_on):  # do not update q_table if test_mode is on ! just use it.
            pass
        else:
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            '''
            s: old state for which a was decided
            a: action taken in state s
            s': new state we are in because of action a (previous action)
            a': new action we are expected to choose if we exploit on s' (new state)
            '''
            action_index = self.action_string_to_index_dict[self.Action]
            self.update_new_observed_state_for_this_agent(new_observed_state_for_this_agent)

            # OLD STATE VARIABLES:
            agent_vel_index = self.last_observed_state_INDEX_for_this_agent.agent_vel
            agent_lane_index = self.last_observed_state_INDEX_for_this_agent.agent_lane
            amb_vel_index = self.last_observed_state_INDEX_for_this_agent.amb_vel
            amb_lane_index = self.last_observed_state_INDEX_for_this_agent.amb_lane
            rel_amb_y_index = self.last_observed_state_INDEX_for_this_agent.rel_amb_y

            # NEW STATE VARIABLES:
            new_agent_vel_index = self.new_observed_state_INDEX_for_this_agent.agent_vel
            new_agent_lane_index = self.new_observed_state_INDEX_for_this_agent.agent_lane
            new_amb_vel_index = self.new_observed_state_INDEX_for_this_agent.amb_vel
            new_amb_lane_index = self.new_observed_state_INDEX_for_this_agent.amb_lane
            new_rel_amb_y_index = self.new_observed_state_INDEX_for_this_agent.rel_amb_y
            rospy.loginfo("\n\n\n\n\n\n")
            rospy.loginfo("Index for q table")

            enter_q_table = True
            rospy.loginfo("\nagent_vel_index is %d, agent_lane_index is %d, amb_vel_index is %d, amb_lane_index is %d,  rel_amb_y_index is %d, action_index is %d",agent_vel_index,agent_lane_index, amb_vel_index, amb_lane_index, rel_amb_y_index,action_index)
            if ((agent_vel_index >= self.agent_vel_dimension) or (agent_lane_index >= self.agent_lane_dimension) or (amb_vel_index >= self.amb_vel_dimension) or (amb_lane_index >= self.amb_lane_dimension) or (rel_amb_y_index >= self.rel_amb_y_dimension)):
                rospy.loginfo("\n\n\n\nSTATE IS OUTSIDE Q TABLE!!!!!!\n\n\n\n\n")
                enter_q_table = False

            # Q(s,a):
            if enter_q_table == True:
                q_of_s_a_value = \
                self.q_table[agent_vel_index][agent_lane_index][amb_vel_index][amb_lane_index][rel_amb_y_index][action_index]

                #ROS edits in problem forumation 
                if (q_of_s_a_value == -1000):
                    q_of_s_a_value = 0

                # max Q(s',a')
                max_q_of_s_value_new = np.max(self.q_table[
                                              new_agent_vel_index, new_agent_lane_index, new_amb_vel_index, new_amb_lane_index, new_rel_amb_y_index])
                if (max_q_of_s_value_new == -1000):
                    max_q_of_s_value_new = 0
            
                # Actual Update:
                q_of_s_a_value = q_of_s_a_value + self.learning_rate * (reward + self.gamma * max_q_of_s_value_new - q_of_s_a_value)


                # actual update step:
                self.q_table[agent_vel_index][agent_lane_index][amb_vel_index][amb_lane_index][rel_amb_y_index][action_index] = q_of_s_a_value

            

    
    def save_q_table(self, variables_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"saved_variables")):
        if(self.test_mode_on):
            pass  # do not save
        else:
            np.save(variables_folder_path+'/Q_TABLE.npy', self.q_table)


    def load_q_table(self, variables_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"saved_variables")):
        #rospy.loginfo("Loaded Q_TABLE from {variables_folder_path + '/Q_TABLE.npy'}")
        return np.load(variables_folder_path+'/Q_TABLE.npy')

          




    
