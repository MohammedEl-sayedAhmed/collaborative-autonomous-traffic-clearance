#!/usr/bin/env python
import rospy
import threading
import numpy as np
import os
import sys
from std_msgs.msg import Bool
from racecar_rl_environments.msg import areWeDone
from rl_algorithms.single_agent_qlearning import *
from env_communicators.basic_env_comm import *
from training_logger import TrainingLogger

# Lock to engage and disengage following the RL model in the test mode
test_activity_lock = threading.Lock()
test_activity_lock.acquire()

# Lock to engage and disengage following the RL model in the train mode
train_activity_lock = threading.Lock()
train_activity_lock.acquire()

class SAQLMaster:
    def __init__(self):
        
        rospy.loginfo("Initializing Single Agent Q-Learning Master.")
        self.is_activated = False # bool: whether the RL model should be activated for this agent
        self.is_episode_done = 0 # int: whether the episode is done -- 0: not done 1-4: done for different reasons
        self.episode_num = 0 # int: current episode number

        # History Variables: #TODO: save variables for displaying results later
        self.total_reward_per_episode = []  # list of doubles: cummulative reward per episode
        self.reward_history_per_episode = []  # list of lists: each is a list of all step rewards per episode

        # subscribe to /RL/is_active_or_episode_done to receive areWeDone message 
        rospy.Subscriber('/RL/is_active_or_episode_done', areWeDone, self.isActivatedOrDoneCallback, queue_size = 2)

        self.getParams()

        # Metrics logging for the live training dashboard (fail-safe: never breaks training).
        # Runs are saved per-run under saved_variables/runs/<timestamp>_<label>/ so they can be
        # browsed and compared later. Label + git SHA come from run.sh via env vars.
        runs_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'saved_variables', 'runs')
        try:
            q_params = rospy.get_param('q_learning_params') if rospy.has_param('q_learning_params') else {}
        except Exception:
            q_params = {}
        self.LOGGER = TrainingLogger(
            runs_root,
            label=os.environ.get('RL_RUN_LABEL', 'run'),
            git_sha=os.environ.get('RL_GIT_SHA', 'unknown'),
            config={'max_num_episodes': self.max_num_episodes,
                    'test_mode_on': self.test_mode_on,
                    'q_learning_params': q_params},
            mode=('test' if self.test_mode_on else 'train'),
        )

        rospy.loginfo("Finished initializing Single Agent Q-Learning Master.")
        
    def getParams(self):

        rospy.loginfo("Getting parameters in Single Agent Q-Learning Master.")

        self.robot_num = rospy.get_param('robot_num')
        rospy.loginfo("Reading the vehicle's ID number: %s.", self.robot_num)
        ####
        if rospy.has_param('test_mode_on'):
            self.test_mode_on = rospy.get_param('test_mode_on')
            rospy.loginfo("Reading test_mode_on parameter: %d.", self.test_mode_on)
        else:
            self.test_mode_on = False
            rospy.loginfo("Setting test_mode_on parameter to default: %d.", self.test_mode_on)
        ####
        #self.max_time_steps_per_episode = rospy.get_param('max_time_steps_per_episode')
        self.max_num_episodes = rospy.get_param('max_num_episodes')
        rospy.loginfo("Reading max_num_episodes parameter: %d.", self.max_num_episodes)
        self.every_n_episodes = rospy.get_param('vis_update_params/every_n_episodes')
        rospy.loginfo("Reading every_n_episodes parameter: %d.", self.every_n_episodes)
        self.every_n_steps = rospy.get_param('vis_update_params/every_n_steps')
        rospy.loginfo("Reading every_n_steps parameter: %d.", self.every_n_steps)
        self.print_reward_every_episode = rospy.get_param('vis_update_params/print_reward_every_episode')
        rospy.loginfo("Reading print_reward_every_episode parameter: %d.", self.print_reward_every_episode)
        ####
        if rospy.has_param('environment'):
            if (rospy.get_param('environment') != "basic_environment"):
                rospy.loginfo("Unknown environment is requested. Will use the basic_environment instead.")
            self.ENV_COMM = BasicEnvComm()
        else:
            self.ENV_COMM = BasicEnvComm()
            rospy.loginfo("Using the default environment: basic_environment.")
        ####
        if rospy.has_param('rl_algorithm'):
            if (rospy.get_param('rl_algorithm') != "single_agent_qlearning"):
                rospy.loginfo("Unknown RL algorithm is requested.")
            rospy.loginfo("Using the default RL algorithm: single_agent_qlearning.")
            self.RL_ALGO = SingleAgentQlearning(environment_init = self.ENV_COMM.getWindowParams(), algo_params = rospy.get_param('q_learning_params'), load_q_table = rospy.get_param('load_q_table'), test_mode_on = self.test_mode_on)
        else:
            rospy.loginfo("Using the default RL algorithm: single_agent_qlearning.")
            self.RL_ALGO = SingleAgentQlearning(environment_init = self.ENV_COMM.getWindowParams(), algo_params = rospy.get_param('q_learning_params'), load_q_table = rospy.get_param('load_q_table'), test_mode_on = self.test_mode_on)
        ####

    # Input data is areWeDone message from topic /RL/is_active_or_episode_done
    def isActivatedOrDoneCallback(self, inputMsg):
        # in case of testing
        if (self.test_mode_on == True):
            # release lock if RL model was disengaged and needs to be engaged noew
            if ((self.is_activated == False) and (inputMsg.is_activated == True)):
                #if (test_activity_lock.locked() == True):
                    test_activity_lock.release()
            self.is_activated = inputMsg.is_activated
        # in case of training
        else:
            # if RL model was disengaged
            if (self.is_activated == False):
                # release lock if needs to be engaged now or the episode ended
                if ((inputMsg.is_activated == True) or (inputMsg.is_episode_done != 0)):
                    if (train_activity_lock.locked() == True):
                        train_activity_lock.release()
            self.is_activated = inputMsg.is_activated
            self.is_episode_done = inputMsg.is_episode_done


    # Main execution function
    def execute(self):

        rospy.loginfo("\n\n\n\n\n\n\n\n\n\n\n\n\n")
        rospy.loginfo("Starting Simulation.")
        resp = self.ENV_COMM.startSim(1, 1)
        if (resp.is_successful == False):
            raise Exception("Could not start simulation. Terminating.")
            return
        rospy.loginfo("Simulation Started Successfully.")


        if (self.test_mode_on == True):
            self.test_model()

        else:
            rospy.loginfo("RL MODEL TRAINING MODE ACTIVATED.")

            self.is_episode_done = 0

            rospy.loginfo("Starting Episode Number: %d.", self.episode_num)
            episode_reward, episode_reward_list = self.episode()
            rospy.loginfo("Logging Rewards For Episode Number: %d.", self.episode_num)
            self.total_reward_per_episode.append(episode_reward)
            self.reward_history_per_episode.append(episode_reward_list)

            rospy.loginfo("Saving a Q-table Version.")
            self.RL_ALGO.save_q_table() #TODO uncomment


            while(self.episode_num < (self.max_num_episodes - 1)):

                self.episode_num += 1
                self.is_episode_done = 0

                rospy.loginfo("Resetting Simulation For Episode Number: %d.", self.episode_num)
                resp = self.ENV_COMM.resetSim()
    
                if (resp.is_successful == False):
                    raise Exception("Could not reset simulation. Terminating.")
                    return
                rospy.loginfo("Simulation Reset Successfully.")

                #TODO: will we load the last saved qtable? and where to else to do so
                rospy.loginfo("Starting Episode Number: %d.", self.episode_num)
                episode_reward, episode_reward_list = self.episode()
                rospy.loginfo("Logging Rewards For Episode Number: %d.", self.episode_num)
                self.total_reward_per_episode.append(episode_reward)
                self.reward_history_per_episode.append(episode_reward_list)

                rospy.loginfo("Saving a Q-table Version.")
                self.RL_ALGO.save_q_table() #TODO uncomment

        
            # Save Q-table after episodes ended:
            rospy.loginfo("Saving Final Q-table.")
            self.RL_ALGO.save_q_table() #TODO TODO: see when to call so that we have a constantly updated qtable saved

            self.LOGGER.finalize("completed")

            rospy.loginfo("Closing Simulation.")
            self.ENV_COMM.closeSim()
            if (resp.is_successful == False):
                raise Exception("Could not close simulation. Terminating.")
                return
            rospy.loginfo("Simulation Closed Successfully.") #TODO: sys.exit??


    def test_model(self):

        rospy.loginfo("RL MODEL TESTING MODE ACTIVATED.")
        step = 0

        # MAIN LOOP

        while (1):

            ######## Check if RL model should be disengaged ########
            if (self.is_activated == False):
                rospy.loginfo("Disengaging the RL model for agent %d because the EV is outside the agent's window.", self.robot_num)
                self.RL_ALGO.disengage()
                test_activity_lock.acquire()
            ########################################################                

            step += 1

            # 1: Store state before taking action
            rospy.loginfo("Getting agent's state before taking action.")
            agent_state_before = self.ENV_COMM.getState(self.robot_num)
            

            # ----------------------------------------------------------------- #
            #              2:    E X E C U T E      A C T I O N                 #
            # ----------------------------------------------------------------- #

            rospy.loginfo("Executing action.")
            executed_action, execution_time = self.RL_ALGO.take_action(agent_state_before)
            rospy.loginfo("Done executing action.")


            # 3: Store state after taking action
            rospy.loginfo("Getting agent's state after taking action.")
            agent_state_after = self.ENV_COMM.getState(self.robot_num)


            # 4: Print step info
            if (step % self.every_n_steps == 0):
                rospy.loginfo("\n\nAgent's State BEFORE Taking Ation: ")
                rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_before.agent_vel, agent_state_before.agent_lane, agent_state_before.amb_vel, agent_state_before.amb_lane, agent_state_before.rel_amb_y)

                rospy.loginfo("Agent's Last Aciton: %s", executed_action)

                rospy.loginfo("\n\nAgent's State AFTER Taking Ation: ")
                rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_after.agent_vel, agent_state_after.agent_lane, agent_state_after.amb_vel, agent_state_after.amb_lane, agent_state_after.rel_amb_y)

            
        return


    def episode(self):

        ########################
        # 1: Inits
        step = 0  # step number

        episode_reward = 0
        episode_reward_list = []

        ########################
        # Measure initial state
        rospy.loginfo("Getting agent's state before taking action.")
        agent_state = self.ENV_COMM.getState(self.robot_num)

        ########################
        # 2: MAIN LOOP

        # Logging
        if (self.episode_num % self.every_n_episodes == 0):
            rospy.loginfo("Episode: %d. Epsilon: %f.", self.episode_num, self.RL_ALGO.epsilon)
            rospy.loginfo("\n\nState at episode start: ")
            rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state.agent_vel, agent_state.agent_lane, agent_state.amb_vel, agent_state.amb_lane, agent_state.rel_amb_y)
            

        while (1):

            step += 1
            rospy.loginfo("Step: %d.", step)


            ######## Check if RL model should be disengaged ########
            if (self.is_activated == False):
                rospy.loginfo("Disengaging the RL model for agent %d in episode %d because the EV is outside the agent's window.", self.robot_num, self.episode_num)
                self.RL_ALGO.disengage() #TODO uncomment
                #train_activity_lock.acquire()
                while ((self.is_activated == False) and (self.is_episode_done == 0)):
                    continue

            ########################################################

            ############### Check if episode is done ###############

            if (self.is_episode_done): # DO NOT REMOVE THIS (IT BREAKS IF WE ARE DONE)
                if(self.episode_num % self.every_n_episodes == 0):
                    if (self.is_episode_done == 1):
                        episode_end_reason = "reahed max time steps"
                    elif (self.is_episode_done == 2):
                        episode_end_reason = "ambulance reached goal"
                    elif (self.is_episode_done == 3):
                        episode_end_reason = "agent reached goal"
                    elif (self.is_episode_done == 4):
                        episode_end_reason = "simulation died"
                    else:
                        episode_end_reason = "unknown"

                    rospy.loginfo("Episode: %d ended for the reason: %s", self.episode_num, episode_end_reason)
                    rospy.loginfo("Episode: %d. Step: %d. CumReward: %f.", self.episode_num, step, episode_reward)
                break
            ########################################################


            # 3.1: Store state before taking action
            rospy.loginfo("Getting agent's state before taking action.")
            agent_state_before = self.ENV_COMM.getState(self.robot_num)
            rospy.loginfo("\n\nAgent's State BEFORE Taking Ation: ")
            rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_before.agent_vel, agent_state_before.agent_lane, agent_state_before.amb_vel, agent_state_before.amb_lane, agent_state_before.rel_amb_y)


            # ----------------------------------------------------------------- #
            # 3.2:   M O V E      O N E      S I M U L A T I O N       S T E P  #
            # ----------------------------------------------------------------- #
            #                   E X E C U T E      A C T I O N                  #
            # ----------------------------------------------------------------- #

            rospy.loginfo("Taking action.") #TODO uncomment below
            executed_action, execution_time = self.RL_ALGO.take_action(agent_state_before)  #TODO: raise error if no action is feasible
           

            # 3.3: measurements and if we are done check
            rospy.loginfo("Getting agent's state after taking action.")
            agent_state_after = self.ENV_COMM.getState(self.robot_num)
            rospy.loginfo("\n\nAgent's State AFTER Taking Ation: ")
            rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_after.agent_vel, agent_state_after.agent_lane, agent_state_after.amb_vel, agent_state_after.amb_lane, agent_state_after.rel_amb_y)



            if executed_action != "NOACTION":
                # 3.4: reward last step's chosen action
                rospy.loginfo("Calculating reward.") #TODO uncomment below
                reward = self.ENV_COMM.calcReward(agent_state_before.amb_vel, execution_time) #TODO need to be saved #FIXME #LOVE<3
                rospy.loginfo("Reward is %f", reward)
                episode_reward = episode_reward + reward  # for history
                episode_reward_list.append(reward)  # for history


                # 3.5: update q table using backward reward logic
                rospy.loginfo("Updating Q-table.") #TODO uncomment below
                self.RL_ALGO.update_q_table(reward, agent_state_after)

                # Logging
                if (step % self.every_n_steps == 0 and self.episode_num % self.every_n_episodes == 0): # print step info
                    rospy.loginfo("Episode: %d. Step: %d. LastActionMethod: %s. LastAction: %s. Reward: %f. CumReward: %f. NewState:", self.episode_num, step, self.RL_ALGO.action_chosing_method, executed_action, reward, episode_reward) #TODO uncomment
            
            else:        
                reward = 0
                rospy.loginfo("Reward FLAG, action NOT Executed")
            
            rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_after.agent_vel, agent_state_after.agent_lane, agent_state_after.amb_vel, agent_state_after.amb_lane, agent_state_after.rel_amb_y)

            # live heartbeat for the dashboard (current episode/step/reward)
            self.LOGGER.heartbeat(self.episode_num, step, episode_reward, self.RL_ALGO.epsilon)

            ############### Check if episode is done ###############
            if (self.is_episode_done): # DO NOT REMOVE THIS (IT BREAKS IF WE ARE DONE)
                if(self.episode_num % self.every_n_episodes == 0):
                    if (self.is_episode_done == 1):
                        episode_end_reason = "reached max time steps"
                    elif (self.is_episode_done == 2):
                        episode_end_reason = "ambulance reached goal"
                    elif (self.is_episode_done == 3):
                        episode_end_reason = "agent reached goal"
                    elif (self.is_episode_done == 4):
                        episode_end_reason = "simulation died"
                    else:
                        episode_end_reason = "unknown"

                    rospy.loginfo("Episode: %d ended for the reason: %s", self.episode_num, episode_end_reason)
                    rospy.loginfo("Episode: %d. Step: %d. LastActionMethod: %s. LastAction: %s. Reward: %f. CumReward: %f. NewState:", self.episode_num, step, self.RL_ALGO.action_chosing_method, executed_action, reward, episode_reward)
                    rospy.loginfo("\nAgent Velocity: %f, \nAgent Lane: %d, \nAmbulance Velocity: %f, \nAmbulance Lane: %d, \nRelative Position: %f\n", agent_state_after.agent_vel, agent_state_after.agent_lane, agent_state_after.amb_vel, agent_state_after.amb_lane, agent_state_after.rel_amb_y)
                break
            ########################################################


        # Episode End
        # 4: Update Epsilon after episode is done
        old_epsilon = self.RL_ALGO.epsilon
        self.RL_ALGO.epsilon = self.RL_ALGO.min_epsilon + (self.RL_ALGO.max_epsilon - self.RL_ALGO.min_epsilon) * \
                             np.exp(-self.RL_ALGO.decay_rate * self.episode_num)

        # Logging
        if (self.episode_num % self.every_n_episodes == 0):
            rospy.loginfo("Episode: %d. FinalCumReward: %f. OldEpsilon: %f. NewEpsilon: %f.", self.episode_num, episode_reward, old_epsilon, self.RL_ALGO.epsilon)

        if(self.print_reward_every_episode and self.episode_num % self.every_n_episodes):
            rospy.loginfo("Episode: %d. FinalCumReward: %f.", self.episode_num, episode_reward)

        # record this episode's summary for the dashboard (learning curve point)
        self.LOGGER.log_episode(self.episode_num, episode_reward, step, old_epsilon, self.is_episode_done)

        return episode_reward, episode_reward_list


if __name__ == "__main__":
    rospy.init_node('single_agent_qlearning_master', disable_signals=True)

    # create object from class SAQLMaster
    saqlm = SAQLMaster()

    # call the main execution function
    saqlm.execute()

    rospy.spin()
