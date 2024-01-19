# RL-Project
## Notes, 16.1.2024


## Not understood:
- what is Goal wrapper?
- what do we need of the text_crafter folder?
- Do we want to condition on the goal or does that need even longer because we have to encode the goals ?
## What I touched: 
- goal_wrapper.py
- crafter_env.py
### What I deleted: **goal_wrapper.py**
#### CrafterGoalWrapper:
##### FLAGS:
- single_task: null  # If not null, train on a single task, e.g. "chop tree" **Deleted: ALL if-cases where self._single_task = True**
- single_goal_hierarchical: False  # Train on a single goal, but suggest necessary subgoals first
--> For FINETUNING only, so first option deleted to train on single task only 

- action_space_type: harder # set to 'easier' for smaller ac space env
--> For harder environmental setting, e.g. with  make crafting table??

##### VARIABLES, LISTS, ET AL:
- custom_goals
- _cur_subtask: not needed if not single_goal_hierarchical
##### METHODS:
- set_single_task: not needed if not flagged --> 
- set_single_goal_hierarchical
- def filter_hard_goals: only for including custom goals
- def on_final_subtask: if hierarchical goals used neccessary only 
- def on_final_subtask: if hierarchical goals used neccessary only 
- def check_multistep: not used anywhere, superflous
- new_obs['old_goals']=self.prev_goals because it seems that they are set to be self.prev_goals='' in the code and i am not sure why we need them?
- 
#### CrafterLMGoalWrapper:
##### FLAGS:

- self._use_state_captioner = use_state_captioner
- self._use_transition_captioner = use_transition_captioner
- if use_state_captioner or use_transition_captioner:
    self.transition_captioner, self.state_captioner, self.captioner_logging = get_captioner()

--> Use hard coded captioner


- if self._single_task == 'survival' and not self._use_env_reward:
            reward = text_reward = .05
            info['action_success'] = False
            info['eval_success'] = False
    --> Use hard coded captioner: I think we wont use that single task ever

##### VARIABLES, LISTS, ET AL:
- self.old_lm_achievements: nowhere used in codebase
- self.old_oracle_achievements nowhere used in codebase

#### crafter_env
- def custom_reward

### What I deleted: **crafter_env.py**
##### FLAGS:
 Not sure what this does: But maybe we dont need the logging wrapper for now:
 ``` 
 if 'CrafterTextEnv' in env_spec['name']:  # NOT baseline, so we have goals and should log
            env = CrafterLoggingWrapper(CrafterLMGoalWrapper(env,
                                                             env_spec['lm_spec'],
                                                             env_spec['env_reward'],
                                                             device=device,
                                                             threshold=env_spec['threshold'],
                                                             debug=debug,
                                                             single_task=env_spec['single_task'],
                                                             single_goal_hierarchical=env_spec['single_goal_hierarchical'],
                                                             use_state_captioner=env_spec['use_state_captioner'],
                                                             use_transition_captioner=env_spec['use_transition_captioner'],
                                                             check_ac_success=env_spec['check_ac_success']
                                                             ))
        else:
            env = CrafterGoalWrapper(env, env_spec['env_reward'], env_spec['single_task'], single_goal_hierarchical=env_spec['single_goal_hierarchical'])
```

##### VARIABLES, LISTS, ET AL:
##### METHODS:

### What I added: **crafter_env.py**

from utils: 
- class ExtendedTimeStep(NamedTuple):
- class ExtendedTimeStepWrapper(dm_env.Environment):



## REFACTORED:
changed from gym to gymnasium:
- everywhere
```
obs, reward, terminated, truncated, info = self._env.step(action)
done = terminated or truncated
```
- crafter_env.py 
```
class Crafter:
self.seed = seed

...

self._env.reset(seed = self.seed)
```
WEIRD: obs['old_goals'] is with the given code not possible to tokenize because they change the " [SEP] " to "sep" which is not in the vocabulary. But i dont understand anyways where "old_goald" is needed, so i set it the same to goal. 

```
    def _get_full_obs(self, obs):
        all_goal_str = f" {self.split_str} ".join([s.lower().strip() for s in self.old_all_suggested_actions])

        goal_str = " ".join([s.lower().strip() for s in self.suggested_actions])
        self.goal_str = goal_str
        self.oracle_goal_str = " ".join([s.lower().strip() for s in self.oracle_suggested_actions])
        if self.use_sbert:
            goal_str = 'Your goal is: ' + goal_str + '.'
        obs['goal'] = goal_str
        obs['old_goals'] = all_goal_str
        return obs

    def reset(self):
        obs, info = self.env.reset()
        self._cur_subtask = 0
        self.goal_str = None
        self.oracle_goal_str = None
        self.lm.reset()
        self.oracle_lm.reset()
        self.prev_info = info
        self._make_predictions()
        self.old_all_suggested_actions = copy.deepcopy(self.suggested_actions)
        self.old_oracle_suggested_actions = []
        obs = self._get_full_obs(obs)
        obs['success'] = False
        obs['goal_success'] = np.array(0)
        obs = self.env.tokenize_obs(obs)
        self._reset_custom_task()
        return self._tokenize_goals(obs), info


def tokenize_obs(self, obs_dict):
        """
        Takes in obs dict and returns a dict where all strings are tokenized.
        """
        if self.use_sbert and isinstance(obs_dict['inv_status'], dict):
            inv_status = ""
            for k, v in obs_dict['inv_status'].items():
                if v != '.' and 'null' not in v:
                    inv_status += v + " "
            obs_dict['text_obs'] = obs_dict['text_obs'] + " " + inv_status

        new_obs = {}
        for k, v in obs_dict.items():
            # If the value is a dictionary of strings, concatenate them into a single string
            if isinstance(v, dict) and isinstance(list(v.values())[0], str):
                v = " ".join(v.values())
            # If the value is a string, tokenize it
            if isinstance(v, str):
                arr = self.tokenize_str(v)
                new_obs[k] = arr
            else:
                # Value is already tokenized (int, array, etc)
                new_obs[k] = v
        if self.use_sbert:
            new_obs['text_obs'] = self.pad_sbert(new_obs['text_obs'])
        return new_obs


```
```
all_goal_str = f" {self.split_str} ".join([s.lower().strip() for s in self.old_all_suggested_actions])
...
obs = self.env.tokenize_obs(obs)

def tokenize_obs(self, obs_dict):
...

    arr = self.tokenize_str(v)
...

def tokenize_str(self, s):

      if " " in s:
            word_list = [w.strip(string.punctuation + ' ').lower() for w in s.split()]
            word_list = [w for w in word_list if len(w) > 0]
```
--> This is the part that is i.m.o. buggy, because is you add " SEP " to the observation and then tokenize it, " SEP " --> "sep" such that it it isn't found in the vocab. 

**TODO** i cheated with the reset, it didnt work as i wanted? Check BaseEnv and Env.reset()