""" Env wrapper which adds goals and rewards """

import pickle as pkl
import time
import copy
import os
import pathlib
import numpy as np
import torch
from text_crafter import lm
from sentence_transformers import SentenceTransformer, util as st_utils


class CrafterGoalWrapper:
    """ Goal wrapper for baselines. Used for baselines and single-goal eval."""
    def __init__(self, env, env_reward):
        self.env = env
        self._use_env_reward = env_reward
        self.prev_goal = ""
        #self._single_goal_hierarchical = single_goal_hierarchical
        self.use_sbert_sim = False
        self.goals_so_far = dict.fromkeys(self.env.action_names) #  dict.fromkeys(self.env.good_action_names)?
        self._cur_subtask = 0
        # self.custom_goals = ['plant row', 'make workshop', 'chop grass with wood pickaxe', 'survival', 'vegetarianism', 'deforestation',
        #                      'work and sleep', 'gardening', 'wilderness survival']
        self.custom_goals = []

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)

    def set_env_reward(self, use_env_reward):
        """ If this is true, we use the env reward, not the text reward."""
        self._use_env_reward = use_env_reward
        
    def goal_compositions(self):
        """ Returns a dictionary with each goal, and the prereqs needed to achieve it. """
        goal_comps = {
            'eat plant' : ['chop grass', 'place plant', 'eat plant'],
            'attack zombie' : ['attack zombie'],
            'attack skeleton' : ['attack skeleton'],
            'attack cow' : ['attack cow'],
            'chop tree' : ['chop tree'],
            'mine stone': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone'],
            'mine coal' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine coal'],
            'mine iron': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine iron'],
            'mine diamond' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron pickaxe', 'mine diamond'],
            'drink water' : ['drink water'],
            'chop grass' : ['chop grass'],
            'sleep' : ['sleep'],
            'place stone' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'place stone'],
            'place crafting table' : ['chop tree', 'place crafting table'],
            'place furnace' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'mine stone', 'mine stone', 'mine stone', 'place furnace'],
            'place plant': ['chop grass', 'place plant'],
            'make wood pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe'],
            'make stone pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe'],
            'make iron pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron pickaxe'],
            'make wood sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood sword'],
            'make stone sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone sword'],
            'make iron sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron sword'],
            'plant row': ['chop grass', 'place plant', 'chop grass', 'plant grass'],
            'chop grass with wood pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'chop grass with wood pickaxe'],
            'vegetarianism': ['drink water', 'chop grass'],
            'make workshop': ['chop tree', 'place crafting table', 'chop tree', 'place crafting table'],
            'survival': ['survival'],
            'deforestation': ['chop tree', 'chop tree', 'chop tree', 'chop tree', 'chop tree'],
            'work and sleep': ['chop tree', 'sleep', 'place crafting table'],
            'gardening': ['chop grass', 'chop tree', 'place plant'],
            'wilderness survival': ['sleep', 'chop grass', 'attack zombie'],
            
        }
        return goal_comps

    def _tokenize_goals(self, new_obs):
        if self.use_sbert:  # Use SBERT tokenizer
            new_obs['goal'] = self.env.pad_sbert(new_obs['goal'])
            new_obs['old_goals'] = self.env.pad_sbert(new_obs['old_goals'])
        return new_obs

    def reset(self):
        """Reset the environment, adding in goals."""
        # Parametrize goal as enum.
        obs, info = self.env.reset()
        self.goal_str = ""
        self.oracle_goal_str = ""
        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        print('old goals', obs['old_goals'])
        print('goals', obs['goal'])
        obs['goal_success'] = np.array(0) # 0 b/c we can't succeed on the first step
        obs = self.tokenize_obs(obs)
        return self._tokenize_goals(obs), info
    

    def set_end_on_success(self, end_on_success):
        """ When end_on_success is set, we end the episode when the agent succeeds."""
        self._end_on_success = end_on_success
        return end_on_success


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # Replace env reward with text reward
        goal_reward = 0
        if not self._use_env_reward:
            if info['action_success']:  # Only reward on an action success
                goal_reward = 1 # reward for any env success if no task is specified
                reward = info['health_reward'] + goal_reward
            else:
                reward = 0 # Don't compute reward if action failed.

        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        obs['goal_success'] = info['eval_success'] and goal_reward > 0
        obs = self.tokenize_obs(obs)
        return self._tokenize_goals(obs), reward, terminated, truncated, info

class CrafterLMGoalWrapper(CrafterGoalWrapper):
    def __init__(self, 
                 env, 
                 lm_spec, 
                 env_reward, 
                 device=None, 
                 threshold=.5, 
                 debug=True, 
                check_ac_success=True): 
        super().__init__(env, env_reward)
        self.env = env
        self.debug = debug
        self.goal_str = "" # Text describing the goal, e.g. "chop tree"
        self.oracle_goal_str = ""
        self.prev_goal = ""
        self.goals_so_far = {}
        self.sbert_time = 0
        self.cache_time = 0
        self.cache_load_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_successes = 0
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'embedding_cache.pkl'

        # Language model setup.
        prompt_format = getattr(lm, lm_spec['prompt'])()
        lm_class = getattr(lm, lm_spec['lm_class'])
        self.check_ac_success = check_ac_success
        if 'Baseline' in lm_spec['lm_class']:
            lm_spec['all_goals'] = self.action_names.copy()
            self.check_ac_success = False
        self.lm = lm_class(prompt_format=prompt_format, **lm_spec)
        self.oracle_lm = lm.SimpleOracle(prompt_format=prompt_format, **lm_spec) 
        self.use_sbert_sim = True
        self.threshold = threshold
        self.embed_lm = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        self.device = torch.device(device)
        self.cache = {}
        self.suggested_actions = []
        self.all_suggested_actions = []
        self.oracle_suggested_actions = []

        self.unit_cache_time, self.unit_query_time = [], []
        #self._end_on_success = False ## Needed if subtasks are used and final goal is achieved
        self.transition_caption = self.state_caption = None
        self.prev_info = None

    def _save_caches(self):
        if self.debug: pass
        start_time = time.time()
        # The cache will be used by multiple processes, so we need to lock it.
        # We will use the file lock to ensure that only one process can write to the cache at a time.
        
        with open(self.cache_path, 'wb') as f:
            pkl.dump(self.cache, f)
        self.cache_load_time += time.time() - start_time

    def load_and_save_caches(self):
        new_cache = self._load_cache()
        # Combine existing and new cache
        self.cache = {**new_cache, **self.cache}
        self._save_caches()

    def _load_cache(self):
        if self.debug:
            self.cache = {}
            return {}
        start_time = time.time()
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pkl.load(f)
            except FileNotFoundError:
                cache = {}
        self.cache_load_time += time.time() - start_time
        return cache

    def text_reward(self, action_embedding, rewarding_actions, update_suggestions=True):
        """
            Return a sparse reward based on how close the task is to the list of actions
            the  LM proposed.
        """
        text_rew = 0
        best_suggestion = None

        # If there are no suggestions, there is no reward
        if len(rewarding_actions) == 0:
            return 0, None

        if self.device is None:
            raise ValueError("Must specify device for real LM")

        # Cosine similarity reward
        suggestion_embeddings, suggestion_strs, updated_cache = self._get_model_embeddings(rewarding_actions)

        # action_name = self.get_action_name(action_embedding)
        action_name = action_embedding
        action_embedding, updated_cache_action = self._get_model_embedding(action_name)

        # Compute the cosine similarity between the action embedding and the suggestion embeddings
        cos_scores = st_utils.pytorch_cos_sim(action_embedding, suggestion_embeddings)[0].detach().cpu().numpy()

        # Compute reward for every suggestion over the threshold
        for suggestion, cos_score in zip(suggestion_strs, cos_scores):
            # print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
            if cos_score > self.threshold:
                print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
                if suggestion in self.all_suggested_actions and update_suggestions: # is that what they mean by only rewarding new suggestions?
                    self.all_suggested_actions.remove(suggestion)
                text_rew = max(cos_score, text_rew)
        if text_rew > 0:
            best_suggestion = suggestion_strs[np.argmax(cos_scores)]
            print(text_rew, best_suggestion)
        return text_rew, best_suggestion
    
    ## TODO: Refactor this to be more efficient

    # def _get_model_embeddings(self, str_list):
    #     num_strs_in_cache = 0
    #     num_strs_not_in_cache = 0
    #     all_embeddings = []
    #     for str in str_list:
    #         embedding, updated_cache_action = self._get_model_embedding(str)
    #         num_strs_in_cache += int(updated_cache_action)
    #         num_strs_not_in_cache += int(not updated_cache_action)
    #         all_embeddings.append(embedding)

    #     self.cache_hits += num_strs_in_cache
    #     self.cache_misses += num_strs_not_in_cache
        
    #     all_embeddings = torch.stack(all_embeddings)
    #     return all_embeddings, str_list, len(num_strs_not_in_cache) > 0
    
    def _get_model_embeddings(self, str_list):
        assert isinstance(str_list, list)
        # Split strings into those in cache and those not in cache
        strs_in_cache = []
        strs_not_in_cache = []
        for str in str_list:
            if str in self.cache:
                strs_in_cache.append(str)
            else:
                strs_not_in_cache.append(str)
        all_suggestions = strs_in_cache + strs_not_in_cache

        # Record how many strings are in/not in the cache
        self.cache_hits += len(strs_in_cache)
        self.cache_misses += len(strs_not_in_cache)

        # Encode the strings which are not in cache
        if len(strs_not_in_cache) > 0:
            start_time = time.time()
            embeddings_not_in_cache = self.embed_lm.encode(strs_not_in_cache, convert_to_tensor=True, device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            assert embeddings_not_in_cache.shape == (len(strs_not_in_cache), 384) # size of sbert embeddings
            # Add each (action, embedding) pair to the cache
            for suggestion, embedding in zip(strs_not_in_cache, embeddings_not_in_cache):
                self.cache[suggestion] = embedding
            updated_cache = True
        else:
            embeddings_not_in_cache = torch.FloatTensor([]).to(self.device)
            updated_cache = False

        # Look up the embeddings of the strings which are in cache
        if len(strs_in_cache) > 0:
            start_time = time.time()
            embeddings_in_cache = torch.stack([self.cache[suggestion] for suggestion in strs_in_cache]).to(self.device)
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            assert embeddings_in_cache.shape == (len(strs_in_cache), 384) # size of sbert embeddings
        else:
            embeddings_in_cache = torch.FloatTensor([]).to(self.device)

        # Concatenate the embeddings of the suggestions in the cache and the suggestions not in the cache
        suggestion_embeddings = torch.cat((embeddings_in_cache, embeddings_not_in_cache), dim=0)
        return suggestion_embeddings, all_suggestions, updated_cache

    def _get_model_embedding(self, action_name):
        " return the embedding for the action name, and a boolean indicating if the cache was updated"
        if action_name in self.cache:
            start_time = time.time()
            embedding = self.cache[action_name]
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            #assert embedding.shape == (1, 384) # size of sbert embeddings
            return embedding, False
        else:
            start_time = time.time()
            embedding = self.embed_lm.encode(action_name, convert_to_tensor=True, device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            self.cache[action_name] = embedding
            return embedding, True

    def _get_full_obs(self, obs):
        all_goal_str = f" {self.split_str} ".join([s.lower().strip() for s in self.old_all_suggested_actions])
        print(all_goal_str)
        goal_str = " ".join([s.lower().strip() for s in self.suggested_actions])
        self.goal_str = goal_str
        self.oracle_goal_str = " ".join([s.lower().strip() for s in self.oracle_suggested_actions])
        if self.use_sbert:
            goal_str = 'Your goal is: ' + goal_str + '.'
        obs['goal'] = goal_str
        obs['old_goals'] = goal_str # TODO: THIS IS WEIRD: all_goal_str
        return obs

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
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
        print(obs)
        obs = self.env.tokenize_obs(obs)
        return self._tokenize_goals(obs), info

    def _make_predictions(self):
        text_obs, inv_status = self.env.text_obs()
        caption = text_obs
        self.suggested_actions = self.lm.predict_options({'obs': caption, **inv_status}, self)
        self.oracle_suggested_actions = self.oracle_lm.predict_options({'obs': text_obs, **inv_status}, self)

        # Filter out bad suggestions
        if self.threshold == 1:  # Exact match
            self.suggested_actions = [s for s in self.suggested_actions if s in self.action_names]
        
        self.all_suggested_actions = copy.deepcopy(self.suggested_actions)

        for suggestion in self.suggested_actions:
            self.goals_so_far[suggestion] = None

   
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.old_all_suggested_actions = copy.deepcopy(self.all_suggested_actions)
        self.old_suggested_actions =  copy.deepcopy(self.suggested_actions)
        self.old_oracle_suggested_actions = copy.deepcopy(self.oracle_suggested_actions)
        info['env_reward'] = reward

        health_reward = info['health_reward']
        # replace with text reward
        text_reward = 0
        closest_suggestion = None
        info['goal_achieved'] = None

        if (info['action_success'] and self.check_ac_success) or not self.check_ac_success:
            # If single_task is true, we only reward for exact matches
            action_name = self.get_action_name(action)
            text_reward, closest_suggestion = self.text_reward(action_name, self.old_suggested_actions, update_suggestions=True)
            if not self._use_env_reward:
                reward = health_reward + text_reward
            self.lm.take_action(closest_suggestion)
            self.oracle_lm.take_action(action_name)
            info['goal_achieved'] = closest_suggestion
        else:
            if not self._use_env_reward:
                reward = health_reward # Don't compute lm reward if action failed
                

        info['text_reward'] = text_reward
        self.prev_info = info
        self._make_predictions()
        obs = self._get_full_obs(obs)
        obs['success'] = info['action_success']
        obs['goal_success']  = int(info['eval_success'] and text_reward > 0)
        if text_reward > 0:
            print(f"Goal success {obs['goal_success']}, {info['action_success']}, {text_reward}")
        obs = self.env.tokenize_obs(obs)
        return self._tokenize_goals(obs), reward, terminated, truncated, info

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)
