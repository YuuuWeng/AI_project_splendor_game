from template import Agent
import random
from collections import Counter

import time  # record time
from copy import deepcopy
import heapq
import numpy as np
from Splendor.splendor_model import SplendorGameRule

gem = {'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}
card = {'score': 0, 'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}

NUM_PLAYER = 2
DEPTH = 2
THINKTIME = 0.9

def dict_slice(adict, start, end):
    keys = list(adict.keys())

    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  # Remeber its own id
        self.gamerule = SplendorGameRule(NUM_PLAYER)
        self.opponet_id = (self.id+1)%2
        self.depth = DEPTH

    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        ##strat the minmax tree base on the rootstate and given actions and node depth
        basenode_depth = 0
        rs = deepcopy(rootstate)
        self.expand = 0
        self.start_time = time.time()
        utility, max_action = self.maximize(rs, basenode_depth)
        print(self.expand)
        return max_action


    def maximize(self, state, current_depth: int, alpha=float('-inf'), beta=float('inf')):
        current_depth += 1
        # initalize maxutility and maxaction
        maxutility = float('-inf')
        maxaction = {}
        actions = self.gamerule.getLegalActions(state, self.id)
        # keep expand until hit the depth
        if current_depth != self.depth:
            board = self.evaluate_board_state(state, self.id)
            for action in actions:
                action_rewards = self.evaluate_action(action)
                self.expand += 1
                nstate = deepcopy(state)
                next_state = self.gamerule.generateSuccessor(nstate, action, self.id)
                max_board = self.heuristic(board, action_rewards)
                # recursive to the next layer
                minutility, minaction = self.minimize(next_state,max_board, current_depth, alpha, beta)

                # pruning when the maxutility is greater than the beta and when the alpha is greater than the beta
                if maxutility >= beta or alpha >= beta:
                    return maxutility, maxaction
                # if the current max is less than new min, we will updata the max
                if maxutility < minutility:
                    maxutility = minutility
                    maxaction = action

                # update the alpha for the future pruning
                if maxutility >= alpha:
                    alpha = maxutility
        return maxutility, maxaction

    def minimize(self, state, max_board, current_depth: int, alpha=float('-inf'), beta=float('inf')):

        current_depth += 1
        minutility = float('inf')
        minaction = {}
        ## possible action for the oppent
        actions = self.gamerule.getLegalActions(state, self.opponet_id)
        # keep expand until hit the depth
        # if reach the depth for the min layer, choose the minimumutility and action for the current node
        board = self.evaluate_board_state(state, self.opponet_id)
        for action in actions:
            if time.time() - self.start_time < THINKTIME:
                action_rewards = self.evaluate_action(action)
                evaluation = max_board - self.heuristic(board, action_rewards)
                if evaluation <= minutility:
                    minutility = evaluation
                    minaction = action
            else:
                break

        return minutility, minaction


    def evaluate_board_state(self,state,id):
        board = {"score":0,"my_gem":{},"my_gemcard":{}, "noble": []}
        ## find score
        board["score"] = state.agents[id].score

        ## agent_state: [initial{},initial{}]
        my_gem = deepcopy(gem)
        my_gemcard = deepcopy(gem)

        this_agent_gem = state.agents[id].gems
        this_agent_card = state.agents[id].cards
        for color in gem.keys():
            try:
                gembycard = len(this_agent_card[color])
                ## totoal color of gem I have = this color gem I have + this color card I have
                my_gem[color] = (this_agent_gem[color] + gembycard)
                ## number of this color's card that I have
                my_gemcard[color] = (gembycard)
            ## If there is no this color
            except:
                my_gem[color] = (this_agent_gem[color])
                my_gemcard[color] = 0

        board["my_gem"] = my_gem
        board["my_gemcard"] = my_gemcard

        ## the nobles card on this state [card{}...]
        nobles = []
        for noble in state.board.nobles:
            ## get the noble relevent dictionary
            noble = noble[1]
            noble_card = deepcopy(card)
            for key in noble_card.keys():
                if key ==  "score":
                    noble_card["score"] = 3
                ## judge the relevant color
                else:
                    try:
                        noble_card[key] = (noble[key])
                    except:
                        noble_card[key] = 0
            nobles.append(noble)
        board["noble"] = nobles
        return board

    def heuristic(self, on_board, action_rewards):
        ##out_board: board = {"score":0,"my_gem":{},"my_gemcard":{}, "noble": []}
        ##action_reward: rewards = {"score_reward":0,"gem_rewards":{},"gem_cards_rewards":{}}

        ## feature 1: the distance between the win !

        f1score = (15 - (on_board["score"] + action_rewards["score_reward"]))

        ## feature 2: we want spend less gem buy cards with higher score
        gem_cost  = action_rewards["gem_rewards"].values()
        gem_change = 0
        for value in gem_cost:
            if value > 0:
                gem_change +=value
            else:
                gem_change += 8 * value
        cardnum= sum(list(action_rewards["gem_cards_rewards"].values())[1:])

        card_income = action_rewards["gem_cards_rewards"]["score"]
        f2score = -(gem_change / (card_income + 1))

        ## whether I use less gem to get cards
        extra =  -(gem_change /(cardnum +1))


        ## feature 3: we don't want to buy too much same card
        ## if we have more than 4 same gem, we gives a penalty, this will be negative, otherwise, we get more gems

        gem_values = list(on_board["my_gem"].values())[:-1]
        gem_cards = np.array(list(on_board["my_gemcard"].values())) + np.array(list(action_rewards["gem_cards_rewards"].values())[1:])
        f3score = sum([4 - i for i in gem_values ])/ (sum(gem_cards[:-1])+1)+gem_cards [-1]*2


        ##feature 4: we want buy nobles if possible
        f4score = 0
        gem_cards = Counter(on_board["my_gemcard"])
        ##remove score in the gem card ward
        re = Counter(dict_slice(action_rewards["gem_cards_rewards"],1,len(action_rewards["gem_cards_rewards"])))

        ## a diction of to store a gem card after this action :{key: color, item: card}
        this_gem_card = dict(re+gem_cards)
        nobles = on_board["noble"]
        for noble in nobles:
            f4score += abs(sum(dict(Counter(noble)-Counter(this_gem_card)).values()))
        f4score = f4score
        ## divide into three different phase with different wegiht
        #
        if f1score>13:
            return -(50 * f1score + 0.55* f2score +20*f3score+50*f4score)
        elif f1score>8 and f1score< 3:
            return -(50 * f1score + 0.55* f2score +20*f3score+50*f4score)
        else:
            return -(50 * f1score + 0.55* f2score +20*f3score+50*f4score)

        return -(20 * f1score + 0.25* f2score +20*f3score+0.5*f4score)


    def evaluate_action (self,action):
        rewards = {"score_reward":0,"gem_rewards":{},"gem_cards_rewards":{}}
        gem_rewards =deepcopy(gem)
        gem_cards_rewards = deepcopy(card)
        ##the type of the action
        action_type = action["type"]
        if action_type == "reserve":
            gem_rewards["yellow"] =1
        ## when I buy cards, I got more/less gems  and possiblely more score
        elif action_type == 'buy_available' or action_type == 'buy_reserve':

            this_card_score = action['card'].points
            rewards["score_reward"]+=this_card_score
            gem_cards_rewards["score"] = this_card_score

            ## we use the gem buy cards, we loss gems
            for color in gem_rewards.keys():
                try:
                    gem_rewards[color] = -action['returned_gems'][color]
                except:
                    gem_rewards[color] = 0
            ## our gem card with relevant color add 1
            gem_cards_rewards[action['card'].colour] = 1

        ## When I collect gems, num of gem changes
        elif action_type == "collect_same" or action_type == "collect_diff":
            for color in gem_rewards.keys():
                try:
                    gem_rewards[color] = action['collected_gems'][color]
                except:
                    gem_rewards[color] = 0
                try:
                    gem_rewards[color] -=action['returned_gems'][color]
                except:
                    gem_rewards[color] = gem_rewards[color]

        if action['noble'] != None:
            rewards["score_reward"] += 3
        rewards["gem_rewards"] = gem_rewards
        rewards["gem_cards_rewards"] = gem_cards_rewards

        return rewards