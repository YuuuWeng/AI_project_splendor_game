THINKTIME = 0.95
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

class PriorityQueue:
    def __init__(self):
        self.queue_index = 0
        self.priority_queue = []

    def push(self, item, priority):
        heapq.heappush(self.priority_queue,(priority, self.queue_index, item))
        self.queue_index += 1

    def isEmpty(self):
        return len(self.priority_queue) == 0

    def pop(self):
        return heapq.heappop(self.priority_queue)[-1]

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
        self.eid = (self.id+1) %2
        self.gamerule = SplendorGameRule(NUM_PLAYER)

    def SelectAction(self, actions, state):
        start_time = time.time()
        priority_queue = PriorityQueue()
        ## eavluate board state
        board = self.evaluate_board_state(state,self.id)
        ## return out_board = {"my_state":[{},{}], "noble_state": {}, "card_state":[{},{}]}

        for action in actions:
            if time.time() - start_time < THINKTIME:
                ## action_reawrd store the reward that this actions brings to us
                # {"score_reward":0,"gem_rewards":{},"cards_rewards":{}}
                action_rewards = self.evaluate_action(action)
                priority_queue.push(action, self.heuristic(board, action_rewards))
            else:
                break

        if priority_queue.isEmpty() == False:
            return priority_queue.pop()
        else:
            return random.choice(actions)


        return priority_queue.pop()



    def heuristic(self, on_board, action_rewards):
        ##out_board: board = {"score":0,"my_gem":{},"my_gemcard":{}, "noble": []}
        ##action_reward: rewards = {"score_reward":0,"gem_rewards":{},"gem_cards_rewards":{}}

        ## feature 1: the distance between the win !

        f1score = (15 - (on_board["score"] + action_rewards["score_reward"]))/15

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
        f2score = -(gem_change / (card_income + 1))/14

        ## whether I use less gem to get cards
        extra =  -(gem_change /(cardnum +1))/14


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
        noble_need = []
        for noble in nobles:
            f4score += abs(sum(dict(Counter(noble)-Counter(this_gem_card)).values()))
        f4score = f4score/9
        ## divide into three different phase with different wegiht
        #
        if f1score>13:
            return 300 * f1score + 20* f2score +250*f3score+100*f4score +20*extra
        elif f1score>8 and f1score< 3:
            return 200 * f1score + 20* f2score +200*f3score+150*f4score +20*extra
        else:
            return 200 * f1score + 20* f2score +100*f3score+250*f4score +20*extra

        return 25 * f1score + 1* f2score +5*f3score+15*f4score +2*extra

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

