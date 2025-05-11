import mesa
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import pandas as pd
import numpy as np
import random

cities = pd.read_csv("data/cities.csv")
demographic = pd.read_csv("data/demographic.csv")
turnout = pd.read_csv("data/turnout.csv")

total_population_by_city = demographic.groupby("city")["population"].sum()


class Agent(mesa.Agent):
    def __init__(self, id, model, sex, age_group, income, education):
        super().__init__(id, model)
        self.sex = sex
        self.age_group = age_group
        self.income = income
        self.education = education
        self.state = "Registered" # Undecided, Registered, WillVote, Voted
        self.utility = 0
    
    def step(self):
        if self.state == "Registered":
             # TODO: Replace benefit and cost based on demographic data
            benefit = 1
            cost = random.uniform(0, 1)

            # gets surrounding 8 cells and sums those that intend to vote
            neighbor_voter_count = sum([1 for n in self.model.grid.get_neighborhood(self.pos, include_center=False, moore=True) if n.state == "WillVote"])
            social_influence = neighbor_voter_count / 8
            print(self.id, "neighbor voter count:", neighbor_voter_count)
            self.utility = benefit - cost + social_influence
            if self.utility > 0.5: # the benefits outweigh the costs
                self.state = "WillVote"
    
    def move(self):
        possible_steps = self.model.grid.get_neigborhood(
            self.pos,
            moore=False, # left, up, down, right only. no diagonals
            include_center=True # can stay in place
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


class Model(mesa.Model):
    def __init__(self, N=200, width=30, height=30):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={
                "WillVote": lambda m: sum([1 for a in m.schedule.agents])
            }
        )
        i = 0
        total_count = sum(turnout[turnout["city"] == row["city"]]["registered"].values[0] * int(row["population"]) / total_population_by_city[row["city"]] for _, row in demographic.iterrows())
        for _, row in demographic.iterrows():
            city = row["city"]
            age_group = row["age_group"]
            sex = row["sex"]
            # TODO: Add the proportion of income group and education group based on the data
            income = 0
            education = 0

            proportion = int(row["population"]) / total_population_by_city[city]
            count = turnout[turnout["city"] == city]["registered"].values[0] * proportion
            for __ in range(int(self.num_agents * count / total_count)):
                agent = Agent(i, self, sex, age_group, income, education)
                self.schedule.add(agent)
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                self.grid.place_agent(agent, (x,y))
                i += 1
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def agent_portrayal(a):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 1
    }
    if a.state == "Undecided":
        portrayal["Color"] = "gray"
        portrayal["Layer"] = 0
    elif a.state == "Registered":
        portrayal["Color"] = "gray"
        portrayal["Layer"] = 1
    elif a.state == "WillVote":
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 2
    else:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 3
    return portrayal

grid = CanvasGrid(agent_portrayal, 30, 30, 800, 800)
# TODO: Label axes of the chart
chart = ChartModule([{
    "Label": "Total number of agents that intend to vote",
    "Color": "black"
}], data_collector_name="datacollector")

server = ModularServer(
    Model,
    [grid, chart],
    "NCR Voter Turnout Model",
    {}
)
server.port = 4000