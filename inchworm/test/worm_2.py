# -*- coding: utf-8 -*-
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, fixtureDef)
import wx
import numpy as np
import random as rd
import math
import itertools
import matplotlib.pyplot as plt
import sys
import os
import json
import glob

import genetic_algorithm as ga

Draw = True
PPM = 5.0  # pixels per meter
FPS = 60

Section_num = 100

"""
Agent_num = 21
Parts_num = 6
Step_num = 9
Frame_perStep = 15
Cycle_per_sec = 2
"""

Limit_agt_x = 3
Cell_x = 54
Cell_y = 11

Interbal = 50
"""
Step_clossCycl = Frame_perStep * Step_num
Frame_perSec = Step_clossCycl * Cycle_per_sec + Interbal
"""

colors = {
    staticBody: (150,150,150),
    dynamicBody: (112,146,190),
}


class Operator(wx.Frame):
    agent_num = 6
    parts_num = 6
    step_num = 9
    FPStep = 15
    cycle_perSec = 2
    step_clossCycl = FPStep * step_num
    FPSec = step_clossCycl * cycle_perSec + Interbal

    key_dict = {"1": [0, 1], "2": [1, 1], "3": [2, 1], "4": [3, 1], "5": [4, 1], "6": [5, 1],
                "!": [0, -1], "\"": [1, -1], "#": [2, -1], "$": [3, -1], "%": [4, -1], "&": [5, -1]}
    value_ctlg = [-1, 0, 1]

    def __init__(self, parent=None, id=-1, title=None, loadLog=False, initValue=None, remakeLower=False, resize=False, demo=False):
        if demo == False:
            if loadLog == False:
                self.ga_instance = ga.G_algorithm(Operator.agent_num, Operator.parts_num-1, Operator.step_num, Operator.value_ctlg)
                self.generation = 0
            else:
                self.ga_instance = ga.G_algorithm(loadLog=True)
                self.generation = self.ga_instance.generation
                Worm.rote_speed = self.ga_instance.loadedData["rote_speed"]
                Worm.torque = self.ga_instance.loadedData["torque"]

                Operator.parts_num = self.ga_instance.loadedData["parts_num"]
                Operator.step_num = self.ga_instance.loadedData["step_num"]
                Operator.FPStep = self.ga_instance.loadedData["FPStep"]

                if initValue != None:
                    self.ga_instance.remake_lower(initValue=0)
                elif remakeLower == True:
                    self.ga_instance.remake_lower()

                if resize == True:
                    self.ga_instance.resize(Operator.agent_num)
        else:
            json_list = glob.glob(ga.G_algorithm.save_path+"agent_data/*.json")
            Operator.agent_num = len(json_list)

        self.demo = demo

        self.numCell_row = math.ceil(Operator.agent_num / Limit_agt_x)
        if Operator.agent_num < Limit_agt_x:
            self.numCell_clm = Operator.agent_num
        else:
            self.numCell_clm = Limit_agt_x
        self.winPx_x = Cell_x * self.numCell_clm * PPM
        self.winPx_y = Cell_y * self.numCell_row * PPM

        # self.winPx_x = 600
        # self.winPx_y = 600

        # 1. 描画ウィンドウの初期化、タイマー初期化
        wx.Frame.__init__(self, parent, id, title)
        self.mainPanel = wx.Panel(self, size=(self.winPx_x, self.winPx_y))
        self.mainPanel.SetBackgroundColour('WHITE')

        self.panel = wx.Panel(self.mainPanel, size = (self.winPx_x, self.winPx_y))
        self.panel.SetBackgroundColour('WHITE')

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(self.panel)

        self.SetSizer(mainSizer)
        self.Fit()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer)

        self.cdc = wx.ClientDC(self.panel)
        w, h = self.panel.GetSize()
        self.bmp = wx.Bitmap(w,h)

        # 2. CloseWindowバインド
        self.Bind(wx.EVT_CLOSE, self.CloseWindow)

        # 3. 物理エンジンの初期化 + 物理オブジェクト(地面とか物体とか)の定義
        self.phys_world = world(gravity=(0, -10), doSleep=True)

        """
        genome_list = self.ga_instance.get_genomes()
        for index in range(Agent_num):
            self.agent_list[index].genome = genome_list[index]
        # self.code = self.ga_instance.get_genomes()
        """

        self.obj_list = []
        self.ground_list = []
        self.wall_list = []
        self.mk_ground()

        self.agent_list = []
        if demo == False:
            self.mk_agent(self.ga_instance.get_genomes())
        else:
            self.mk_agent(load=True)

        self.idx_finish = []
        self.idx_ovTurned = []

        self.centerX_list_1 = self.collect_centerX()
        self.centerX_list_2 = {}

        # 4. タイマーループ開始
        self.timer.Start(1000/FPS)

        if demo == False:
            self.tick = 0
        else:
            self.tick = 1
        # self.step = 0
        self.sec_time = 0
        self.count_sec = 1

        self.txt_list = {}
        if demo == False:
            self.txt_list["section"] = wx.StaticText(self.panel, pos=(30, 0))
            self.txt_list["generation"] = wx.StaticText(self.panel, pos=(90, 0))
            self.set_label()
        else:
            self.txt_list["section"] = wx.StaticText(self.panel, pos=(30, 0))
            self.txt_list["section"].SetLabel("section: demonstration")

        self.layout_index = wx.GridSizer(rows=self.numCell_row, cols=self.numCell_clm, gap=(0, 0))
        index_list = range(self.numCell_clm*self.numCell_row)
        index_list = reversed([index_list[index:index + self.numCell_clm] for index in range(0, len(index_list), self.numCell_clm)])
        for list_in in index_list:
            for index in list_in:
                self.layout_index.Add(wx.StaticText(self.panel, label=str(index)),flag=wx.TOP|wx.LEFT, border=5)
        self.panel.SetSizer(self.layout_index)

        if demo == False:
            print("*****************")
            self.view_info()

    def view_info(self):
        print("section:", self.count_sec)
        print("generation:", self.generation)
        # ave_vari = self.ga_instance.similarity()
        print("average:", "{:.3g}".format(self.ga_instance.cosSim_ave), end="")
        print("  variance:", "{:.3g}".format(self.ga_instance.cosSim_vari))
        for index in range(Operator.agent_num):
            if self.ga_instance.mute_list[index] == False:
                print(index)
            elif self.ga_instance.mute_list[index] == True:
                print(index, "(mutation)")
            elif self.ga_instance.mute_list[index] == "clone":
                print(index, "(clone)")
            print(self.agent_list[index].genome, "\n")

    def collect_centerX(self):
        centerX_list = {}
        for index in range(Operator.agent_num):
            if index not in self.idx_finish:
                centerX_list[index] = self.agent_list[index].get_centerX()
        return centerX_list

    def distances(self):
        list_2 = {key:self.centerX_list_2[key] for key in sorted(self.centerX_list_2)}
        list_1 = list(self.centerX_list_1.values())
        list_2 = list(list_2.values())
        return [x2-x1 for x1, x2 in zip(list_1, list_2)]

    def penalty(self, score):
        for index in self.idx_ovTurned:
            if score[index] >= 0:
                score[index] *= 0.5
            else:
                score[index] *= 2
        return score

    def mk_ground(self):
        for index in range(self.numCell_row):
            self.ground_list.append(self.phys_world.CreateStaticBody(
                position=(0, 0.5 + Cell_y * index)
            ))
            fixture = fixtureDef(
                shape=polygonShape(box=(Cell_x * self.numCell_clm, 0.5))
            )
            # fixture.filter.groupIndex = -2
            self.ground_list[-1].CreateFixture(fixture)
            self.obj_list.append(self.ground_list[-1])

        for index in range(self.numCell_clm+1):
            self.wall_list.append(self.phys_world.CreateStaticBody(
                position=(Cell_x * index, 0),
                shapes=polygonShape(box=(0.5, Cell_y * self.numCell_row)),
            ))
            self.obj_list.append(self.wall_list[-1])

    def mk_agent(self, genome_list=None, load=False):
        posi_list = list(itertools.product(range(self.numCell_row), range(self.numCell_clm)))[0:Operator.agent_num]
        if load == False:
            for position, genome in zip(posi_list, genome_list):
                self.agent_list.append(Worm(Operator.parts_num, Operator.step_num, Worm.rote_speed, Worm.torque, Operator.FPStep, position, self.phys_world, genome))
            self.obj_list += sum([agent.parts_list for agent in self.agent_list], [])
            # self.motor_lastValue = [[0 for __ in range(Parts_num-1)] for _ in range(Agent_num)]
        else:
            name_list = os.listdir(ga.G_algorithm.save_path+"agent_data")
            for name, position in zip(name_list, posi_list):
                with open(ga.G_algorithm.save_path+"agent_data/"+name, "r") as file:
                    dict = json.load(file)
                dict["world"] = self.phys_world
                dict["position"] = position
                self.agent_list.append(Worm(**dict))
            self.obj_list += sum([agent.parts_list for agent in self.agent_list], [])

    """
    def ctrl(self, idx_agent, code_clm):
        for idx_motor, value in enumerate(code_clm):
            if value != self.motor_lastValue[idx_agent][idx_motor]:
                self.agent_list[idx_agent].rote(idx_motor, value)
                self.motor_lastValue[idx_agent][idx_motor] = value
    """

    def superCtrl(self):
        for agent in self.agent_list:
            agent.control(self.tick)

    """
    def stop(self):
        self.code = np.array(
            [[[0 for _ in range(Step_num)] for __ in range(Parts_num-1)] for ___ in range(Agent_num)]
            )
    """

    def superStop(self):
        for agent in self.agent_list:
            agent.stop()

    def save(self, filename):
        add_data = {"parts_num": Operator.parts_num, "step_num": Operator.step_num, "rote_speed": Worm.rote_speed, "torque": Worm.torque, "FPStep": Operator.FPStep}
        add_agentData = [{
            "parts_num": agent.parts_num, "step_num": agent.step_num,
            "rote_speed": agent.rote_speed, "torque": agent.torque,
            "FPStep": agent.FPStep,
        } for agent in self.agent_list]

        self.ga_instance.save(filename, add_data, add_agentData)

    """
    def save_agent(self):
        if not os.path.exists("agent_data"):
            os.mkdir("agent_data")

        for index, agent in enumerate(self.agent_list):
            data = {"parts_num": agent.parts_num, "step_num": agent.step_num,
                    "rote_speed": agent.rote_speed, "torque": agent.torque,
                    "FPStep": agent.FPStep, "genome": agent.genome.tolist()}

            with open("agent_data/"+str(index)+".json", "w") as file:
                json.dump(data, file)
    """

    def reStart(self):
        for index in range(Operator.agent_num):
            for idx_parts in range(Operator.parts_num):
                self.phys_world.DestroyBody(self.agent_list[index].parts_list[idx_parts])

        self.agent_list = []
        self.motor_lastValue = []
        self.idx_finish = []
        self.idx_ovTurned = []
        self.centerX_list_2 = {}
        self.obj_list = self.obj_list[0:- Operator.agent_num * Operator.parts_num]

        self.mk_agent(self.ga_instance.get_genomes())

    def slct_reStart(self):
        self.centerX_list_2.update(self.collect_centerX())
        distance_list = self.distances()
        distance_list = self.penalty(distance_list)
        print("score: ", end="")
        for index, score in enumerate(distance_list):
            print(index, ":", "{:.3g}".format(score), sep="", end=", ")
        print("\n(ranking: ", end="")
        print(np.array(sorted(enumerate(distance_list), key=lambda x: x[1], reverse=True))[:,0], end="")
        print(")\n*****************")

        self.ga_instance.select(distance_list, -1)
        self.save("learn_data.json")
        self.generation += 1
        if self.generation % 10 == 0:
            self.ga_instance.backup()

        self.reStart()
        # self.save_agent()

        self.sec_time = 0
        self.count_sec += 1
        self.tick = 0
        # self.step = 0

        self.set_label()

    def detect_cntct(self):
        for index in range(Operator.agent_num):
            agent = self.agent_list[index]
            contact_list = [contact.other for contact in agent.parts_list[agent.idx_head].contacts]
            if self.wall_list[(index%Limit_agt_x)+1] in contact_list:
                if index not in self.idx_finish:
                    print("finish:", index)
                    self.centerX_list_2[index] = agent.get_centerX()
                    self.idx_finish.append(index)

    def detect_ovTurned(self):
        for index in range(Operator.agent_num):
            if index not in self.idx_ovTurned:
                if self.agent_list[index].overTurned():
                    print("overturned:", index)
                    self.idx_ovTurned.append(index)

    def set_label(self):
        self.txt_list["section"].SetLabel("section: "+str(self.count_sec))
        self.txt_list["generation"].SetLabel("generation: "+str(self.generation))

    def CloseWindow(self, event):
        # 単純な終了処理
        wx.Exit()

    def OnTimer(self, event):
        # 1. ウィンドウをきれいに消去
        if Draw == True:
            self.bdc = wx.BufferedDC(self.cdc, self.bmp)
            self.gcdc = wx.GCDC(self.bdc)
            self.gcdc.Clear()

            self.gcdc.SetPen(wx.Pen('white'))
            self.gcdc.SetBrush(wx.Brush('white'))
            self.gcdc.DrawRectangle(0,0,self.winPx_x,self.winPx_x)

            # 2. 物理オブジェクトの描画
            for body in self.obj_list:  # or: world.bodies
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [(body.transform * v) * PPM for v in shape.vertices]
                    vertices = [(int(v[0]), int(self.winPx_y - v[1])) for v in vertices]

                    self.gcdc.SetPen(wx.Pen(wx.Colour(50,50,50)))
                    self.gcdc.SetBrush(wx.Brush(wx.Colour(colors[body.type])))
                    self.gcdc.DrawPolygon(vertices)

        # 3. 物理シミュレーション1step分 実施
        self.phys_world.Step(0.04, 10, 10)

        if self.demo == False:
            self.detect_cntct()
            self.detect_ovTurned()

            if self.tick == Operator.FPStep:
                self.superCtrl()
                self.tick = 0
            elif self.sec_time == Operator.FPSec - Interbal:
                self.superStop()
            else:
                self.tick += 1

            if self.sec_time == Operator.FPSec:
                if self.count_sec < Section_num:
                    self.slct_reStart()
                    self.view_info()
                else:
                    self.save("learn_data.json")
                    self.CloseWindow(wx.EVT_CLOSE)
            else:
                self.sec_time += 1
        else:
            self.superCtrl()
            self.tick += 1


class Worm:
    box_size = (1, 0.25)
    rote_speed = 1
    torque = 170
    space = Cell_x - Operator.parts_num * 2
    num_call = 0

    def __init__(self, parts_num, step_num, rote_speed, torque, FPStep, position, world, genome):
        offset_x = (Worm.box_size[0] * (parts_num * 2) + Worm.space) * position[1]
        offset_y = 1.5 + Cell_y * position[0]

        # self.idx_posi = math.ceil(parts_num / 2)
        self.idx_posi = parts_num-1
        self.idx_head = parts_num-1
        if parts_num % 2 == 1:
            self.idx_center = math.ceil(parts_num / 2) - 1
        else:
            self.idx_center = math.ceil(parts_num / 2)

        self.parts_num = parts_num
        self.step_num = step_num
        self.rote_speed = rote_speed
        self.torque = torque
        self.FPStep = FPStep
        self.genome = np.array(genome)

        self.step = 0
        self.motor_lastValue = [0 for __ in range(parts_num-1)]

        self.parts_list = []
        self.rj_list = []
        for index in range(parts_num):
            self.parts_list.append(world.CreateDynamicBody(position=(offset_x + Worm.box_size[0] * 2 * (index + 1), offset_y)))
            if index != self.idx_head:
                fixture = fixtureDef(
                    shape=polygonShape(box=Worm.box_size),
                    density=1,
                    friction=0.3,
                    )
            else:
                fixture = fixtureDef(
                    shape=polygonShape(box=Worm.box_size),
                    density=1,
                    friction=1.0,
                    )
                # categoryBits=Worm.colli_filter[-(Agent_num-Worm.num_call)],
                # maskBits=Worm.colli_filter[-(Agent_num-Worm.num_call)]
            # fixture.filter.groupIndex = -1
            self.parts_list[-1].CreateFixture(fixture)
            if index != 0:
                rj = world.CreateRevoluteJoint(
                    bodyA=self.parts_list[-1],
                    bodyB=self.parts_list[-2],
                    anchor=(self.parts_list[-2].worldCenter + self.parts_list[-1].worldCenter) * 0.5,
                    lowerAngle=-2.61,
                    upperAngle=2.61,
                    enableLimit=True,
                    maxMotorTorque = torque,
                    motorSpeed = 0,
                    enableMotor = True
                    )
                self.rj_list.append(rj)
        Worm.num_call += 1

    def control(self, tick):
        if tick % self.FPStep == 0:
            code_clm = self.genome[:,self.step]
            for idx_motor, value in enumerate(code_clm):
                if value != self.motor_lastValue[idx_motor]:
                    self.rote(idx_motor, value)
                    self.motor_lastValue[idx_motor] = value
            if self.step < self.step_num - 1:
                self.step += 1
            else:
                self.step  = 0

    def rote(self, index, direction):
        if direction > 0:
            self.rj_list[index].motorSpeed = self.rote_speed
        elif direction < 0:
            self.rj_list[index].motorSpeed = - self.rote_speed
        elif direction == 0:
            self.rj_list[index].motorSpeed = 0

    def stop(self):
        self.genome = np.array([[0 for _ in range(self.step_num)] for __ in range(self.parts_num-1)])

    def get_centerX(self):
        # print(id(self.parts_list[self.idx_posi]))
        return self.parts_list[self.idx_posi].worldCenter[0]

    def overTurned(self):
        if self.parts_list[self.idx_center].angle < -1.74:
            return True
        else:
            return False



if __name__ == '__main__':
    args = sys.argv

    app = wx.App()
    print("1: new, 2: continue, 3: initial value, 4: remake lower, 5: resize, 6: demonstration")
    reply = input()
    if reply == "1" or reply == "":
        print("new\n")
        w = Operator(title='@tsukuba_takaoka')
    elif reply == "2":
        print("continue\n")
        w = Operator(title='@tsukuba_takaoka', loadLog=True)
    elif reply == "3":
        print("initial value\n")
        w = Operator(title='@tsukuba_takaoka', loadLog=True, initValue=0)
    elif reply == "4":
        print("remake lower\n")
        w = Operator(title='@tsukuba_takaoka', loadLog=True, remakeLower=True)
    elif reply == "5":
        print("resize\n")
        w = Operator(title='@tsukuba_takaoka', loadLog=True, resize=True)
    elif reply == "6":
        print("demonstration\n")
        w = Operator(title='@tsukuba_takaoka', demo=True)
    w.Center()

    if len(args) > 1 and args[1] == "ndraw":
        Draw = False
    if Draw == True:
        w.Show()

    app.MainLoop()
