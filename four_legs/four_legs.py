# -*- coding: utf-8 -*-
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, fixtureDef)
import wx
import numpy as np
import random as rd
import math
import itertools
import sys
import os
import json
import glob
sys.path.append('../inchworm')
import genetic_algorithm as ga

Draw = True
PPM = 5.0  # pixels per meter
FPS = 60
PartsSize_default = (1, 0.25)
PartsNum_default = 6

Loop_num = 150

Limit_visible = 16
Limit_agt_clm = 3
Cell_x = 60
Cell_y = 10

Interbal = 50

colors = {
    staticBody: (150,150,150),
    dynamicBody: (112,146,190),
}


class FourLegs:
    parts_num = 9 # 6
    gn_rowSize = 4
    step_num = 7 # 9

    body_size = (3, 1)
    leg_size = (0.5, 0.9)

    # parts_size = (1, 0.25)
    # parts_size = (1*(PartsNum_default/parts_num), 0.25)
    rote_speed = 1.1# 1
    torque = 210 # 170
    FPStep = 17

    # space = Cell_x - parts_num * 2
    # space = Cell_x - parts_num * (2 * (PartsNum_default/parts_num))
    firstStep = True
    # friction_list = [0.3 for _ in range(parts_num)]
    foot_friction = 0.5
    # friction_list[-1] = 1.5
    # friction_list[0] = 0.9
    limit_angle = [0.9, 1.39] # 90-1.57, 50-0.9
    num_call = 0

    def __init__(
            self, position, world, genome,
            parts_num=parts_num, step_num=step_num, rote_speed=rote_speed,
            torque=torque, FPStep=FPStep, firstStep=firstStep,
            foot_friction=foot_friction, parts_size=(body_size, leg_size), limit_angle=limit_angle):
        self.world = world
        self.gn_rowSize = FourLegs.gn_rowSize

        self.id = FourLegs.num_call

        self.parts_size = parts_size
        self.body_size = parts_size[0]
        self.leg_size = parts_size[1]
        self.offset_x = Operator.wall_thick + Cell_x * position[1] + self.body_size[0] + 3
        self.offset_y = Operator.wall_thick + Cell_y * position[0] + self.leg_size[1] * 4 + self.body_size[1]

        self.idx_posi = None

        """
        self.usedParts_turned = ["center", "tail"]
        self.idx_parts = {"head": parts_num-1, "center": parts_num-1, "tail": 0}
        self.angle_turned = {"head": None, "center": -2.61, "tail": 2.79}
        """
        self.idx_head = None
        self.idx_center = None
        """
        if parts_num % 2 == 1:
            self.idx_center = math.ceil(parts_num / 2) - 1
        else:
            self.idx_center = math.ceil(parts_num / 2)
        """
        self.parts_num = parts_num
        self.step_num = step_num
        self.rote_speed = rote_speed
        self.torque = torque
        self.FPStep = FPStep
        self.firstStep = firstStep
        self.foot_friction = foot_friction
        # self.friction_list = friction_list
        self.limit_angle = limit_angle

        half = math.floor(self.step_num/2)
        genome_L_first = np.zeros((4, half))
        genome_L_latter = genome[:, :self.step_num-half]
        genome_L = np.concatenate([genome_L_first, genome_L_latter], 1)

        self.genome_R = genome
        self.genome = np.concatenate([genome_L, genome])[[0, 1, 4, 5, 2, 3, 6, 7], :]
        self.step = 0
        self.motor_lastValue = np.zeros(parts_num-1)

        self.cycle = 0

        self.parts_list = []
        self.rj_list = []

        self.idxUpperLeg_list = []
        self.body_index = None

        for index in range(2):
            if index == 1:
                self.mk_body(world)
            for position in [i * (self.body_size[0]-self.leg_size[0]) for i in [-1, 1]]:
                # 後ろ、前、後ろ、前
                self.mk_leg(position, world)

        for index in self.idxUpperLeg_list:
            rj = world.CreateRevoluteJoint(
                bodyA=self.parts_list[index],
                bodyB=self.parts_list[self.body_index],
                anchor=self.parts_list[index].worldCenter+(0, self.leg_size[1]),
                lowerAngle=-self.limit_angle[0], # -2.44,
                upperAngle=self.limit_angle[0],# 2.44,
                enableLimit=True,
                maxMotorTorque = self.torque,
                motorSpeed = 0,
                enableMotor = True
                )
            self.rj_list.append(rj)
        len_rj = len(self.rj_list)

        """
        for index in range(parts_num):
            self.parts_list.append(world.CreateDynamicBody(position=(self.offset_x + self.parts_size[0] * 2 * (index + 1), self.offset_y)))
            fixture = fixtureDef(
                shape=polygonShape(box=self.parts_size),
                density=1,
                friction=friction_list[index],
                )
                # categoryBits=FourLegs.colli_filter[-(Agent_num-FourLegs.num_call)],
                # maskBits=FourLegs.colli_filter[-(Agent_num-FourLegs.num_call)]
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
        """
        self.idx_posi = self.body_index
        self.idx_head = self.body_index
        self.idx_center = self.body_index

        FourLegs.num_call += 1

    def mk_leg(self, position, world):
        fixture = fixtureDef(
        shape=polygonShape(box=self.leg_size),
        density=1,
        friction=0.3,
        )
        if position >= 0:
            fixture.filter.groupIndex = -2
        else:
            fixture.filter.groupIndex = -3

        self.idxUpperLeg_list.append(len(self.parts_list))
        self.parts_list.append(world.CreateDynamicBody(position=(self.offset_x+position, self.offset_y - (self.leg_size[1]+self.body_size[1]*0.5)) ))
        self.parts_list[-1].CreateFixture(fixture)

        self.parts_list.append(world.CreateDynamicBody(position=(self.offset_x+position, self.parts_list[-1].worldCenter[1] - self.leg_size[1]*2)))
        fixture.friction=self.foot_friction
        self.parts_list[-1].CreateFixture(fixture)

        rj = world.CreateRevoluteJoint(
            bodyA=self.parts_list[-1],
            bodyB=self.parts_list[-2],
            anchor=(self.parts_list[-2].worldCenter + self.parts_list[-1].worldCenter) * 0.5,
            lowerAngle=0,
            upperAngle=self.limit_angle[1], # 2.96,
            enableLimit=True,
            maxMotorTorque = self.torque,
            motorSpeed = 0,
            enableMotor = True
            )
        self.rj_list.append(rj)

    def mk_body(self, world):
        self.body_index = len(self.parts_list)
        self.parts_list.append(world.CreateDynamicBody(position=(self.offset_x, self.offset_y)))
        fixture = fixtureDef(
            shape=polygonShape(box=self.body_size),
            density=1,
            friction=0.3,
            )
        fixture.filter.groupIndex = -1
        self.parts_list[-1].CreateFixture(fixture)

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
                if self.firstStep == False:
                    self.step  = 0
                else:
                    self.step  = 1
                    if self.cycle == 0:
                        half = math.floor(self.step_num/2)+1
                        index_list_latter = list(range(1, half))
                        index_list_first = list(range(half, self.step_num))
                        index_list = [0]+index_list_first+index_list_latter
                        genome_L = self.genome_R[:, index_list]
                        self.genome = np.concatenate([genome_L, self.genome_R])[[0, 1, 4, 5, 2, 3, 6, 7], :]
                    self.cycle += 1

    def rote(self, index, direction):
        if direction > 0:
            self.rj_list[index].motorSpeed = self.rote_speed
        elif direction < 0:
            self.rj_list[index].motorSpeed = - self.rote_speed
        elif direction == 0:
            self.rj_list[index].motorSpeed = 0

    def stop(self):
        self.genome = np.zeros_like(self.genome)

    def get_centerX(self):
        # print(id(self.parts_list[self.idx_posi]))
        return self.parts_list[self.idx_posi].worldCenter[0]

    def get_centerY(self):
        # print(id(self.parts_list[self.idx_posi]))
        return self.parts_list[self.idx_posi].worldCenter[1]

    def overTurned(self):
        if self.parts_list[self.idx_center].angle < -2.61: # or self.parts_list[0].angle > 2.79:
            return True
        else:
            return False


class Operator(wx.Frame):
    agent_num = 50# 21
    cycle_perLoop = 5 # 3
    step_clossCycl = FourLegs.FPStep * FourLegs.step_num
    FPLoop = step_clossCycl * cycle_perLoop + Interbal

    key_dict = {"1": [0, 1], "2": [1, 1], "3": [2, 1], "4": [3, 1], "5": [4, 1], "6": [5, 1],
                "!": [0, -1], "\"": [1, -1], "#": [2, -1], "$": [3, -1], "%": [4, -1], "&": [5, -1]}
    value_ctlg = [-1, 0, 1]

    wall_thick = 0.5

    span_collect_Y = 40

    def __init__(self, parent=None, id=-1, title=None, loadLog=False, initValue=None, remakeLower=False, resize=False, test=False):
        if test == False:
            if loadLog == False:
                self.ga_instance = ga.G_algorithm(Operator.agent_num, FourLegs.gn_rowSize, FourLegs.step_num, Operator.value_ctlg)
                self.generation = 0
            else:
                self.ga_instance = ga.G_algorithm(loadLog=True)
                self.generation = self.ga_instance.generation
                FourLegs.rote_speed = self.ga_instance.loadedData["rote_speed"]
                FourLegs.torque = self.ga_instance.loadedData["torque"]

                FourLegs.parts_num = self.ga_instance.loadedData["parts_num"]
                FourLegs.step_num = self.ga_instance.loadedData["step_num"]
                FourLegs.FPStep = self.ga_instance.loadedData["FPStep"]

                if initValue != None:
                    self.ga_instance.remake_lower(initValue=0)
                elif remakeLower == True:
                    self.ga_instance.remake_lower()

                if resize == True:
                    self.ga_instance.resize(Operator.agent_num)
        else:
            json_list = glob.glob(ga.G_algorithm.path_agentData+"/*.json")
            Operator.agent_num = len(json_list)

        self.test = test

        if Operator.agent_num > Limit_visible:
            self.agent_num_vs = Limit_visible
        else:
            self.agent_num_vs = Operator.agent_num

        self.numCell_row = math.ceil(Operator.agent_num / Limit_agt_clm)
        self.numCell_row_vs = math.ceil(self.agent_num_vs / Limit_agt_clm)
        if Operator.agent_num < Limit_agt_clm:
            self.numCell_clm = Operator.agent_num
        else:
            self.numCell_clm = Limit_agt_clm
        if self.agent_num_vs < Limit_agt_clm:
            self.numCell_clm_vs = self.agent_num_vs
        else:
            self.numCell_clm_vs = Limit_agt_clm

        self.winPx_x = Cell_x * self.numCell_clm_vs * PPM + Operator.wall_thick * PPM * 2
        self.winPx_y = Cell_y * self.numCell_row_vs * PPM + Operator.wall_thick * PPM * 2

        # self.winPx_x = 1000
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

        self.obj_list = []
        self.ground_list = []
        self.wall_list = []
        self.mk_ground()

        self.agent_list = []
        if test == False:
            self.mk_agent(self.ga_instance.get_genomes())
        else:
            self.mk_agent(load=True)

        self.num_obj_vs = len(self.ground_list)+len(self.wall_list)+sum(agent.parts_num for agent in self.agent_list[0:self.agent_num_vs])
        self.obj_list_vs = self.obj_list[0:self.num_obj_vs]

        self.idx_finish = set()
        self.idx_ovTurned = set()

        self.centerX_list_1 = self.collect_centerX()
        self.centerX_list_2 = {}

        self.centerY_list = []

        # 4. タイマーループ開始
        self.timer.Start(1000/FPS)

        if test == False:
            self.tick = 0
        else:
            self.tick = 1
        # self.step = 0
        self.loop_time = 0
        self.count_loop = 1

        self.txt_list = {}
        if test == False:
            self.txt_list["loop"] = wx.StaticText(self.panel, pos=(30, 6))
            self.txt_list["generation"] = wx.StaticText(self.panel, pos=(90, 6))
            self.set_label()
        else:
            self.txt_list["loop"] = wx.StaticText(self.panel, pos=(30, 6))
            self.txt_list["loop"].SetLabel("test")

        self.layout_index = wx.GridSizer(rows=self.numCell_row_vs, cols=self.numCell_clm_vs, gap=(0, 0))
        index_list = range(self.numCell_clm_vs*self.numCell_row_vs)
        index_list = reversed([index_list[index:index + self.numCell_clm_vs] for index in range(0, len(index_list), self.numCell_clm_vs)])
        for list_in in index_list:
            for index in list_in:
                self.layout_index.Add(wx.StaticText(self.panel, label=str(index)),flag=wx.TOP|wx.LEFT, border=5)
        self.panel.SetSizer(self.layout_index)

        if test == False:
            print("*****************")
            self.view_info()

    def view_info(self):
        print("loop:", self.count_loop, " generation:", self.generation)
        """
        for index in range(Operator.agent_num):
            if self.ga_instance.mute_list[index] == False:
                print(index)
            elif self.ga_instance.mute_list[index] == True:
                print(index, "(mutation)")
            elif self.ga_instance.mute_list[index] == "clone":
                print(index, "(clone)")
            print(self.agent_list[index].genome, "\n")
        """

    def collect_centerX(self):
        centerX_list = {}
        for index in range(Operator.agent_num):
            if index not in self.idx_finish:
                centerX_list[index] = self.agent_list[index].get_centerX()
        return centerX_list

    def collect_centerY(self):
        self.centerY_list.append([agent.get_centerY() for agent in self.agent_list])
        # print(self.centerY_list)

    def distances(self):
        list_2 = [self.centerX_list_2[key] for key in sorted(self.centerX_list_2)]
        list_1 = list(self.centerX_list_1.values())
        return [x2-x1 for x1, x2 in zip(list_1, list_2)]

    def centerY_average(self):
        gound_Y = [[ground.worldCenter[1]]*Limit_agt_clm for ground in self.ground_list]
        gound_Y = np.array(sum(gound_Y, [])[0:Operator.agent_num])
        return np.mean(self.centerY_list, axis=0) - gound_Y

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
                position=(Cell_x * self.numCell_clm * Operator.wall_thick, Operator.wall_thick + Cell_y * index)
            ))
            fixture = fixtureDef(
                shape=polygonShape(box=(Cell_x * self.numCell_clm * Operator.wall_thick, Operator.wall_thick))
            )
            # fixture.filter.groupIndex = -2
            self.ground_list[-1].CreateFixture(fixture)
            self.obj_list.append(self.ground_list[-1])

        for index in range(self.numCell_clm+1):
            self.wall_list.append(self.phys_world.CreateStaticBody(
                position=(Cell_x * index, Cell_y * self.numCell_row * Operator.wall_thick),
                shapes=polygonShape(box=(Operator.wall_thick, Cell_y * self.numCell_row * Operator.wall_thick)),
            ))
            self.obj_list.append(self.wall_list[-1])

    def mk_agent(self, genome_list=None, load=False):
        posi_list = list(itertools.product(range(self.numCell_row), range(self.numCell_clm)))[0:Operator.agent_num]
        if load == False:
            for position, genome in zip(posi_list, genome_list):
                self.agent_list.append(FourLegs(position, self.phys_world, genome))
            self.obj_list += sum([agent.parts_list for agent in self.agent_list], [])
            # self.motor_lastValue = [[0 for __ in range(Parts_num-1)] for _ in range(Agent_num)]
        else:
            name_list = os.listdir(ga.G_algorithm.path_agentData)
            # name_list = sorted(name_list, key=lambda name: int(name.split(".")[0]))

            count = 0
            for name, position in zip(name_list, posi_list):
                print("["+str(count)+"] \"", name, "\"", sep="")
                count += 1
                with open(ga.G_algorithm.path_agentData+"/"+name, "r") as file:
                    dict = json.load(file)
                dict["genome"] = np.array(dict["genome"])
                dict["world"] = self.phys_world
                dict["position"] = position
                self.agent_list.append(FourLegs(**dict))
            self.obj_list += sum([agent.parts_list for agent in self.agent_list], [])

    def superCtrl(self):
        for agent in self.agent_list:
            agent.control(self.tick)

    def superStop(self):
        for agent in self.agent_list:
            agent.stop()

    def save(self, filename):
        add_data = {"parts_num": FourLegs.parts_num, "step_num": FourLegs.step_num, "rote_speed": FourLegs.rote_speed, "torque": FourLegs.torque, "FPStep": FourLegs.FPStep}
        add_agentData = [{
            "parts_num": agent.parts_num, "step_num": agent.step_num,
            "rote_speed": agent.rote_speed, "torque": agent.torque,
            "FPStep": agent.FPStep, "firstStep": agent.firstStep,
            "foot_friction": agent.foot_friction, "parts_size": agent.parts_size, "limit_angle": agent.limit_angle
        } for agent in self.agent_list]

        self.ga_instance.save(filename, add_data, add_agentData)

    def reStart(self):
        for index in range(Operator.agent_num):
            for idx_parts in range(FourLegs.parts_num):
                self.phys_world.DestroyBody(self.agent_list[index].parts_list[idx_parts])

        self.agent_list = []
        self.motor_lastValue = []
        self.idx_finish = set()
        self.idx_ovTurned = set()
        self.centerX_list_2 = {}
        self.centerY_list = []
        self.obj_list = self.obj_list[0:- Operator.agent_num * FourLegs.parts_num]

        self.mk_agent(self.ga_instance.get_genomes())
        self.obj_list_vs = self.obj_list[0:self.num_obj_vs]

    def slct_reStart(self):
        self.centerX_list_2.update(self.collect_centerX())
        distance_list = self.distances()
        centerYmean_list = self.centerY_average()
        distance_list = np.array(self.penalty(distance_list))

        score_list = distance_list*0.5 + centerYmean_list*2

        """
        print("score: ", end="")
        for index, score in enumerate(distance_list):
            print(index, ":", "{:.3g}".format(score), sep="", end=", ")
        print("\n(ranking: ", end="")
        print(np.array(sorted(enumerate(distance_list), key=lambda x: x[1], reverse=True))[:,0], end="")
        print(")\n*****************")
        """

        self.ga_instance.select(score_list, 0)
        self.save(ga.G_algorithm.name_saveData)
        self.generation += 1
        if self.generation % 10 == 0:
            self.ga_instance.backup()

        self.reStart()
        # self.save_agent()

        self.loop_time = 0
        self.count_loop += 1
        self.tick = 0
        # self.step = 0

        self.set_label()

    def detect_cntct(self):
        for index, agent in enumerate(self.agent_list):
            contact_list = {contact.other for contact in agent.parts_list[agent.idx_head].contacts}
            if self.wall_list[(index%Limit_agt_clm)+1] in contact_list:
                if index not in self.idx_finish:
                    print("finish:", index)
                    self.centerX_list_2[index] = agent.get_centerX()
                    self.idx_finish.add(index)

    def detect_ovTurned(self):
        for index in range(Operator.agent_num):
            if index not in self.idx_ovTurned:
                if self.agent_list[index].overTurned():
                    print("overturned:", index)
                    self.idx_ovTurned.add(index)

    def set_label(self):
        self.txt_list["loop"].SetLabel("loop: "+str(self.count_loop))
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
            self.gcdc.DrawRectangle(0, 0, self.winPx_x, self.winPx_y)


            # 2. 物理オブジェクトの描画
            for body in self.obj_list_vs:  # or: world.bodies
                self.gcdc.SetPen(wx.Pen(wx.Colour(50,50,50)))
                self.gcdc.SetBrush(wx.Brush(wx.Colour(colors[body.type])))
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [(body.transform * v + (Operator.wall_thick, 0)) * PPM for v in shape.vertices]
                    vertices = [(int(v[0]), int(self.winPx_y - v[1])) for v in vertices]

                    # self.gcdc.SetPen(wx.Pen(wx.Colour(50,50,50)))
                    # self.gcdc.SetBrush(wx.Brush(wx.Colour(colors[body.type])))
                    self.gcdc.DrawPolygon(vertices)

        # 3. 物理シミュレーション1step分 実施
        self.phys_world.Step(0.04, 10, 10)

        if self.test == False:
            self.detect_cntct()
            # self.detect_ovTurned()

            if self.tick == FourLegs.FPStep:
                self.superCtrl()
                self.tick = 0
            elif self.loop_time == Operator.FPLoop - Interbal:
                self.superStop()
            else:
                self.tick += 1

            if self.loop_time % Operator.span_collect_Y == 0:
                self.collect_centerY()

            if self.loop_time == Operator.FPLoop:
                if self.count_loop < Loop_num:
                    self.slct_reStart()
                    self.view_info()
                else:
                    self.save(ga.G_algorithm.name_saveData)
                    self.CloseWindow(wx.EVT_CLOSE)
            else:
                self.loop_time += 1
        else:
            self.superCtrl()
            self.tick += 1


if __name__ == '__main__':
    args = sys.argv

    app = wx.App()
    print("1: new, 2: continue, 3: initialize, 4: remake lower, 5: resize, 6: test")
    reply = input()
    if reply == "1" or reply == "":
        print("new\n")
        w = Operator(title='four_legs.py')
    elif reply == "2":
        print("continue\n")
        w = Operator(title='four_legs.py', loadLog=True)
    elif reply == "3":
        print("initial value\n")
        w = Operator(title='four_legs.py', loadLog=True, initValue=0)
    elif reply == "4":
        print("remake lower\n")
        w = Operator(title='four_legs.py', loadLog=True, remakeLower=True)
    elif reply == "5":
        print("resize\n")
        w = Operator(title='four_legs.py', loadLog=True, resize=True)
    elif reply == "6":
        print("test\n")
        w = Operator(title='four_legs.py', test=True)
    w.Center()

    if len(args) > 1 and args[1] == "ndraw":
        Draw = False
    if Draw == True:
        w.Show()

    app.MainLoop()
