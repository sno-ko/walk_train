import numpy as np
import random as rd
import math
import statistics
import json
import itertools
import shutil
import os

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class G_algorithm:
    rate_elite = 0.1
    rate_spawnMute = 0.5 # 0.2
    # rate_muteGene = 0.2 # 0.1
    rate_spawnClone = 0 # 0.1

    saveDir = "data"
    saveAgentDir = "agent_data"
    path_agentData = saveDir+"/"+saveAgentDir

    name_saveData = "learn_data.json"

    # path_saveDir = "data/"


    def __init__(self, agent_num=None, num_row=None, num_clm=None, value_ctlg=None, loadLog=False):
        if loadLog == False:
            self.agent_num = agent_num
            self.generation = 0
            self.num_row = num_row
            self.num_clm = num_clm
            self.num_elite = math.ceil(agent_num * G_algorithm.rate_elite)
            self.value_ctlg = value_ctlg
            self.score_log = []
            self.simMean_log = []
            self.simMedi_log = []
            self.simVari_log = []
            self.genera_clone = []
            self.mute_list = [False for _ in range(agent_num)]
            self.loadedData = None
            self.statistics = ["mean", "vari"]
            if (agent_num - self.num_elite) % 2 == 1:
                self.num_elite += 1

            self.population = np.array([self.mk_genome() for _ in range(agent_num)])

            if not os.path.exists(G_algorithm.saveDir):
                os.mkdir(G_algorithm.saveDir)
            if not os.path.exists(G_algorithm.path_agentData):
                os.mkdir(G_algorithm.path_agentData)

            if self.agent_num > 1:
                self.append_cosSim()
        else:
            self.load()
            self.mute_list = [False for _ in range(self.agent_num)]

        self.count_save = 0
        self.count_backup = 0

    def get_genomes(self):
        return self.population

    def mk_genome(self):
        return [rd.choices(self.value_ctlg, k=self.num_clm) for _ in range(self.num_row)]

    def remake_lower(self, initValue=None):
        if initValue == None:
            for index in range(1, self.agent_num):
                self.population[index] = self.mk_genome()
        else:
            base = self.population[initValue].tolist()
            genome_list = [base] + [self.mutation(base) for _ in range(self.agent_num-1)]
            self.population = np.array(genome_list)

    def resize(self, size=None, num=None):
        if num != None:
            size = self.agent_num + num

        if size > self.agent_num:
            self.population = np.concatenate([self.population, [self.mk_genome() for _ in range(size-self.agent_num)]])
        elif size < self.agent_num:
            self.population = self.population[0:size]

        self.agent_num = size
        self.num_elite = math.ceil(self.agent_num * G_algorithm.rate_elite)
        if (self.agent_num - self.num_elite) % 2 == 1:
            self.num_elite += 1
        self.mute_list = [False for _ in range(self.agent_num)]

    def mutation(self, genome):
        rate_muteGene = rd.choice(range(1, 7))
        genome_mute = []
        for gene in genome:
            gene_mute = []
            for geneValue in gene:
                if rd.choice(range(1, 11)) > rate_muteGene:
                    gene_mute.append(geneValue)
                else:
                    gene_mute.append(rd.choice(self.value_ctlg))
            genome_mute.append(gene_mute)
        return genome_mute

    def clossOver(self, idx_agentA, idx_agentB, axis):
        if axis == 1:
            distribution = rd.choices([idx_agentA, idx_agentB], k=self.num_clm)

            genome = np.array([self.population[idx_parent][:,idx_clm] for idx_clm, idx_parent in enumerate(distribution)])
            genome = genome.T

            if rd.choice(range(1, 11)) > G_algorithm.rate_spawnMute * 10:
                self.mute_list.append(False)
            else:
                self.mute_list.append(True)
                genome = self.mutation(genome)

        elif axis == 0:
            distribution = rd.choices([idx_agentA, idx_agentB], k=self.num_row)
            # distri_list = np.array([rd.sample([idx_agentA, idx_agentB], 2) for _ in range(self.num_row)]).T
            # print(distri_list)

            genome = [self.population[idx_parent][idx_gene] for idx_gene, idx_parent in enumerate(distribution)]
            if rd.choice(range(1, 11)) > G_algorithm.rate_spawnMute * 10:
                self.mute_list.append(False)
            else:
                self.mute_list.append(True)
                genome = self.mutation(genome)

        return genome

    def select(self, fitness_list, axis):
        fitness_list_sorted = sorted(enumerate(fitness_list), key=lambda x: x[1], reverse=True)
        self.score_log.append(fitness_list_sorted[0][1])
        self.mute_list = []
        offspring = []
        ranking_idx = np.array(fitness_list_sorted)[:,0]
        idxElite_list = ranking_idx[0:self.num_elite].tolist()
        idxElite_list = list(map(int, idxElite_list))

        offspring += [self.population[index] for index in idxElite_list]
        self.mute_list += [False for _ in range(self.num_elite)]

        if fitness_list_sorted[-1][1] < 0:
            fitness_list = np.array(fitness_list) - fitness_list_sorted[-1][1]

        sum_fit = sum(fitness_list)
        per_list = [value/sum_fit for value in fitness_list]
        idxParent_list = [np.random.choice(self.agent_num, 2, replace=False, p=per_list) for _ in range(self.agent_num - self.num_elite)]
        """
        idxLower_list = ranking_idx[self.num_elite:]
        idxLower_list = list(map(int, idxLower_list))

        idxParent_list = [idxLower_list[index:index + 2] for index in range(0, len(idxLower_list), 2)] # [0:-1]
        # idxParent_list.insert(0, idxElite_list[0:2])
        """

        for idx_parentA, idx_parentB in idxParent_list:
            # print("paretA:", idx_parentA, ", paretB:", idx_parentB)
            children = self.clossOver(idx_parentA, idx_parentB, axis)
            offspring.append(children)

        if rd.choice(range(1, 11)) <= G_algorithm.rate_spawnClone * 10:
            offspring[self.num_elite] = self.mutation(offspring[0])
            self.mute_list[self.num_elite] = "clone"
            self.genera_clone.append(self.generation)

        offspring = np.array(offspring)
        self.population = offspring

        self.generation += 1
        if self.agent_num > 1:
            self.append_cosSim()

    def save(self, filename, add_data=None, add_agentData=None):
        dict = {"agent_num": self.agent_num, "generation": self.generation,
                "num_row": self.num_row, "num_clm": self.num_clm,
                "num_elite": self.num_elite, "value_ctlg": self.value_ctlg,
                "statistics": self.statistics,"score_log": self.score_log,
                "simMean_log": self.simMean_log, "simMedi_log": self.simMedi_log,
                "simVari_log": self.simVari_log, "genera_clone": self.genera_clone}
                # "population": self.population.tolist()}
        if add_data != None:
            dict.update(add_data)

        with open(G_algorithm.saveDir+"/"+filename, "w") as file:
            json.dump(dict, file)

        if add_agentData == None:
            add_agentData = [{} for _ in range(self.agent_num)]

        for index, data in enumerate(add_agentData):
            data.update({"genome": self.population[index].tolist()})
            with open(G_algorithm.path_agentData+"/"+str(index)+".json", "w") as file:
                json.dump(data, file)

        if self.count_save == 0:
            differ_list = set(os.listdir(G_algorithm.path_agentData)) - set([str(index)+".json" for index in range(self.agent_num)])
            for name in differ_list:
                os.remove(G_algorithm.path_agentData+"/"+name)

        self.count_save += 1

    def backup(self):
        path = G_algorithm.saveDir+"_backup"
        if not os.path.exists(path):
            os.mkdir(path)

        dir_name = str(self.count_backup % 2)
        if os.path.exists(path+"/"+dir_name):
            shutil.rmtree(path+"/"+dir_name)

        shutil.copytree(G_algorithm.saveDir, path+"/"+dir_name)
        self.count_backup += 1


    def load(self):
        with open(G_algorithm.saveDir+"/"+G_algorithm.name_saveData, "r") as file:
            dict = json.load(file)
        self.agent_num = dict["agent_num"]
        self.generation = dict["generation"]
        self.num_row = dict["num_row"]
        self.num_clm = dict["num_clm"]
        self.num_elite = dict["num_elite"]
        self.value_ctlg = dict["value_ctlg"]
        if "statistics" in dict:
            self.statistics = dict["statistics"]
        else:
            self.statistics = ["mean", "vari"]
        self.score_log = dict["score_log"]
        self.simMean_log = dict["simMean_log"]
        self.simMedi_log = dict["simMedi_log"]
        self.simVari_log = dict["simVari_log"]
        self.genera_clone = dict["genera_clone"]
        # self.population = np.array(dict["population"])

        genome_list = []
        for index in range(self.agent_num):
            with open(G_algorithm.path_agentData+"/"+str(index)+".json", "r") as file:
                genome_list.append(json.load(file)["genome"])
        self.population = np.array(genome_list)

        dict.update({"population": genome_list})
        self.loadedData = dict

    def similarity(self, *opt):
        output = {}
        flatten = [np.ravel(genome).tolist() for genome in self.population]
        combination = itertools.combinations((index for index in range(self.agent_num)), 2)
        sim_list = [cos_sim(flatten[idx_A], flatten[idx_B]) for idx_A, idx_B in combination]
        if "mean" in opt:
            output["mean"] = sum(sim_list) / len(sim_list)
        if "medi" in opt:
            output["medi"] = np.median(sim_list)
        if "vari" in opt:
            output["vari"] = np.sum((np.array(sim_list) - output["mean"]) ** 2) / len(sim_list)

        return output

    def append_cosSim(self):
        if len(self.simMean_log) == self.generation:
            output = self.similarity(*self.statistics)
            for name in output:
                if name == "mean":
                    self.simMean_log.append(output["mean"])
                elif name == "medi":
                    self.simMedi_log.append(output["medi"])
                elif name == "vari":
                    self.simVari_log.append(output["vari"])
        else:
            print("ERROR: G_algorithm.append_cosSim")

if __name__ == "__main__":
    pass
