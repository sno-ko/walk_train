import numpy as np
import random as rd
import math
import json
import itertools
import shutil
import os

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class G_algorithm:
    rate_elite = 0.1
    rate_spawnMute = 0.2
    rate_muteGene = 0.1
    rate_spawnClone = 0.1

    save_path = "data/"

    def __init__(self, agent_num=None, num_row=None, num_clm=None, value_ctlg=None, loadLog=False):
        if loadLog == False:
            self.agent_num = agent_num
            self.generation = 0
            self.num_row = num_row
            self.num_clm = num_clm
            self.num_elite = math.ceil(agent_num * G_algorithm.rate_elite)
            self.value_ctlg = value_ctlg
            self.score_log = []
            self.cosSim_ave = None
            self.cosSim_vari = None
            self.simAve_log = []
            self.simVari_log = []
            self.genera_clone = []
            self.mute_list = [False for _ in range(agent_num)]
            self.loadedData = None
            if (agent_num - self.num_elite) % 2 == 1:
                self.num_elite += 1

            self.population = np.array([self.mk_genome() for _ in range(agent_num)])

            if not os.path.exists(G_algorithm.save_path[0:-1]):
                os.mkdir(G_algorithm.save_path[0:-1])
            if not os.path.exists(G_algorithm.save_path+"agent_data"):
                os.mkdir(G_algorithm.save_path+"agent_data")

        else:
            self.load()
            self.cosSim_ave = None
            self.cosSim_vari = None
            self.mute_list = [False for _ in range(self.agent_num)]
        self.append_cosSim()

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
        genome_mute = []
        for gene in genome:
            gene_mute = []
            for geneValue in gene:
                if rd.choice(range(1, 10)) > G_algorithm.rate_muteGene * 10:
                    gene_mute.append(geneValue)
                else:
                    gene_mute.append(rd.choice(self.value_ctlg))
            genome_mute.append(gene_mute)
        return genome_mute

    def clossOver(self, idx_agentA, idx_agentB, axis):
        if axis == -1:
            distri_list = np.array([rd.sample([idx_agentA, idx_agentB], 2) for _ in range(self.num_clm)]).T
            genome_A, genome_B = [], []

            genome_list = []
            for distribution in distri_list:
                genome = np.array([self.population[idx_parent][:,idx_clm] for idx_clm, idx_parent in enumerate(distribution)])
                genome_list.append(genome.T)

                if rd.choice(range(1, 10)) > G_algorithm.rate_spawnMute * 10:
                    self.mute_list.append(False)
                else:
                    self.mute_list.append(True)
                    genome_list[-1] = self.mutation(genome_list[-1])

        elif axis == 1:
            distri_list = np.array([rd.sample([idx_agentA, idx_agentB], 2) for _ in range(self.num_row)]).T
            genome_A, genome_B = [], []

            genome_list = []
            for distribution in distri_list:
                genome_list.append([self.population[idx_parent][idx_gene] for idx_gene, idx_parent in enumerate(distribution)])

                if rd.choice(range(1, 10)) > G_algorithm.rate_spawnMute * 10:
                    self.mute_list.append(False)
                else:
                    self.mute_list.append(True)
                    genome_list[-1] = self.mutation(genome_list[-1])

        return genome_list

    def select(self, fitness_list, axis):
        self.score_log.append(fitness_list[0])
        self.mute_list = []
        offspring = []
        ranking_idx = np.array(sorted(enumerate(fitness_list), key=lambda x: x[1], reverse=True))[:,0]
        idxElite_list = ranking_idx[0:self.num_elite].tolist()
        idxElite_list = list(map(int, idxElite_list))

        offspring += [self.population[index] for index in idxElite_list]
        self.mute_list += [False for _ in range(self.num_elite)]

        idxLower_list = ranking_idx[self.num_elite:]
        idxLower_list = list(map(int, idxLower_list))

        idxParent_list = [idxLower_list[index:index + 2] for index in range(0, len(idxLower_list), 2)] # [0:-1]
        # idxParent_list.insert(0, idxElite_list[0:2])

        for idx_parentA, idx_parentB in idxParent_list:
            # print("paretA:", idx_parentA, ", paretB:", idx_parentB)
            children = self.clossOver(idx_parentA, idx_parentB, axis)
            offspring += children

        if rd.choice(range(1, 10)) <= G_algorithm.rate_spawnClone * 10:
            offspring[self.num_elite] = self.mutation(offspring[0])
            self.mute_list[self.num_elite] = "clone"
            self.genera_clone.append(self.generation)

        offspring = np.array(offspring)
        self.population = offspring

        self.generation += 1
        self.append_cosSim()

    def save(self, filename, add_data=None, add_agentData=None):
        dict = {"agent_num": self.agent_num, "generation": self.generation,
                "num_row": self.num_row, "num_clm": self.num_clm,
                "num_elite": self.num_elite, "value_ctlg": self.value_ctlg,
                "score_log": self.score_log, "simAve_log": self.simAve_log,
                "simVari_log": self.simVari_log, "genera_clone": self.genera_clone}
                # "population": self.population.tolist()}
        if add_data != None:
            dict.update(add_data)

        with open(G_algorithm.save_path+filename, "w") as file:
            json.dump(dict, file)

        if add_agentData == None:
            add_agentData = [{} for _ in range(self.agent_num)]

        for index, data in enumerate(add_agentData):
            data.update({"genome": self.population[index].tolist()})
            with open(G_algorithm.save_path+"agent_data/"+str(index)+".json", "w") as file:
                json.dump(data, file)

    def backup(self):
        path = G_algorithm.save_path[0:-1]+"_backup"
        if not os.path.exists(path):
            os.mkdir(path)

        folder_name = len(os.listdir(path))

        shutil.copytree(G_algorithm.save_path[0:-1], path+"/"+str(folder_name))

    def load(self):
        with open(G_algorithm.save_path+"learn_data.json", "r") as file:
            dict = json.load(file)
        self.agent_num = dict["agent_num"]
        self.generation = dict["generation"]
        self.num_row = dict["num_row"]
        self.num_clm = dict["num_clm"]
        self.num_elite = dict["num_elite"]
        self.value_ctlg = dict["value_ctlg"]
        self.score_log = dict["score_log"]
        self.simAve_log = dict["simAve_log"]
        self.simVari_log = dict["simVari_log"]
        self.genera_clone = dict["genera_clone"]
        # self.population = np.array(dict["population"])

        genome_list = []
        for index in range(self.agent_num):
            with open(G_algorithm.save_path+"agent_data/"+str(index)+".json", "r") as file:
                genome_list.append(json.load(file)["genome"])
        self.population = np.array(genome_list)

        dict.update({"population": genome_list})
        self.loadedData = dict

    def similarity(self):
        flatten = [np.ravel(genome).tolist() for genome in self.population]
        combination = itertools.combinations((index for index in range(self.agent_num)), 2)
        sim_list = [cos_sim(flatten[idx_A], flatten[idx_B]) for idx_A, idx_B in combination]

        ave = sum(sim_list) / len(sim_list)
        vari = np.sum((np.array(sim_list) - ave) ** 2) / len(sim_list)
        return {"ave": ave, "vari": vari}

        """
        self.cosSim_ave = sum(sim_list) / len(sim_list)
        self.cosSim_vari= np.sum((np.array(sim_list) - self.cosSim_ave) ** 2) / len(sim_list)
        """

    def append_cosSim(self):
        if len(self.simAve_log) == self.generation:
            result = self.similarity()
            self.cosSim_ave, self.cosSim_vari = result["ave"], result["vari"]
            self.simAve_log.append(self.cosSim_ave)
            self.simVari_log.append(self.cosSim_vari)
        else:
            print("ERROR: G_algorithm.append_cosSim")

if __name__ == "__main__":
    pass
