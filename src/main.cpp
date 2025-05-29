#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <ctime>
#include <iostream>
#include <numeric>
#include <cmath>
#include <chrono>
#include <tuple>
#include <set>
#include <unordered_set>
#include <climits>

#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct Config {
    double mutationProbability;
    int populationsSize;
    int chromosomesPreservedPercentage;
    double crossoverRatio;
    int generations;
    double mutationPressure;
    string dataFile;
};

Config loadConfig(const string& configFile) {
    ifstream input(configFile);
    if (!input.is_open()) {
        cerr << "Could not open config file: " << configFile << endl;
        exit(1);
    }

    json j;
    input >> j;

    Config config;
    config.mutationProbability = j.value("mutationProbability", 0.35);
    config.populationsSize = j.value("populationsSize", 50);
    config.chromosomesPreservedPercentage = j.value("chromosomesPreservedPercentage", 5);
    config.crossoverRatio = j.value("crossoverRatio", 0.5);
    config.generations = j.value("generations", 50000);
    config.mutationPressure = j.value("mutationPressure", 0.15);
    config.dataFile = j.value("dataFile", "../data/data.txt");

    return config;
}

struct Gene {
    int task;
    int machine;

    Gene(int t, int m) : task(t), machine(m) {}
};

struct BestChromosome {
    vector<Gene> chromosome;
    int fitness;
    int generation;
};

pair<int, vector<int>> parseData(const string& filename);
vector<Gene> greedy(int numMachines, vector<int>& taskDurations);
pair<vector<vector<Gene>>, vector<int>> initialGeneration(vector<int> taskDurations, int populationsSize, int numMachines);
pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness);

int fitnessCalculation(int machines, const vector<Gene>& chromosome, const vector<int>& taskDurations);

void mutation(double mutationProbability, vector<Gene>& chromosome, int numMachines, int mutationRange, const vector<int>& taskDurations, double pressure, mt19937& gen);
pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion, const vector<int>& taskDurations, mt19937& gen);
pair<vector<vector<Gene>>, vector<int>> evolution(vector<vector<Gene>>& chromosomes, vector<int>& fitness, double mutationProbability, int chromosomesPreserved, int maxNewChromosomes, int numMachines, vector<int>& taskDurations, double crossoverRatio, double pressure, mt19937& gen);


void mutation(double mutationProbability, vector<Gene>& chromosome, int numMachines, int mutationRange, const vector<int>& taskDurations, double pressure, mt19937& gen) {
    vector<int> machineLoads(numMachines, 0);
    for (const auto& gene : chromosome) {
        machineLoads[gene.machine] += taskDurations[gene.task];
    }

    int currentCmax = *max_element(machineLoads.begin(), machineLoads.end());

    uniform_real_distribution<> probDist(0.0, 1.0);
    for (auto& gene : chromosome) {
        double criticality = (double)machineLoads[gene.machine] / currentCmax;
        double mutationProb = mutationProbability * (1.0 + pressure * criticality);

        if (probDist(gen) < mutationProb) {
            int oldMachine = gene.machine;
            int oldLoad = machineLoads[oldMachine];

            int bestMachine = -1;
            int minLoad = INT_MAX;

            for (int m = 0; m < numMachines; ++m) {
                if (m != oldMachine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }

            if (bestMachine != -1) {
                int newLoad = machineLoads[bestMachine];

                int updatedOldLoad = oldLoad - taskDurations[gene.task];
                int updatedNewLoad = newLoad + taskDurations[gene.task];

                int localOldMax = max({machineLoads[oldMachine], machineLoads[bestMachine]});
                int localNewMax = max({updatedOldLoad, updatedNewLoad});

                if (localNewMax <= localOldMax) {
                    machineLoads[oldMachine] = updatedOldLoad;
                    machineLoads[bestMachine] = updatedNewLoad;
                    gene.machine = bestMachine;
                }
            }
        }
    }
}

vector<Gene> greedy(int numMachines, vector<int>& taskDurations) {
    vector<int> taskOrder(taskDurations.size());
    vector<int> machinesLoad(numMachines, 0); 
    vector<Gene> chromosome;

    iota(taskOrder.begin(), taskOrder.end(), 0);

    sort(taskOrder.begin(), taskOrder.end(),
        [&](int a, int b) {
            return taskDurations[a] > taskDurations[b];
        }
    );

    for (int task : taskOrder) {
        int minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back(Gene{task, minMachine});
        machinesLoad[minMachine] += taskDurations[task];
    }

    return chromosome;
}

int fitnessCalculation(int machines, const vector<Gene>& chromosome, const vector<int>& taskDurations) {
    vector<int> timesList(machines, 0);

    for (const auto& gene : chromosome) {
        if (gene.machine < 0 || gene.machine >= machines) {
            cerr << "Invalid machine number: " << gene.machine << endl;
            exit(1);
        }
        if (gene.task < 0 || gene.task >= taskDurations.size()) {
            cerr << "Invalid task number: " << gene.task << endl;
            exit(1);
        }
        timesList[gene.machine] += taskDurations[gene.task];
    }

    return *max_element(timesList.begin(), timesList.end());
}

pair<int, vector<int>> parseData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int numMachines, task_count;
    file >> numMachines >> task_count;

    vector<int> taskDurations(task_count);
    for (int i = 0; i < task_count; ++i) {
        if (!(file >> taskDurations[i])) {
            cerr << "Error reading task data" << endl;
            exit(1);
        }
    }
    return {numMachines, taskDurations};
}

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion, const vector<int>& taskDurations, mt19937& gen) {
    if (chromosome1.size() != chromosome2.size() || chromosome1.size() != taskDurations.size()) {
        cerr << "Error: Chromosome size mismatch in crossover" << endl;
        exit(1);
    }

    proportion = max(0.1, min(0.9, proportion));
    int splitPoint = static_cast<int>(chromosome1.size() * proportion);
    
    splitPoint = max(1, min(static_cast<int>(chromosome1.size()) - 1, splitPoint));

    vector<Gene> child1(chromosome1.begin(), chromosome1.begin() + splitPoint);
    vector<Gene> child2(chromosome2.begin(), chromosome2.begin() + splitPoint);
    
    unordered_set<int> child1Tasks;
    unordered_set<int> child2Tasks;
    
    for (const auto& gene : child1) {
        child1Tasks.insert(gene.task);
    }
    for (const auto& gene : child2) {
        child2Tasks.insert(gene.task);
    }

    for (const auto& gene : chromosome2) {
        if (child1Tasks.find(gene.task) == child1Tasks.end()) {
            child1.push_back(gene);
            child1Tasks.insert(gene.task);
        }
    }

    for (const auto& gene : chromosome1) {
        if (child2Tasks.find(gene.task) == child2Tasks.end()) {
            child2.push_back(gene);
            child2Tasks.insert(gene.task);
        }
    }

    if (child1.size() != taskDurations.size() || child2.size() != taskDurations.size()) {
        cerr << "Error: Crossover produced invalid chromosome size" << endl;
        cerr << "Child1 size: " << child1.size() << " Child2 size: " << child2.size() << endl;
        exit(1);
    }

    return {child1, child2};
}

pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness) {
    vector<pair<vector<Gene>, int>> zipped;
    for (size_t i = 0; i < chromosomes.size(); ++i) {
        zipped.emplace_back(chromosomes[i], fitness[i]);
    }

    sort(zipped.begin(), zipped.end(), 
        [](const pair<vector<Gene>, int>& a, const pair<vector<Gene>, int>& b) {
            return a.second < b.second;
        }
    );

    for (size_t i = 0; i < zipped.size(); ++i) {
        chromosomes[i] = zipped[i].first;
        fitness[i] = zipped[i].second;
    }

    return {chromosomes, fitness};
}

pair<vector<vector<Gene>>, vector<int>> initialGeneration(vector<int> taskDurations, int populationsSize, int numMachines, mt19937& gen) {
    vector<vector<Gene>> chromosomes;
    vector<int> fitness;

    chromosomes.push_back(greedy(numMachines, taskDurations));
    fitness.push_back(fitnessCalculation(numMachines, chromosomes.back(), taskDurations));
    cout << "Greedy Cmax: " << fitness.back() << endl;


    uniform_int_distribution<int> machineDist(0, numMachines - 1);

    for (int i = 1; i < populationsSize; ++i) {
        vector<Gene> chromosome;
        for (int j = 0; j < taskDurations.size(); ++j) {
            chromosome.emplace_back(j, machineDist(gen));
        }
        chromosomes.push_back(chromosome);
        fitness.push_back(fitnessCalculation(numMachines, chromosome, taskDurations));
    }

    return sortChromosomes(chromosomes, fitness);
}

pair<vector<vector<Gene>>, vector<int>> evolution(vector<vector<Gene>>& chromosomes, vector<int>& fitness, double mutationProbability, int chromosomesPreserved, int maxNewChromosomes, int numMachines, vector<int>& taskDurations, double crossoverRatio, double pressure, mt19937& gen) {
    vector<vector<Gene>> newPopulation(chromosomes.begin(), chromosomes.begin() + chromosomesPreserved);
    
    uniform_int_distribution<> parentDist(0, chromosomes.size() - 1);
    int offspringNeeded = maxNewChromosomes;
    int offspringGenerated = 0;

    while (offspringGenerated < offspringNeeded) {
        int parent1 = parentDist(gen);
        int parent2 = parentDist(gen);
        
        auto [child1, child2] = crossing(chromosomes[parent1], chromosomes[parent2], crossoverRatio, taskDurations, gen);
        
        newPopulation.push_back(child1);
        offspringGenerated++;
        
        if (offspringGenerated < offspringNeeded) {
            newPopulation.push_back(child2);
            offspringGenerated++;
        }
    }

    uniform_int_distribution<> mutationRangeDist(1, numMachines - 1);
    for (size_t i = chromosomesPreserved; i < newPopulation.size(); ++i) {
        mutation(mutationProbability, newPopulation[i], numMachines, mutationRangeDist(gen), taskDurations, pressure, gen);
    }

    vector<int> newFitness;
    for (const auto& chromosome : newPopulation) {
        newFitness.push_back(fitnessCalculation(numMachines, chromosome, taskDurations));
    }

    return sortChromosomes(newPopulation, newFitness);
}

int main() {
    random_device rd;
    mt19937 gen(rd());

    Config config = loadConfig("config.json");

    auto [numMachines, taskDurations] = parseData(config.dataFile);
    if (numMachines <= 0 || taskDurations.empty()) {
        cerr << "Invalid input data" << endl;
        return 1;
    }

    int totalTaskTime = accumulate(taskDurations.begin(), taskDurations.end(), 0);
    int lowerBound = static_cast<int>(ceil(static_cast<double>(totalTaskTime) / numMachines));
    cout << "Approximate theoretical minimum Cmax: " << lowerBound << endl;

    int chromosomesPreserved = max(1, static_cast<int>(config.populationsSize * config.chromosomesPreservedPercentage / 100.0));
    int maxNewChromosomes = config.populationsSize - chromosomesPreserved;

    auto [chromosomes, fitness] = initialGeneration(taskDurations, config.populationsSize, numMachines, gen);
    BestChromosome bestChromosome = {chromosomes[0], fitness[0], 0};

    for (int generation = 1; generation <= config.generations; ++generation) {
        if (generation % 10000 == 0) {
            cout << "GENERATION: " << generation << endl;
        }

        tie(chromosomes, fitness) = evolution(chromosomes, fitness, config.mutationProbability, chromosomesPreserved, maxNewChromosomes, numMachines, taskDurations, config.crossoverRatio, config.mutationPressure, gen);

        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome = {chromosomes[0], fitness[0], generation};
            cout << "Generation " << generation << ": New best Cmax = " << bestChromosome.fitness << endl;
        }
    }

    cout << "\nFinal Results:" << endl;
    cout << "Best Cmax: " << bestChromosome.fitness << endl;
    cout << "Found in generation: " << bestChromosome.generation << endl;

    return 0;
}
