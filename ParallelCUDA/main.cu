#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <climits>
#include <tuple> 
#include <iomanip> 
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "json.hpp"

using namespace std;
using json = nlohmann::json;

#define MAX_MACHINES_PER_CHROMOSOME 512
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct Config {
    double mutationProbability;
    int populationSize;
    int chromosomesPreservedPercentage;
    double splitPointRatio; 
    int generations;
    double mutationPressure;
    string dataFile;
    int maxTime;
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
    config.mutationProbability = j.value("mutationProbability", 0.05); 
    config.populationSize = j.value("populationSize", 200);        
    config.chromosomesPreservedPercentage = j.value("chromosomesPreservedPercentage", 10); 
    config.splitPointRatio = j.value("splitPointRatio", 0.5);
    config.generations = j.value("generations", 50000);
    config.mutationPressure = j.value("mutationPressure", 0.5);        
    config.dataFile = j.value("dataFile", "../data/data.txt");
    config.maxTime = j.value("maxTime", 300);
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
vector<Gene> greedy(int numMachines, const vector<int>& taskDurations);
int fitnessCalculation(int numMachines, const vector<Gene>& chromosome, const vector<int>& taskDurations);
pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness);
pair<vector<vector<Gene>>, vector<int>> initialGeneration(const vector<int>& taskDurations, int populationSize, int numMachines);

__global__ void fitnessCalculationKernel(int* chromosomes, const int* taskDurations, int* fitness, int numMachines, int numTasks, int populationSize);
__global__ void mutationKernel(int* chromosomes, const int* taskDurations, int numMachines, int numTasks, int populationSize, double mutationProbBase, double mutationPressure, curandState* states);
__global__ void crossoverKernel(const int* parentChromosomes, int* childChromosomes, int numTasks, int populationSize, curandState* states, double proportion);
__global__ void initCurandStatesKernel(unsigned int seed, int offset, int sequence_offset, curandState *states);

void runGeneticAlgorithmOnGPU(Config config, int numMachines, const vector<int>& taskDurations, BestChromosome& bestChromosome);

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

vector<Gene> greedy(int numMachines, const vector<int>& taskDurations) {
    vector<int> taskOrder(taskDurations.size());
    vector<int> machinesLoad(numMachines, 0); 
    vector<Gene> chromosome;
    iota(taskOrder.begin(), taskOrder.end(), 0);
    sort(taskOrder.begin(), taskOrder.end(),
         [&](int a, int b) {
             return taskDurations[a] > taskDurations[b];
         });
    for (int task : taskOrder) {
        int minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back(Gene{task, minMachine});
        machinesLoad[minMachine] += taskDurations[task];
    }
    return chromosome;
}

int fitnessCalculation(int numMachines, const vector<Gene>& chromosome, const vector<int>& taskDurations) {
    vector<int> timesList(numMachines, 0); 
    for (const auto& gene : chromosome) {
        if (gene.machine < 0 || gene.machine >= numMachines) {
            cerr << "Invalid machine number: " << gene.machine << endl;
            exit(1);
        }
        if (gene.task < 0 || gene.task >= taskDurations.size()) {
            cerr << "Invalid task number: " << gene.task << endl;
            exit(1);
        }
        timesList[gene.machine] += taskDurations[gene.task];
    }
    return static_cast<int>(*max_element(timesList.begin(), timesList.end()));
}

pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness) {
    vector<pair<vector<Gene>, int>> zipped;
    for (size_t i = 0; i < chromosomes.size(); ++i) {
        zipped.emplace_back(chromosomes[i], fitness[i]);
    }
    sort(zipped.begin(), zipped.end(),
         [](const pair<vector<Gene>, int>& a, const pair<vector<Gene>, int>& b) {
             return a.second < b.second;
         });
    for (size_t i = 0; i < zipped.size(); ++i) {
        chromosomes[i] = zipped[i].first;
        fitness[i] = zipped[i].second;
    }
    return {chromosomes, fitness};
}

pair<vector<vector<Gene>>, vector<int>> initialGeneration(const vector<int>& taskDurations, int populationSize, int numMachines) {
    vector<vector<Gene>> chromosomes;
    vector<int> fitness;
    mt19937 gen(chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> machineDist(0, numMachines - 1);
    chromosomes.push_back(greedy(numMachines, taskDurations));
    fitness.push_back(fitnessCalculation(numMachines, chromosomes.back(), taskDurations));
    cout << "Greedy Cmax: " << fitness.back() << endl;
    for (int i = 1; i < populationSize; ++i) {
        vector<Gene> chromosome;
        for (int j = 0; j < taskDurations.size(); ++j) {
            chromosome.emplace_back(j, machineDist(gen));
        }
        chromosomes.push_back(chromosome);
        fitness.push_back(fitnessCalculation(numMachines, chromosome, taskDurations));
    }
    return sortChromosomes(chromosomes, fitness);
}

__global__ void fitnessCalculationKernel(int* chromosomes, const int* taskDurations, int* fitness, int numMachines, int numTasks, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx >= populationSize) {
        return;
    }
    
    if (numMachines > MAX_MACHINES_PER_CHROMOSOME) {
        return; 
    }

    int machineLoads[MAX_MACHINES_PER_CHROMOSOME]; 

    for (int i = 0; i < numMachines; ++i) {
        machineLoads[i] = 0;
    }

    for (int i = 0; i < numTasks; ++i) {
        int assignedMachine = chromosomes[idx * numTasks + i]; 
        if (assignedMachine >= 0 && assignedMachine < numMachines) {
            machineLoads[assignedMachine] += taskDurations[i];
        }
    }

    int maxLoad = 0;
    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > maxLoad) {
            maxLoad = machineLoads[i];
        }
    }

    fitness[idx] = maxLoad;
}

__global__ void mutationKernel(int* chromosomes, const int* taskDurations, int numMachines, int numTasks, int populationSize, double mutationProbBase, double mutationPressure, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (idx >= populationSize) {
        return;
    }

    curandState localState = states[idx]; 

    if (numMachines > MAX_MACHINES_PER_CHROMOSOME) {
        return; 
    }

    int machineLoads[MAX_MACHINES_PER_CHROMOSOME]; 

    for (int i = 0; i < numMachines; ++i) {
        machineLoads[i] = 0;
    }

    for (int i = 0; i < numTasks; ++i) {
        int assignedMachine = chromosomes[idx * numTasks + i];
        if (assignedMachine >= 0 && assignedMachine < numMachines) {
            machineLoads[assignedMachine] += taskDurations[i];
        }
    }

    int currentCmax = 0;

    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > currentCmax) {
            currentCmax = machineLoads[i];
        }
    }

    for (int i = 0; i < numTasks; ++i) { 
        int oldMachine = chromosomes[idx * numTasks + i]; 
        int taskDuration = taskDurations[i];
        double criticality = (double)machineLoads[oldMachine] / currentCmax;
        double mutationProb = mutationProbBase * (1.0 + mutationPressure * criticality);

        if (curand_uniform_double(&localState) < mutationProb) {
            int bestMachine = -1;
            int minLoad = INT_MAX;

            for (int m = 0; m < numMachines; ++m) {
                if (m != oldMachine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }

            if (bestMachine != -1) {
                int updatedOldLoad = machineLoads[oldMachine] - taskDuration;
                int updatedNewLoad = machineLoads[bestMachine] + taskDuration;
                int localOldMax = max(machineLoads[oldMachine], machineLoads[bestMachine]);
                int localNewMax = max(updatedOldLoad, updatedNewLoad);
                if (localNewMax <= localOldMax) { 
                    machineLoads[oldMachine] = updatedOldLoad;
                    machineLoads[bestMachine] = updatedNewLoad;
                    chromosomes[idx * numTasks + i] = bestMachine; 
                }
            }
        }
    }
}

__global__ void crossoverKernel(const int* parentChromosomes, int* childChromosomes, int numTasks, int populationSize, curandState* states, double proportion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx >= populationSize/2) { 
        return;
    }

    curandState localState = states[idx]; 

    int parent1_idx = curand(&localState) % populationSize;
    int parent2_idx = curand(&localState) % populationSize;
    
    while (populationSize > 1 && parent2_idx == parent1_idx) { 
        parent2_idx = curand(&localState) % populationSize;
    }

    int splitPoint = static_cast<int>(numTasks * proportion);
    splitPoint = max(1, min(numTasks - 1, splitPoint));

    int child1_idx = 2 * idx;
    int child2_idx = 2 * idx + 1;

    bool usedInChild1[MAX_MACHINES_PER_CHROMOSOME]; 
    bool usedInChild2[MAX_MACHINES_PER_CHROMOSOME];
    for (int i = 0; i < numTasks; ++i) {
        usedInChild1[i] = false;
        usedInChild2[i] = false;
    }

    for (int i = 0; i < splitPoint; ++i) {
        int task = parentChromosomes[parent1_idx * numTasks + i];
        childChromosomes[child1_idx * numTasks + i] = task;
        usedInChild1[task] = true;
    }

    for (int i = 0; i < splitPoint; ++i) {
        int task = parentChromosomes[parent2_idx * numTasks + i];
        childChromosomes[child2_idx * numTasks + i] = task;
        usedInChild2[task] = true;
    }

    int pos = splitPoint;
    for (int i = 0; i < numTasks && pos < numTasks; ++i) {
        int task = parentChromosomes[parent2_idx * numTasks + i];
        if (!usedInChild1[task]) {
            childChromosomes[child1_idx * numTasks + pos] = task;
            usedInChild1[task] = true;
            pos++;
        }
    }

    pos = splitPoint;
    for (int i = 0; i < numTasks && pos < numTasks; ++i) {
        int task = parentChromosomes[parent1_idx * numTasks + i];
        if (!usedInChild2[task]) {
            childChromosomes[child2_idx * numTasks + pos] = task;
            usedInChild2[task] = true;
            pos++;
        }
    }
}

__global__ void initCurandStatesKernel(unsigned int seed, int offset, int sequence_offset, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx + offset, sequence_offset, &states[idx]);
}

void runGeneticAlgorithmOnGPU(Config config, int numMachines, const vector<int>& taskDurations, BestChromosome& bestChromosome) {
    int numTasks = taskDurations.size();
    int populationSize = config.populationSize;
    int chromosomesPreserved = max(1, static_cast<int>(populationSize * config.chromosomesPreservedPercentage / 100.0));
    int numOffspring = populationSize;

    if (numMachines > MAX_MACHINES_PER_CHROMOSOME) {
        cerr << "Error: numMachines (" << numMachines << ") exceeds MAX_MACHINES_PER_CHROMOSOME (" << MAX_MACHINES_PER_CHROMOSOME << "). Please increase MAX_MACHINES_PER_CHROMOSOME." << endl;
        exit(EXIT_FAILURE);
    }

    vector<vector<Gene>> population;
    vector<int> fitness;
    std::tie(population, fitness) = initialGeneration(taskDurations, populationSize, numMachines);

    if (fitness[0] < bestChromosome.fitness) {
        bestChromosome.chromosome = population[0];
        bestChromosome.fitness = fitness[0];
        bestChromosome.generation = 0;
    }

    int* d_population;
    int* d_taskDurations;
    int* d_fitness;
    int* d_offspring; 
    curandState* d_randStates;     

    size_t populationSizeBytes = (size_t)populationSize * numTasks * sizeof(int);
    size_t taskDurationsSizeBytes = (size_t)numTasks * sizeof(int);
    size_t fitnessSizeBytes = (size_t)populationSize * sizeof(int);
    size_t randStatesSizeBytes = (size_t)populationSize * sizeof(curandState);

    CUDA_CHECK(cudaMalloc(&d_population, populationSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_taskDurations, taskDurationsSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_fitness, fitnessSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_offspring, populationSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_randStates, randStatesSizeBytes));

    CUDA_CHECK(cudaMemcpy(d_taskDurations, taskDurations.data(), taskDurationsSizeBytes, cudaMemcpyHostToDevice));

    unsigned int seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    dim3 curand_blocks((populationSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 curand_threads(THREADS_PER_BLOCK);
    initCurandStatesKernel<<<curand_blocks, curand_threads>>>(seed, 0, 0, d_randStates);
    CUDA_CHECK(cudaDeviceSynchronize()); 

    auto globalStartTime = chrono::steady_clock::now();
    int currentBestFitness = bestChromosome.fitness; 

    vector<int> flatPopulationMachines(populationSize * numTasks);

    for (int generation = 1; generation <= config.generations; ++generation) {
        auto currentTime = chrono::steady_clock::now();
        chrono::duration<double> elapsedTime = currentTime - globalStartTime;

        if (elapsedTime.count() >= config.maxTime) {
            cout << "\nTime limit reached!" << endl;
            break;
        }

        if (generation % 10000 == 0) { 
            cout << "Generation " << generation << ": Best Cmax: " << currentBestFitness << " Elapsed Time: " << fixed << setprecision(2) << elapsedTime.count() << "s" << endl;
        }

        for (int i = 0; i < populationSize; ++i) {
            for (int j = 0; j < numTasks; ++j) {
                flatPopulationMachines[i * numTasks + j] = population[i][j].machine;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_population, flatPopulationMachines.data(), populationSizeBytes, cudaMemcpyHostToDevice));

        dim3 crossover_blocks((numOffspring + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 crossover_threads(THREADS_PER_BLOCK);
        crossoverKernel<<<crossover_blocks, crossover_threads>>>(d_population, d_offspring, numTasks, populationSize, d_randStates, config.splitPointRatio);
        CUDA_CHECK(cudaDeviceSynchronize()); 

        dim3 mutation_blocks((numOffspring + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 mutation_threads(THREADS_PER_BLOCK);
        mutationKernel<<<mutation_blocks, mutation_threads>>>(d_offspring, d_taskDurations, numMachines, numTasks, numOffspring, config.mutationProbability, config.mutationPressure, d_randStates);
        CUDA_CHECK(cudaDeviceSynchronize()); 

        dim3 fitness_blocks((numOffspring + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 fitness_threads(THREADS_PER_BLOCK);
        fitnessCalculationKernel<<<fitness_blocks, fitness_threads>>>(d_offspring, d_taskDurations, d_fitness, numMachines, numTasks, numOffspring);
        CUDA_CHECK(cudaDeviceSynchronize()); 

        vector<int> offspringFitness(numOffspring);
        vector<int> flatOffspringMachines(numOffspring * numTasks);
        CUDA_CHECK(cudaMemcpy(flatOffspringMachines.data(), d_offspring, populationSizeBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(offspringFitness.data(), d_fitness, fitnessSizeBytes, cudaMemcpyDeviceToHost));

        vector<vector<Gene>> nextGenerationChromosomes;
        vector<int> nextGenerationFitness;

        for (int i = 0; i < chromosomesPreserved; ++i) {
            nextGenerationChromosomes.push_back(population[i]); 
            nextGenerationFitness.push_back(fitness[i]);
        }

        for (int i = 0; i < numOffspring; ++i) {
            vector<Gene> offspringChromosome;
            for (int j = 0; j < numTasks; ++j) {
                offspringChromosome.emplace_back(j, flatOffspringMachines[i * numTasks + j]);
            }
            nextGenerationChromosomes.push_back(offspringChromosome);
            nextGenerationFitness.push_back(offspringFitness[i]);
        }

        std::tie(nextGenerationChromosomes, nextGenerationFitness) = sortChromosomes(nextGenerationChromosomes, nextGenerationFitness);

        population.assign(nextGenerationChromosomes.begin(), nextGenerationChromosomes.begin() + populationSize);
        fitness.assign(nextGenerationFitness.begin(), nextGenerationFitness.begin() + populationSize);

        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome.chromosome = population[0]; 
            bestChromosome.fitness = fitness[0];
            bestChromosome.generation = generation;
            currentBestFitness = bestChromosome.fitness; 
            cout << "Generation " << generation << ": New best Cmax = " << bestChromosome.fitness << " Elapsed time: " << fixed << setprecision(2) << elapsedTime.count() << "s" << endl;
        }
    }

    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_taskDurations));
    CUDA_CHECK(cudaFree(d_fitness));
    CUDA_CHECK(cudaFree(d_offspring));
    CUDA_CHECK(cudaFree(d_randStates));
}

int main() {
    Config config = loadConfig("config.json");
    int numMachines;
    vector<int> taskDurations;
    std::tie(numMachines, taskDurations) = parseData(config.dataFile);

    if (numMachines <= 0 || taskDurations.empty()) {
        std::cerr << "Invalid input data" << std::endl;
        return 1;
    }
    BestChromosome bestChromosome = {{}, INT_MAX, 0}; 
    runGeneticAlgorithmOnGPU(config, numMachines, taskDurations, bestChromosome);
    int totalTaskTime = accumulate(taskDurations.begin(), taskDurations.end(), 0);
    int lowerBound = static_cast<int>(ceil(static_cast<double>(totalTaskTime) / numMachines));
    std::cout << "\n--- Final Results ---" << std::endl;
    std::cout << "Best Cmax: " << bestChromosome.fitness << std::endl;
    std::cout << "Found in generation: " << bestChromosome.generation << std::endl;
    std::cout << "Lower bound: " << lowerBound << endl;
    return 0;
}