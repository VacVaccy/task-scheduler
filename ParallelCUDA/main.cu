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

#define MAX_MACHINES_PER_CHROMOSOME 1024
#define THREADS_PER_BLOCK 1024

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/**
 * @struct Config
 * @brief Struktura przechowująca konfigurację algorytmu.
 */
struct Config {
    double mutationProbability;              ///< Prawdopodobieństwo mutacji.
    int populationSize;                      ///< Liczba chromosomów w populacji.
    int chromosomesPreservedPercentage;      ///< Procent zachowanych chromosomów elitarnych.
    double splitPointRatio;                  ///< Punkt podziału w krzyżowaniu.
    int generations;                         ///< Liczba pokoleń algorytmu.
    double mutationPressure;                 ///< Wpływ presji mutacyjnej.
    string dataFile;                         ///< Ścieżka do pliku z danymi zadań.
    int maxTime;                             ///< Maksymalny czas działania algorytmu w sekundach.
};

/**
 * @brief Wczytuje konfigurację z pliku JSON.
 * @param configFile Ścieżka do pliku z konfiguracją.
 * @param dataFilePath Ścieżka do pliku z danymi wejściowymi.
 * @return Zainicjalizowana struktura Config.
 */
Config loadConfig(const string& configFile, const string& dataFilePath) {
    ifstream input(configFile);
    if (!input.is_open()) {
        cerr << "Could not open config file: " << configFile << endl;
        exit(1);
    }

    json j;
    input >> j;

    Config config;
    config.mutationProbability = j.value("mutationProbability", 0.35);
    config.populationSize = j.value("populationSize", 50);
    config.chromosomesPreservedPercentage = j.value("chromosomesPreservedPercentage", 5);
    config.splitPointRatio = j.value("splitPointRatio", 0.5);
    config.generations = j.value("generations", 50000);
    config.mutationPressure = j.value("mutationPressure", 0.15);
    config.dataFile = dataFilePath;
    config.maxTime = j.value("maxTime", 300);

    return config;
}

/**
 * @struct Gene
 * @brief Pojedynczy gen reprezentujący przypisanie zadania do maszyny.
 */
struct Gene {
    int task;      ///< ID zadania.
    int machine;   ///< ID maszyny.
    
    /**
     * @brief Konstruktor inicjalizujący gen.
     * @param t ID zadania.
     * @param m ID maszyny.
     */
    Gene(int t, int m) : task(t), machine(m) {}
};

/**
 * @struct BestChromosome
 * @brief Struktura przechowująca najlepszy znaleziony chromosom.
 */
struct BestChromosome {
    vector<Gene> chromosome; ///< Najlepszy chromosom (zadania i przypisania).
    int fitness;             ///< Wartość dopasowania (Cmax).
    int generation;          ///< Pokolenie, w którym znaleziono najlepszy wynik.
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

/**
 * @brief Wczytuje dane z pliku.
 * @param filename Ścieżka do pliku z danymi.
 * @return Para: liczba maszyn i lista czasów trwania zadań.
 */
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

/**
 * @brief Algorytm zachłanny przypisujący zadania do najmniej obciążonej maszyny.
 * @param numMachines Liczba maszyn.
 * @param taskDurations Lista czasów trwania zadań.
 * @return Chromosom utworzony przez algorytm zachłanny.
 */
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

/**
 * @brief Oblicza funkcję celu (Cmax) dla danego chromosomu.
 * @param machines Liczba maszyn.
 * @param chromosome Lista genów (zadania i przypisania).
 * @param taskDurations Lista czasów trwania zadań.
 * @return Czas zakończenia (Cmax).
 */
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

/**
 * @brief Sortuje chromosomy na podstawie ich dopasowania.
 * @param chromosomes Lista chromosomów.
 * @param fitness Lista wartości dopasowania.
 * @return Posortowane chromosomy i ich dopasowania.
 */
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

/**
 * @brief Generuje początkową populację.
 * @param taskDurations Lista czasów trwania zadań.
 * @param populationSize Rozmiar populacji.
 * @param numMachines Liczba maszyn.
 * @param gen Generator liczb losowych.
 * @return Populacja i jej dopasowania.
 */
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

/**
 * @brief Jądro CUDA obliczające dopasowanie (fitness) dla każdej jednostki w populacji.
 *
 * Każdy wątek przetwarza jeden chromosom, obliczając jego Cmax na podstawie
 * przypisań zadań do maszyn oraz ich czasów trwania.
 *
 * @param chromosomes Tablica chromosomów (zadania przypisane do maszyn), rozmiar: populationSize × numTasks.
 * @param taskDurations Czas trwania każdego zadania, rozmiar: numTasks.
 * @param fitness Wyjściowa tablica wartości fitness (Cmax), rozmiar: populationSize.
 * @param numMachines Liczba maszyn dostępnych w systemie.
 * @param numTasks Liczba zadań w każdym chromosomie.
 * @param populationSize Liczba chromosomów w populacji.
 */
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

/**
 * @brief Jądro CUDA odpowiedzialne za mutację populacji chromosomów.
 *
 * Dla każdego zadania w chromosomie istnieje prawdopodobieństwo mutacji zależne
 * od jego wpływu na Cmax. Mutacja może zmienić przypisanie zadania do innej maszyny.
 *
 * @param chromosomes Tablica chromosomów do zmodyfikowania, rozmiar: populationSize × numTasks.
 * @param taskDurations Czas trwania każdego zadania, rozmiar: numTasks.
 * @param numMachines Liczba maszyn.
 * @param numTasks Liczba zadań.
 * @param populationSize Liczba chromosomów w populacji.
 * @param mutationProbBase Bazowe prawdopodobieństwo mutacji.
 * @param mutationPressure Presja mutacyjna wpływająca na dynamiczne zwiększanie prawdopodobieństwa.
 * @param states Tablica stanów losowych CURAND, rozmiar: populationSize.
 */
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

/**
 * @brief Jądro CUDA wykonujące krzyżowanie (crossover) chromosomów.
 *
 * Dla każdej pary potomków losowani są dwaj rodzice, po czym wykonuje się krzyżowanie
 * jednopunktowe (1-point crossover) według proporcji `proportion`.
 *
 * @param parentChromosomes Tablica chromosomów rodziców, rozmiar: populationSize × numTasks.
 * @param childChromosomes Wyjściowa tablica chromosomów potomnych, rozmiar: populationSize × numTasks.
 * @param numTasks Liczba zadań w chromosomie.
 * @param populationSize Liczba chromosomów w populacji (musi być parzysta).
 * @param states Tablica stanów CURAND do generowania liczb pseudolosowych.
 * @param proportion Współczynnik podziału chromosomu (0 < proportion < 1).
 */
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

/**
 * @brief Inicjalizuje stany generatora CURAND.
 *
 * Każdy wątek inicjalizuje własny stan CURAND na podstawie ziarna, offsetu i sekwencji.
 *
 * @param seed Ziarno (seed) generatora losowego.
 * @param offset Offset w indeksie.
 * @param sequence_offset Offset sekwencji losowej.
 * @param states Wyjściowa tablica stanów CURAND, rozmiar: liczba wątków.
 */
__global__ void initCurandStatesKernel(unsigned int seed, int offset, int sequence_offset, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx + offset, sequence_offset, &states[idx]);
}

/**
 * @brief Uruchamia algorytm genetyczny na GPU.
 * 
 * Funkcja inicjalizuje populację chromosomów, alokuje pamięć na GPU, 
 * wykonuje kolejne pokolenia algorytmu genetycznego z operacjami 
 * krzyżowania, mutacji oraz oceny dopasowania (fitness). 
 * Po zakończeniu działania aktualizuje najlepszy znaleziony chromosom.
 * 
 * @param config Konfiguracja algorytmu (parametry takie jak rozmiar populacji, liczba generacji itp.)
 * @param numMachines Liczba maszyn dostępnych do przydzielenia zadań.
 * @param taskDurations Wektor z czasami trwania zadań.
 * @param bestChromosome Referencja do struktury przechowującej najlepszy znaleziony chromosom oraz jego fitness.
 */
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

/**
 * @brief Główna funkcja programu.
 * 
 * Wczytuje dane wejściowe z pliku, ładuje konfigurację algorytmu, 
 * wywołuje algorytm genetyczny działający na GPU, a następnie wyświetla
 * końcowe wyniki, takie jak najlepszy czas zakończenia (Cmax), generacja, 
 * w której znaleziono rozwiązanie, oraz dolna granica optymalnego rozwiązania.
 * 
 * @param argc Liczba argumentów wiersza poleceń.
 * @param argv Tablica argumentów wiersza poleceń.
 * @return int Kod zakończenia programu (0 - sukces, 1 - błąd).
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataFile.txt>" << endl;
        return 1;
    }

    string dataFile = argv[1];

    Config config = loadConfig("config.json", "data/" + dataFile);
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