import random

def generate_adversarial_instance(machines=5, total_tasks=600, seed=None):
    if seed is not None:
        random.seed(seed)

    task_durations = []

    heavy_tasks = [random.randint(150, 200) for _ in range(machines)]
    task_durations.extend(heavy_tasks)

    medium_tasks = [random.randint(50, 80) for _ in range(machines * 3)]
    task_durations.extend(medium_tasks)

    small_tasks = [random.randint(1, 10) for _ in range(total_tasks - len(task_durations))]
    task_durations.extend(small_tasks)

    random.shuffle(task_durations)

    return [machines, len(task_durations)] + task_durations

def instance_to_file(filename="data.txt", seed=None):
    instance = generate_adversarial_instance(seed=seed)
    with open(filename, "w") as f:
        for value in instance:
            f.write(f"{value}\n")

instance_to_file()