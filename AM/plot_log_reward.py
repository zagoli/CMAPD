from matplotlib import pyplot as plt
from pathlib import Path


def is_epoch_line(line: str):
    return line.startswith('epoch')


def get_reward_from_line(line: str) -> float:
    line_array = line.split(',')
    reward_str = line_array[3]
    reward = reward_str.split('=')[1]
    return float(reward)


if __name__ == '__main__':
    log_path = Path('CONVERGENCE_TEST_13-04-2023_19:15:58.log')
    rewards = []
    assert log_path.exists()
    log_file = open(log_path, 'r')
    while True:
        line = log_file.readline()
        if not line:
            break
        if is_epoch_line(line):
            rewards.append(get_reward_from_line(line))
    log_file.close()
    plt.plot(rewards)
    plt.xlim([0, len(rewards)-1])
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title(str(log_path.stem))
    plt.show()
