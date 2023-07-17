import runner_configs
import subprocess
import sys
import time
CONFIGS_FOR_RUNNING = [runner_configs.big, runner_configs.small, runner_configs.na_rezinke]

def run_single_conf(var: runner_configs.Varezhka):
    for i in range(10):
        print(var)


def main():
    args = sys.argv[1:]
    config_name = args[0]
    kro = runner_configs.__dict__[config_name]
    run_single_conf(kro)
    #time.sleep(30)


if __name__ == '__main__':
    main()