import runner_configs
import subprocess
import sys
def main():
    magick_list = ["big", "small"]
    for i in magick_list:
        result = subprocess.Popen(["powershell.exe", "-NoExit", "-Command", "python", "single_runner.py", i], shell=True)
        #subprocess.Popen(["powershell.exe", "-NoExit", "-Command", command], shell=True)
if __name__ == '__main__':
    main()