import subprocess

# Function to create a new terminal instance
def create_terminal_instance(command):
    subprocess.Popen(["powershell.exe", "-NoExit", "-Command", command], shell=True)

create_terminal_instance("python single_runner.py big")
create_terminal_instance("python single_runner.py small")