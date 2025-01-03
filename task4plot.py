from dependencies import *
from trainer import Trainer

def run_script(cmd):
    print(f"Running {cmd}...")
    subprocess.run(cmd,bufsize=1,shell=True)
    print(f"Finished {cmd}")

if __name__ == "__main__":
    # 训练所有模型，输出单模型的图片
    log_dir = "./log/task_4"
    os.makedirs(log_dir, exist_ok=True)
    scripts = sorted(os.listdir())
    pattern = r"task4_(.*?)\.py"
    scripts = [item for item in scripts if re.search(pattern, item)]
    print(scripts)
    cmd_list = [f"python {item} >> {log_dir}/{item[:-3]}.txt 2>&1 " for item in scripts]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run_script, cmd_list)
    print("All scripts have been executed.")