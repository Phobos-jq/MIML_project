from dependencies import *
from trainer import Trainer

def run_script(cmd):
    print(f"Running {cmd}...")
    subprocess.run(cmd,bufsize=1,shell=True)
    print(f"Finished {cmd}")

if __name__ == "__main__":
    
    flag_trained = True # 是否已经训练完毕并保存了相应的validation信息
    

    if flag_trained:
        # 只通过保存的内容作图
        scripts = os.listdir()
        pattern = r"cfq_demo_task3_(.*?)\.py"
        name_list = []
        for item in scripts:
            match = re.search(pattern, item)
            if match:
                name_list.append(match.group(1))
        print(name_list)
        x_list = []
        y_list = []
        for item in name_list:
            path = f"./Q3_cfq_result/df_best_val_acc_{item}.pq"
            if os.path.exists(path):
                val_res = pd.read_parquet(path)
            else:
                logger.warning(f"{path} do not exist! Use previous one.")
            y_list.append(val_res.values * 100)
            x_list.append(list(val_res.columns))
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            y = np.mean(y_list[i],axis=0)
            ax.plot(x_list[i], y , label="Best validation accuracy")
            for j in range(len(y_list[i])):
                ax.scatter(x_list[i], y_list[i][j,:])
            ax.set_title(name_list[i])
            ax.set_ylim(0, 102)
            ax.set_xlabel("Training data fraction")
            ax.set_ylabel("Best validation accuracy")
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"./Q3_cfq_result/fig/q3_final_fig.png")
        plt.show()

    else:
        # 训练所有模型，输出单模型的图片
        log_dir = "./log/task_3"
        os.makedirs(log_dir, exist_ok=True)
        scripts = os.listdir()
        pattern = r"cfq_demo_task3_(.*?)\.py"
        scripts = [item for item in scripts if re.search(pattern, item)]
        print(scripts)
        cmd_list = [f"python {item} > {log_dir}/{item[:-3]}.txt 2>&1 " for item in scripts]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(run_script, cmd_list)
        print("All scripts have been executed.")