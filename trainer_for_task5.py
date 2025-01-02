from dependencies import *
from data import creat_dataloader
from model import Transformer, MLP, LSTM

@dataclass
class Trainer:
    p: int = 97
    k: int = 2
    Q4: bool = "Task4"     # 判断图像保存在Q3文件夹还是Q4文件夹
    train_data_proportion: float = 0.5
    random_seed: int = 42
    batch_size: int = 512
    verbose: int = 1
    eval_step: int = 1  # 每隔多少个step计算一次train和test的acc
    num_epochs: int = 50
    num_layers: int = 2
    seq_len: int = 2  
    lr: float = 1e-4
    weight_decay: float = 1e-4
    model_type: str = "MLP" # Transformer, MLP, LSTM
    MLP_layer_dims: list[int] = field(default_factory=lambda:[512, 128, 64, 64])
    optimizer_type: str = "AdamW"  # 支持不同优化器类型：AdamW, SGD, RMSprop 等
    lr_gamma: float = 0.99  # 每过lr_step个epoch, lr 乘以 lr_gamma
    lr_step: int = 50
    momentum: float = 0.9  # SGD 和 RMSprop 可用的动量参数
    nesterov: bool = False  # 是否使用 Nesterov 动量 (仅对 SGD 有效)
    dampening: float = 0.0  # SGD 可用的动量阻尼参数
    dropout: float = 0.2  # Dropout 概率
    # 新增自适应噪声相关参数
    use_adaptive_noise: bool = False  # 是否使用自适应噪声
    noise_lambda1: float = 0.5  # 论文中模运算任务中使用的λ1
    noise_lambda2: float = 0.4  # 论文中模运算任务中使用的λ2
    def __post_init__(self):      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose >= 1:
            logger.info(f"Device: {self.device}")

        # 初始化数据加载器，模型, 优化器，学习率， 损失等
        self.train_dataloader, self.test_dataloader = creat_dataloader(
                p=self.p,
                k=self.k,
                train_data_proportion=self.train_data_proportion,
                random_seed=self.random_seed,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
        if self.model_type == "Transformer":
            self.model = Transformer(num_layers=self.num_layers, p=self.p, seq_len=self.seq_len, dropout=self.dropout).to(self.device)
        elif self.model_type == "MLP":
            self.model = MLP(layer_dims=self.MLP_layer_dims, p=self.p).to(self.device)
        elif self.model_type == "LSTM":
            self.model = LSTM(input_dim=self.p, seq_len=self.seq_len).to(self.device)
        
        # 初始化优化器
        if self.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif self.optimizer_type == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
            
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR( 
                self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma
            )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.stepwise_train_acc = [] # 用来画图
        self.stepwise_test_acc = []
        self.epoch_l2_norms = []  # 存储每个epoch的L2范数
        self.epoch_train_acc = []  # 存储每个epoch的训练准确率
        self.epoch_test_acc = []   # 存储每个epoch的测试准确率

    def fit(self):
        for epoch_idx, epoch in enumerate(range(self.num_epochs)):
            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Learning Rate: {self.lr_scheduler.get_last_lr()[0]:e}")
            self._train_epoch()
            if self.verbose >= 1:
                logger.info(f"Begin epoch {epoch_idx+1} evaluating.")
            if self._eval_on_epoch_end(epoch_idx+1):
                logger.info(f"Test accuracy reached 100%. Stopping early at epoch {epoch_idx+1}.")
                break  # 提前停止
        # self._eval_on_train_end()

    def _train_epoch(self):
        step = 0
        for X, y in self.train_dataloader:
            self.model.train()
            X, y = X.to(self.device).float(), y.to(self.device)
            self.optimizer.zero_grad()

            # 计算当前的训练准确率用于自适应噪声
            with torch.no_grad():
                clean_out = self.model(X)[:,-1,:]
                current_acc = (torch.argmax(clean_out, dim=1) == y).float().mean()
                
            # 计算自适应噪声强度
            if self.use_adaptive_noise:
                noise_std = max(
                    self.noise_lambda1 * (1 - current_acc), 
                    self.noise_lambda2
                )
            else:
                noise_std = 0.0

            # 使用噪声训练
            out = self.model(X, add_noise=True, noise_std=noise_std)[:,-1,:]
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            step += 1
            if step%self.eval_step==0:
                self._eval_on_batch_end(step)
        self.lr_scheduler.step()
    
    def _get_subset_dataloader(self, dataloader, test_size=4096):
        """
        创建 DataLoader 的子集用于计算train和test的acc （抽样计算acc）。
        """
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)  # 随机打乱索引

        # 选择部分索引
        subset_size = test_size
        subset_indices = indices[:subset_size]

        # 创建子集采样器
        sampler = SubsetRandomSampler(subset_indices)
        return DataLoader(dataset, sampler=sampler, batch_size=dataloader.batch_size)

    def _eval_on_epoch_end(self, epoch_idx):
        self.model.eval()

        # 计算并存储L2范数
        l2_norm = self._calculate_l2_norm()
        self.epoch_l2_norms.append(l2_norm)

        # 创建子集 DataLoader
        train_subset_dataloader = self._get_subset_dataloader(self.train_dataloader, 4096*4)
        loss = 0.0
        correct_cnt = 0
        with torch.no_grad():
            for X, y in train_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
                loss += self.criterion(out, y) * len(y)
        train_acc = correct_cnt / len(train_subset_dataloader.sampler)
        loss = loss / len(train_subset_dataloader.sampler)
        self.epoch_train_acc.append(train_acc.item())
        logger.info(f"Epoch [{epoch_idx}/{self.num_epochs}], train_loss: {loss:.6f}, train_accuracy: {train_acc:.6f}")
        
        test_subset_dataloader = self._get_subset_dataloader(self.test_dataloader, 4096*4)
        loss = 0.0
        correct_cnt = 0
        with torch.no_grad():
            for X, y in test_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
                loss += self.criterion(out, y) * len(y)
        test_acc = correct_cnt / len(test_subset_dataloader.sampler)
        loss = loss / len(test_subset_dataloader.sampler)
        self.epoch_test_acc.append(test_acc.item())
        logger.info(f"Epoch [{epoch_idx}/{self.num_epochs}], test_loss: {loss:.6f}, test_accuracy: {test_acc:.6f}")

        # 判断是否达到了 99% 的 test accuracy
        if test_acc >= 0.99:
            return True  # 返回 True 以触发提前停止
        return False  # 返回 False 继续训练

    def _eval_on_batch_end(self, step_idx):
        self.model.eval()

        # 创建子集 DataLoader
        train_subset_dataloader = self._get_subset_dataloader(self.train_dataloader)
        correct_cnt = 0
        with torch.no_grad():
            for X, y in train_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X, add_noise=False)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
        acc = correct_cnt / len(train_subset_dataloader.sampler)
        if self.verbose == 2:
            logger.info(f"Step {step_idx}, train_accuracy: {acc:.6f}")
        self.stepwise_train_acc.extend([acc.to("cpu")] * self.eval_step)

        test_subset_dataloader = self._get_subset_dataloader(self.test_dataloader)
        correct_cnt = 0
        with torch.no_grad():
            for X, y in test_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
        acc = correct_cnt / len(test_subset_dataloader.sampler)
        if self.verbose == 2:
            logger.info(f"Step {step_idx}, test_accuracy: {acc:.6f}")
        self.stepwise_test_acc.extend([acc.to("cpu")] * self.eval_step)
        # self.model.train() # 注意复原训练状态
    
    def _calculate_l2_norm(self):
        """计算模型所有参数的L2范数"""
        total_norm = 0
        with torch.no_grad():
            for p in self.model.parameters():
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return total_norm

    def plt_train_test_acc(self):
        if self.Q4 == "Task5":
            # 创建Task5的图表
            fig, ax1 = plt.subplots(figsize=(12, 9), dpi=300)
            ax2 = ax1.twinx()

            # 绘制准确率（左y轴）
            epochs = range(1, len(self.epoch_train_acc) + 1)
            ln1 = ax1.plot(epochs, self.epoch_train_acc, 'b-', label='Train Accuracy')
            ln2 = ax1.plot(epochs, self.epoch_test_acc, 'g-', label='Test Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_xscale('log')  


            # 绘制L2范数（右y轴）
            ln3 = ax2.plot(epochs, self.epoch_l2_norms, 'r-', label='L2 Norm')
            ax2.set_ylabel('L2 Norm')

            # 合并图例
            lns = ln1 + ln2 + ln3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='center right')

            # 设置标题
            plt.title(f'Training Progress with L2 Norm - {self.model_type}')

            # 创建保存路径
            save_dir = f'Q5/p_{self.p}__k_{self.k}__model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}'
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存图表
            data_size = self.num_epochs * self.train_data_proportion
            filename = f'{save_dir}/dropout_{self.dropout}__momentum_{self.momentum}__nesterov_{self.nesterov}__dampening_{self.dampening}__lrGamma_{self.lr_gamma}__lrStep_{self.lr_step}__bs_{self.batch_size}__ds_{data_size}__alpha_{self.train_data_proportion}_with_l2norm.png'
            plt.savefig(filename)
            plt.close()

            return self.epoch_test_acc[-1]
        
        elif self.Q4 == "Task5_accelerate":
            # 创建Task5的图表
            fig, ax1 = plt.subplots(figsize=(12, 9), dpi=300)
            ax2 = ax1.twinx()

            # 绘制准确率（左y轴）
            epochs = range(1, len(self.epoch_train_acc) + 1)
            ln1 = ax1.plot(epochs, self.epoch_train_acc, 'b-', label='Train Accuracy')
            ln2 = ax1.plot(epochs, self.epoch_test_acc, 'g-', label='Test Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_xscale('log')  


            # 绘制L2范数（右y轴）
            ln3 = ax2.plot(epochs, self.epoch_l2_norms, 'r-', label='L2 Norm')
            ax2.set_ylabel('L2 Norm')

            # 合并图例
            lns = ln1 + ln2 + ln3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='center right')

            # 设置标题
            plt.title(f'Training Progress with L2 Norm - {self.model_type}')

            # 创建保存路径
            save_dir = f'Q5/accelerate_p_{self.p}__k_{self.k}__model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}'
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存图表
            data_size = self.num_epochs * self.train_data_proportion
            filename = f'{save_dir}/dropout_{self.dropout}__momentum_{self.momentum}__nesterov_{self.nesterov}__dampening_{self.dampening}__lrGamma_{self.lr_gamma}__lrStep_{self.lr_step}__bs_{self.batch_size}__ds_{data_size}__alpha_{self.train_data_proportion}_with_l2norm.png'
            plt.savefig(filename)
            plt.close()

            return self.epoch_test_acc[-1]
            
        else:
            # 保持原有Task1-4的逻辑完全不变
            os.makedirs('eval_result', exist_ok=True)
            np.save('eval_result/train_acc.npy', np.array(self.stepwise_train_acc))
            np.save('eval_result/test_acc.npy', np.array(self.stepwise_test_acc))
            
            plt.figure(figsize=(12, 9), dpi=300)  # 调整图形大小和分辨率
            plt.plot(list(range(len(self.stepwise_train_acc))), self.stepwise_train_acc, label='train_accuracy')
            plt.plot(list(range(len(self.stepwise_test_acc))), self.stepwise_test_acc, label='test_accuracy')
            plt.xscale('log')
            plt.legend()

            if self.Q4 == "Task4":  # Task 4 保存路径
                os.makedirs('Q4', exist_ok=True)
                os.makedirs(
                    f'Q4/p_{self.p}__k_{self.k}__model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}',
                    exist_ok=True)
                data_size = self.num_epochs * self.train_data_proportion
                plt.savefig(
                    f'Q4/p_{self.p}__k_{self.k}__model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}/'
                    f'dropout_{self.dropout}__momentum_{self.momentum}__nesterov_{self.nesterov}__dampening_{self.dampening}__lrGamma_{self.lr_gamma}__lrStep_{self.lr_step}__bs_{self.batch_size}__ds_{data_size}__alpha_{self.train_data_proportion}.png'
                )

            elif self.Q4 == "Task3":  # Task 3 保存路径
                os.makedirs('Q3', exist_ok=True)
                os.makedirs(
                    f'Q3/model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}',
                    exist_ok=True)
                data_size = self.num_epochs * self.train_data_proportion
                plt.savefig(
                    f'Q3/model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}/'
                    f'dropout_{self.dropout}__momentum_{self.momentum}__nesterov_{self.nesterov}__dampening_{self.dampening}__lrGamma_{self.lr_gamma}__lrStep_{self.lr_step}__bs_{self.batch_size}__ds_{data_size}__alpha_{self.train_data_proportion}.png'
                )

            elif self.Q4 == "Task2":  # Task 2 保存路径
                os.makedirs('Q2', exist_ok=True)
                os.makedirs(f'Q2/p_{self.p}__k_{self.k}__model_{self.model_type}', exist_ok=True)
                data_size = self.num_epochs * self.train_data_proportion
                plt.savefig(
                    f'Q2/p_{self.p}__k_{self.k}__model_{self.model_type}/'
                    f'bs_{self.batch_size}__ds_{data_size}__alpha_{self.train_data_proportion}.png'
                )

            elif self.Q4 == "Task1":  # Task 1 保存路径
                os.makedirs('Q1', exist_ok=True)
                os.makedirs(f'Q1/p_{self.p}__alpha_{self.train_data_proportion}__lr_{self.lr}__lrGamma_{self.lr_gamma}', exist_ok=True)
                data_size = self.num_epochs * self.train_data_proportion
                plt.savefig(
                    f'Q1/p_{self.p}__alpha_{self.train_data_proportion}__lr_{self.lr}__lrGamma_{self.lr_gamma}/'
                    f'bs_{self.batch_size}__ds_{data_size}.png'
                )

            else:
                raise ValueError("Invalid task type. Please set self.Q4 to 'Task1', 'Task2', 'Task3', 'Task4', or 'Task5'.")
                
            plt.close()
            return self.stepwise_test_acc[-1]