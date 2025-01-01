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
    max_iter_step: int = -1 # 迭代次数的限制，如果设为-1表示没有限制
    stop_acc: float = 99 # 达到99测试正确率后停止
    subnetwork: bool = False # 是否检查子网络
    threshold: float = 1.0 # 子网络判断阈值

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
                dampening=self.dampening,  # SGD 可用的动量阻尼参数
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,  # 是否使用 Nesterov 动量
            )
        elif self.optimizer_type == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
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
        self.stepwise_train_loss = []
        self.stepwise_test_loss = []
        self.iter_step = 0 # 记录迭代步数
        self.sparsity=[] # 记录稀疏度
    
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
            out = self.model(X)[:,-1,:] # 如果是时序的输出只取最后一个
            loss = self.criterion(out, y)
            # acc = (torch.argmax(loss, dim=1) == y).sum() / len(y)
            loss.backward()
            self.optimizer.step()

            step += 1
            
            if step % self.eval_step==0:
                self._eval_on_batch_end(step)

            self.iter_step += 1
            if self.max_iter_step > 0 and self.iter_step >= self.max_iter_step:
                logger.warning(f"Reach maximum iteration steps = {self.max_iter_step}, stop training!")
                break

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
        acc = correct_cnt / len(train_subset_dataloader.sampler)
        loss = loss / len(train_subset_dataloader.sampler)
        logger.info(f"Epoch [{epoch_idx}/{self.num_epochs}], train_loss: {loss:.6f}, train_accuracy: {acc:.6f}")
        
        test_subset_dataloader = self._get_subset_dataloader(self.test_dataloader, 4096*4)
        loss = 0.0
        correct_cnt = 0
        with torch.no_grad():
            for X, y in test_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
                loss += self.criterion(out, y) * len(y)
        acc = correct_cnt / len(test_subset_dataloader.sampler)
        loss = loss / len(test_subset_dataloader.sampler)
        logger.info(f"Epoch [{epoch_idx}/{self.num_epochs}], test_loss: {loss:.6f}, test_accuracy: {acc:.6f}")

        # 判断是否达到了 99% 的 test accuracy
        if acc >= self.stop_acc/100:
            return True  # 返回 True 以触发提前停止
        return False  # 返回 False 继续训练
    
    def _eval_on_batch_end(self, step_idx):
        self.model.eval()

        # 创建子集 DataLoader
        train_subset_dataloader = self._get_subset_dataloader(self.train_dataloader)
        correct_cnt = 0
        loss = 0.0
        with torch.no_grad():
            for X, y in train_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
                loss += self.criterion(out, y) * len(y)
        acc = correct_cnt / len(train_subset_dataloader.sampler)
        loss = loss / len(train_subset_dataloader.sampler)
        if self.verbose == 2:
            logger.info(f"Step {step_idx}, train_loss: {loss:.6f}, train_accuracy: {acc:.6f}")

        self.stepwise_train_acc.extend([acc.to("cpu")] * self.eval_step)
        self.stepwise_train_loss.extend([loss.to("cpu")] * self.eval_step)

        test_subset_dataloader = self._get_subset_dataloader(self.test_dataloader)
        correct_cnt = 0
        loss = 0.0
        with torch.no_grad():
            for X, y in test_subset_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device)
                out = self.model(X)[:, -1, :]
                correct_cnt += (torch.argmax(out, dim=1) == y).sum()
                loss += self.criterion(out, y) * len(y)
        acc = correct_cnt / len(test_subset_dataloader.sampler)
        loss = loss / len(test_subset_dataloader.sampler)
        if self.verbose == 2:
            logger.info(f"Step {step_idx}, test_loss: {loss:.6f}, test_accuracy: {acc:.6f}")
        self.stepwise_test_acc.extend([acc.to("cpu")] * self.eval_step)
        self.stepwise_test_loss.extend([loss.to("cpu")] * self.eval_step)

        if self.subnetwork:
            # model_before = copy.deepcopy(self.model)
            # share = check_shared_memory(model_before, self.model)
            evaluator=SubnetworkEvaluator(self.threshold, self.model, train_subset_dataloader, self.device)
            full_accuracy = evaluator.evaluate_accuracy()
            layer_indices = [1]  # 假设要分析第1层  TODO
            subnetwork_info = evaluator.find_subnetwork(full_accuracy, layer_indices, method="linear")
            self.sparsity.append(subnetwork_info[layer_indices[0]]['size'])
            # is_equal = compare_model_weights(model_before, self.model)
            # assert is_equal, "Model weights should be equal after testing"
        # self.model.train() # 注意复原训练状态
    
    def plt_train_test_acc(self):
        os.makedirs('eval_result', exist_ok=True)
        np.save('eval_result/train_acc.npy', np.array(self.stepwise_train_acc))
        np.save('eval_result/test_acc.npy', np.array(self.stepwise_test_acc))
        plt.figure(figsize=(12, 9), dpi=300)  # 调整图形大小和分辨率
        plt.plot(list(range(len(self.stepwise_train_acc))), self.stepwise_train_acc, label='train_accuracy')
        plt.plot(list(range(len(self.stepwise_test_acc))), self.stepwise_test_acc, label='test_accuracy')
        plt.xscale('log')
        plt.ylim(0, 1.05)  # 确保 y 轴范围固定
        plt.gca().autoscale(False)  # 禁止自动调整轴范围
        plt.legend()

        # 判断任务类型
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
            plt.show()

        elif self.Q4 == "Task3":  # Task 3 保存路径
            os.makedirs('Q3', exist_ok=True)
            os.makedirs(
                f'Q3/model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}',
                exist_ok=True)
            data_size = self.num_epochs * self.train_data_proportion
            plt.savefig(
                f'Q3/model_{self.model_type}__optim_{self.optimizer_type}__lr_{self.lr}__wd_{self.weight_decay}/'
                f'dropout_{self.dropout}__momentum_{self.momentum}__nesterov_{self.nesterov}__dampening_{self.dampening}__lrGamma_{self.lr_gamma}__lrStep_{self.lr_step}__bs_{self.batch_size}__ds_{int(data_size)}__alpha_{self.train_data_proportion:.2f}_seed_{self.random_seed}.png'
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
        
        elif self.Q4 == "test":
            os.makedirs('test', exist_ok=True)
            os.makedirs(f'test/p_{self.p}__alpha_{self.train_data_proportion}__lr_{self.lr}__lrGamma_{self.lr_gamma}', exist_ok=True)
            data_size = self.num_epochs * self.train_data_proportion
            plt.savefig(
                f'test/p_{self.p}__alpha_{self.train_data_proportion}__lr_{self.lr}__lrGamma_{self.lr_gamma}/'
                f'bs_{self.batch_size}__ds_{data_size}.png'
            )

        else:
            raise ValueError("Invalid task type. Please set self.Q4 to 'Task1', 'Task2', 'Task3', or 'Task4'.")

        return self.stepwise_test_acc[-1]


    def plt_train_test_loss_sparsity(self):
        # 确保保存目录存在
        os.makedirs('Q5', exist_ok=True)

        # 保存训练损失、测试损失和稀疏性到文件
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__train_loss.npy', np.array(self.stepwise_train_loss))
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__test_loss.npy', np.array(self.stepwise_test_loss))
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__sparsity.npy', np.array(self.sparsity))

        # 对 train_loss 进行归一化
        train_loss_min = np.min(self.stepwise_train_loss)
        train_loss_max = np.max(self.stepwise_train_loss)
        normalized_train_loss = (np.array(self.stepwise_train_loss) - train_loss_min) / (train_loss_max - train_loss_min)

        # 对 test_loss 进行归一化
        test_loss_min = np.min(self.stepwise_test_loss)
        test_loss_max = np.max(self.stepwise_test_loss)
        normalized_test_loss = (np.array(self.stepwise_test_loss) - test_loss_min) / (test_loss_max - test_loss_min)

        # 对 sparsity 进行归一化
        sparsity_min = np.min(self.sparsity)
        sparsity_max = np.max(self.sparsity)
        normalized_sparsity = np.array(self.sparsity) / 64  # TODO: 需要根据模型调整神经元总数，此处假设为 64

        # 绘制图形
        plt.figure(figsize=(12, 9), dpi=300)  # 调整图形大小和分辨率
        
        # 绘制归一化后的训练损失
        plt.plot(
            list(range(len(normalized_train_loss))),
            normalized_train_loss,
            label=f'Train Loss (normalized, [{train_loss_min:.2f}, {train_loss_max:.2f}])',
            color='blue'
        )
        
        # 绘制归一化后的测试损失
        plt.plot(
            list(range(len(normalized_test_loss))),
            normalized_test_loss,
            label=f'Test Loss (normalized, [{test_loss_min:.2f}, {test_loss_max:.2f}])',
            color='orange'
        )
        
        # 绘制归一化后的稀疏性
        plt.plot(
            list(range(len(normalized_sparsity))),
            normalized_sparsity,
            label=f'Sparsity (normalized, [{sparsity_min:.2f}, {sparsity_max:.2f}])',
            color='green'
        )

        # 设置轴的缩放、范围和标签
        plt.xscale('log')
        plt.ylim(0, 1.05)  # 确保 y 轴范围固定
        plt.gca().autoscale(False)  # 禁止自动调整轴范围
        plt.xlabel('Training Steps (log scale)')
        plt.ylabel('Metrics')
        plt.title('Normalized Train Loss, Test Loss, and Sparsity over Training Steps')
        plt.legend()

        # 保存图像
        plt.savefig(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__train_test_loss_sparsity_normalized.png')
        plt.close()  # 关闭当前绘图以释放内存


    def plt_train_test_acc_sparsity(self):
        # 确保保存目录存在
        os.makedirs('Q5', exist_ok=True)

        # 保存训练正确率、测试正确率和稀疏性到文件
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__train_acc.npy', np.array(self.stepwise_train_acc))
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__test_acc.npy', np.array(self.stepwise_test_acc))
        np.save(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__sparsity.npy', np.array(self.sparsity))

        # 对 sparsity 进行归一化
        sparsity_min = np.min(self.sparsity)
        sparsity_max = np.max(self.sparsity)
        normalized_sparsity = np.array(self.sparsity) / 64  #TODO 需要增加判断神经元总数的逻辑，此处是当前模型的神经元数

        # 绘制图形
        plt.figure(figsize=(12, 9), dpi=300)  # 调整图形大小和分辨率
        
        # 绘制训练正确率
        plt.plot(
            list(range(len(self.stepwise_train_acc))),
            self.stepwise_train_acc,
            label='Train Accuracy',
            color='blue'
        )
        
        # 绘制测试正确率
        plt.plot(
            list(range(len(self.stepwise_test_acc))),
            self.stepwise_test_acc,
            label='Test Accuracy',
            color='orange'
        )
        
        # 绘制归一化后的稀疏性
        plt.plot(
            list(range(len(normalized_sparsity))),
            normalized_sparsity,
            label=f'Sparsity (normalized, [{sparsity_min:.2f}, {sparsity_max:.2f}])',
            color='green'
        )

        # 设置轴的缩放、范围和标签
        plt.xscale('log')
        plt.ylim(0, 1.05)  # 确保 y 轴范围固定
        plt.gca().autoscale(False)  # 禁止自动调整轴范围
        plt.xlabel('Training Steps (log scale)')
        plt.ylabel('Metrics')
        plt.title('Train Accuracy, Test Accuracy, and Sparsity over Training Steps')
        plt.legend()

        # 保存图像
        plt.savefig(f'Q5/fraction_{self.train_data_proportion}__threshold_{self.threshold}__train_test_acc_sparsity.png')
        plt.close()  # 关闭当前绘图以释放内存



class SubnetworkEvaluator:
    def __init__(self, threshold, model, train_subset_dataloader, device="cuda"):
        self.model = model
        self.threshold = threshold
        self.train_subset_dataloader = train_subset_dataloader
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def find_linear_layers(self,module, linear_layers):
        """
        递归遍历模型，找到所有 nn.Linear 层及其索引。
        Args:
            module: 模型或子模块。
            linear_layers: 用于存储找到的 nn.Linear 层的列表。
        """
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
            elif isinstance(layer, nn.Sequential):
                self.find_linear_layers(layer, linear_layers)

    @torch.no_grad()
    def evaluate_accuracy(self, masks=None):
        """
        评估模型在当前 mask 下的准确率。
        如果没有提供 mask，则评估完整模型。
        Args:
            masks: 字典，键为层索引，值为该层的 mask。
        """
        correct_cnt = 0
        for X, y in self.train_subset_dataloader:
            X, y = X.to(self.device).float(), y.to(self.device)
            if masks is not None:
                output = self.model.masked_forward(X, masks)
            else:
                output = self.model(X)
            out = output[:, -1, :]
            correct_cnt += (torch.argmax(out, dim=1) == y).sum()
        acc = correct_cnt / len(self.train_subset_dataloader.sampler)
        return acc
    

    def get_feature_importance(self, layer_indices):
        """
        计算指定线性层的神经元权重范数。
        Args:
            layer_indices: 需要分析的逻辑线性层索引列表（非 nn.Sequential 的索引）。
        Returns:
            feature_importance: 字典，键为逻辑层索引，值为该层神经元的范数组成的数组。
        """
        feature_importance = {}
        # 获取所有 nn.Linear 层（按顺序存储）
        linear_layers = []
        self.find_linear_layers(self.model, linear_layers)

        # 遍历逻辑索引列表
        for logical_idx in layer_indices:
            if logical_idx >= len(linear_layers):
                raise ValueError(f"Layer index {logical_idx} exceeds the number of linear layers in the model.")
            
            # 获取对应的 nn.Linear 层权重
            layer_weights = linear_layers[logical_idx].weight.detach().clone()
            norms = torch.linalg.norm(layer_weights, dim=1).detach().clone().cpu().numpy()
            feature_importance[logical_idx] = norms

        return feature_importance

    def find_subnetwork(self, full_acc, layer_indices, method="linear"):
        """
        通过线性或二分搜索找到每一层的最小子网络。
        Args:
            full_acc: 完整模型的准确率。
            layer_indices: 需要分析的层的索引列表。
            method: 搜索方法，"linear" 或 "binary"。
        Returns:
            subnetwork_info: 字典，记录每一层的子网络大小和重要神经元索引。
        """
        subnetwork_info = {}
        feature_importance = self.get_feature_importance(layer_indices)

        for layer_idx, importances in feature_importance.items():
            sorted_indices = np.argsort(-importances)  # 按范数降序排序
            if method == "linear":
                k, active_indices = self.linear_search(layer_idx, sorted_indices, full_acc)
            elif method == "binary":
                k, active_indices = self.binary_search(layer_idx, sorted_indices, full_acc)
            else:
                raise ValueError("Unsupported search method: Choose 'linear' or 'binary'.")
            subnetwork_info[layer_idx] = {"size": k, "indices": active_indices}
        return subnetwork_info

    def linear_search(self, layer_idx, sorted_indices, full_acc):
        """
        线性搜索，逐步增加指定层的神经元，直到性能接近完整模型。
        """
        for k in range(1, len(sorted_indices) + 1):
            active_indices = sorted_indices[:k]
            masks = self.create_mask(layer_idx, active_indices)
            subnetwork_acc = self.evaluate_accuracy(masks)
            if subnetwork_acc >= full_acc*self.threshold:
                return k, active_indices
        return float("inf"), None

    def binary_search(self, layer_idx, sorted_indices, full_acc):
        """
        二分搜索，快速找到指定层的最小子网络。
        """
        left, right = 1, len(sorted_indices)
        prev_k = -1
        best_k, best_indices = float("inf"), None
        while left <= right:
            k = (left + right) // 2
            if (prev_k == k):
                break
            active_indices = sorted_indices[:k]
            mask = self.create_mask(layer_idx, active_indices)
            masks = {layer_idx: mask}
            subnetwork_acc = self.evaluate_accuracy(masks)
            if subnetwork_acc >= full_acc*self.threshold:
                best_k, best_indices = k, active_indices
                right = k - 1
            else:
                left = k + 1
            prev_k = k
        return best_k, best_indices

    def create_mask(self, layer_idx, active_indices):
        """
        为 nn.Sequential 的特定层创建 mask。
        Args:
            layer_idx: 逻辑上的第几层线性层（不是 nn.Sequential 的索引）。
            active_indices: 当前层中需要激活的神经元索引。
        """
        masks = {}
        # 获取所有 nn.Linear 层（按顺序存储）
        linear_layers = []
        self.find_linear_layers(self.model, linear_layers)

        if layer_idx >= len(linear_layers):
            raise ValueError(f"Layer index {layer_idx} exceeds the number of linear layers in the model.")
        
        # 获取对应的 nn.Linear 层权重
        layer_weights = linear_layers[layer_idx].weight
        output_dim = layer_weights.size(0)  # 输出维度
    
        # 检查 active_indices 是否超出范围
        if max(active_indices) >= output_dim:
            raise IndexError(f"Active indices exceed the number of neurons in layer {layer_idx} (output_dim={output_dim}).")
                
        # 检查 active_indices 是否超出范围
        if max(active_indices) >= layer_weights.size(0):
            raise IndexError(f"Active indices exceed the number of neurons in layer {layer_idx}.")

        # 创建 mask，形状为 [output_dim]
        mask = torch.zeros(output_dim, device=self.device) # ATTENTION:内存共享
        # mask = torch.zeros_like(layer_weights).to(self.device)
        mask[active_indices] = 1  # 激活指定的神经元
        masks[layer_idx] = mask
        return masks


def compare_model_weights(model1, model2):
    """
    比较两个模型的权重是否一致。
    Args:
        model1: 第一个模型。
        model2: 第二个模型。
    Returns:
        是否一致 (bool)。
    """
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Layer mismatch: {name1} != {name2}")
            return False
        if not torch.equal(param1.data, param2.data):
            print(f"Weight mismatch in layer: {name1}")
            return False
    return True

# 检查是否共享内存
def check_shared_memory(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, "Model layers do not match!"
        if param1.data.data_ptr() == param2.data.data_ptr():
            print(f"Shared memory detected in layer: {name1}")
            return False
    return True