from dependencies import *

# 天然生成的是时序的数据，即每个数据是2d的，用于 MLP 需要拉成向量
class DataloaderCreator:
    def __init__(self, p=97, k=2, train_data_proportion=0.5, random_seed=42, batch_size=64, verbose=1):
        self.p = p  # 素数，用于模运算
        self.k = k  # k 元加法
        self.train_data_proportion = train_data_proportion
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.verbose = verbose
        self._set_random_seed()

    def _set_random_seed(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

    def _cal_add_mod(self, X):  # X: (sample_num, k)
        return np.sum(X, axis=1) % self.p

    def _onehot_encode(self, X):  # X: (sample_num, k)
        encode_mat = np.eye(self.p)  # 独热编码矩阵
        X_onehot = np.stack([encode_mat[X[:, i]] for i in range(self.k)], axis=1)  # (sample_num, k, p)
        if self.verbose:
            logger.info(f"X_onehot: {X_onehot.shape}")
        return X_onehot

    def make_data(self):
        # 创建所有可能的组合
        values = np.arange(self.p)
        X = np.array(list(product(values, repeat=self.k)))  # (p**k, k)
        y = self._cal_add_mod(X)  # (p**k,)
        X_onehot = self._onehot_encode(X)  # (p**k, k, p)
        if self.verbose:
            logger.info(f"X: {X.shape}, y: {y.shape}, X_onehot: {X_onehot.shape}")
        return X_onehot, y

    def creat_dataloader(self):
        X, y = self.make_data()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long))
        train_size = int(self.train_data_proportion * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_dataloader, test_dataloader


# 只需调用此函数即可得到dataloader用于训练
def creat_dataloader(p=97, k=2, train_data_proportion=0.5, random_seed=42, batch_size=64, verbose=1):
    return DataloaderCreator(
        p, k, train_data_proportion, random_seed, batch_size, verbose
    ).creat_dataloader()
