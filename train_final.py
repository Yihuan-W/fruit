import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import os
import json
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class FruitDataset(Dataset):
    def __init__(self, time_data, spectrum_data, labels):
        self.time_data = torch.FloatTensor(time_data)
        self.spectrum_data = torch.FloatTensor(spectrum_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'time': self.time_data[idx],
            'spectrum': self.spectrum_data[idx],
            'label': self.labels[idx]
        }


class TimeDomainCNN(nn.Module):
    def __init__(self, num_classes, input_length=100):
        super(TimeDomainCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpectrumCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(50, 50)):
        super(SpectrumCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, num_classes, time_length=100, spectrum_shape=(50, 50)):
        super(FusionModel, self).__init__()

        # 时域分支
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 频谱分支
        self.spectrum_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, time_x, spectrum_x):
        # 时域特征提取
        time_feat = self.time_branch(time_x)
        time_feat = time_feat.view(time_feat.size(0), -1)

        # 频谱特征提取
        spectrum_feat = self.spectrum_branch(spectrum_x)
        spectrum_feat = spectrum_feat.view(spectrum_feat.size(0), -1)

        # 特征融合
        fused = torch.cat([time_feat, spectrum_feat], dim=1)
        output = self.classifier(fused)

        return output


class FruitClassifier:
    def __init__(self, model, device, model_name):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            time_data = batch['time'].to(self.device)
            spectrum_data = batch['spectrum'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()

            if isinstance(self.model, FusionModel):
                outputs = self.model(time_data, spectrum_data)
            elif isinstance(self.model, TimeDomainCNN):
                outputs = self.model(time_data)
            else:  # SpectrumCNN
                outputs = self.model(spectrum_data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                time_data = batch['time'].to(self.device)
                spectrum_data = batch['spectrum'].to(self.device)
                labels = batch['label'].to(self.device)

                if isinstance(self.model, FusionModel):
                    outputs = self.model(time_data, spectrum_data)
                elif isinstance(self.model, TimeDomainCNN):
                    outputs = self.model(time_data)
                else:  # SpectrumCNN
                    outputs = self.model(spectrum_data)

                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, all_predictions, all_labels

    def predict(self, dataloader):
        """预测整个数据集"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                time_data = batch['time'].to(self.device)
                spectrum_data = batch['spectrum'].to(self.device)
                labels = batch['label'].to(self.device)

                if isinstance(self.model, FusionModel):
                    outputs = self.model(time_data, spectrum_data)
                elif isinstance(self.model, TimeDomainCNN):
                    outputs = self.model(time_data)
                else:  # SpectrumCNN
                    outputs = self.model(spectrum_data)

                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_predictions, all_labels

    def extract_features(self, dataloader):
        """提取特征用于t-SNE"""
        self.model.eval()
        features = []
        labels_list = []

        with torch.no_grad():
            for batch in dataloader:
                time_data = batch['time'].to(self.device)
                spectrum_data = batch['spectrum'].to(self.device)
                labels = batch['label'].to(self.device)

                # 获取特征
                if isinstance(self.model, FusionModel):
                    # 获取融合前的特征
                    time_feat = self.model.time_branch(time_data)
                    spectrum_feat = self.model.spectrum_branch(spectrum_data)
                    time_feat = time_feat.view(time_feat.size(0), -1)
                    spectrum_feat = spectrum_feat.view(spectrum_feat.size(0), -1)
                    feature = torch.cat([time_feat, spectrum_feat], dim=1)
                elif isinstance(self.model, TimeDomainCNN):
                    feature = self.model.conv_layers(time_data)
                    feature = feature.view(feature.size(0), -1)
                else:  # SpectrumCNN
                    feature = self.model.conv_layers(spectrum_data)
                    feature = feature.view(feature.size(0), -1)

                features.append(feature.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        features = np.concatenate(features)
        labels_list = np.concatenate(labels_list)

        return features, labels_list

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
        best_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if scheduler:
                scheduler.step()

            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model_name': self.model_name
                }, f'best_{self.model_name}.pth')

        return best_acc


def load_and_preprocess_data(train_path, val_path):
    """加载和预处理数据"""
    try:
        # 尝试加载数据
        train_data = np.load(train_path)
        val_data = np.load(val_path)

        print("数据加载成功!")
        print(f"训练集键: {list(train_data.keys())}")
        print(f"验证集键: {list(val_data.keys())}")

        # 根据实际的数据键名调整
        X_train_time = train_data['X_time']
        X_train_spec = train_data['X_spec']
        Y_train = train_data['Y']

        X_val_time = val_data['X_time']
        X_val_spec = val_data['X_spec']
        Y_val = val_data['Y']

        # 数据预处理
        # 标准化
        X_train_time = (X_train_time - np.mean(X_train_time)) / np.std(X_train_time)
        X_val_time = (X_val_time - np.mean(X_val_time)) / np.std(X_val_time)

        X_train_spec = (X_train_spec - np.mean(X_train_spec)) / np.std(X_train_spec)
        X_val_spec = (X_val_spec - np.mean(X_val_spec)) / np.std(X_val_spec)

        print(f"训练集 - 时域: {X_train_time.shape}, 频谱: {X_train_spec.shape}, 标签: {Y_train.shape}")
        print(f"验证集 - 时域: {X_val_time.shape}, 频谱: {X_val_spec.shape}, 标签: {Y_val.shape}")

        return X_train_time, X_train_spec, Y_train, X_val_time, X_val_spec, Y_val

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请检查文件路径是否正确")
        return None
    except KeyError as e:
        print(f"数据键错误: {e}")
        print("请检查NPZ文件中的键名")
        return None


def plot_training_history(trainer, model_name):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 准确率
    ax1.plot(trainer.train_accs, label='训练准确率')
    ax1.plot(trainer.val_accs, label='验证准确率')
    ax1.set_title(f'{model_name} - 准确率')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    # 损失
    ax2.plot(trainer.train_losses, label='训练损失')
    ax2.plot(trainer.val_losses, label='验证损失')
    ax2.set_title(f'{model_name} - 损失')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_confusion_matrix_data(y_true, y_pred, class_names, model_name, dataset_type):
    """保存混淆矩阵数据"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 保存原始混淆矩阵
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f'{model_name}_{dataset_type}_confusion_matrix.csv')

    # 保存百分比混淆矩阵
    cm_percentage_df = pd.DataFrame(cm_percentage, index=class_names, columns=class_names)
    cm_percentage_df.to_csv(f'{model_name}_{dataset_type}_confusion_matrix_percentage.csv')

    print(f"已保存混淆矩阵数据: {model_name}_{dataset_type}_confusion_matrix.csv")
    print(f"已保存百分比混淆矩阵数据: {model_name}_{dataset_type}_confusion_matrix_percentage.csv")

    return cm, cm_percentage


def plot_confusion_matrix_percentage(cm_percentage, class_names, model_name, dataset_type, cmap='Blues'):
    """绘制百分比形式的混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '百分比 (%)'})
    plt.title(f'{model_name} - {dataset_type}混淆矩阵 (%)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'{model_name}_{dataset_type}_confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_tsne_data(features_2d, labels, class_names, model_name, dataset_type):
    """保存t-SNE数据"""
    tsne_df = pd.DataFrame({
        'tsne_x': features_2d[:, 0],
        'tsne_y': features_2d[:, 1],
        'label': labels,
        'class_name': [class_names[i] for i in labels]
    })
    tsne_df.to_csv(f'{model_name}_{dataset_type}_tsne_data.csv', index=False)
    print(f"已保存t-SNE数据: {model_name}_{dataset_type}_tsne_data.csv")

    return tsne_df


def plot_tsne_from_data(tsne_df, class_names, model_name, dataset_type, cmap='tab10', figsize=(12, 10)):
    """从保存的数据绘制t-SNE图"""
    plt.figure(figsize=figsize)
    scatter = plt.scatter(tsne_df['tsne_x'], tsne_df['tsne_y'],
                          c=tsne_df['label'], cmap=cmap, alpha=0.7, s=10)

    # 添加图例
    legend_elements = scatter.legend_elements()[0]
    plt.legend(legend_elements, class_names, title='水果类别')

    plt.title(f'{model_name} - {dataset_type} t-SNE 可视化')
    plt.xlabel('t-SNE 特征 1')
    plt.ylabel('t-SNE 特征 2')

    # 去除背景网格
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(f'{model_name}_{dataset_type}_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()


def compute_tsne(features, labels, class_names, model_name, dataset_type, perplexity=30):
    """计算并保存t-SNE"""
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    # 保存t-SNE数据
    tsne_df = save_tsne_data(features_2d, labels, class_names, model_name, dataset_type)

    return features_2d, tsne_df


class VisualizationManager:
    """可视化管理器，用于随时调用模型重新生成图表"""

    def __init__(self, class_names):
        self.class_names = class_names
        self.saved_data = {}

    def load_model(self, model, model_path, device):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"已加载模型: {model_path}")
        return model

    def regenerate_confusion_matrix(self, model_name, dataset_type, cmap='Blues'):
        """重新生成混淆矩阵图"""
        cm_percentage_path = f'{model_name}_{dataset_type}_confusion_matrix_percentage.csv'
        if os.path.exists(cm_percentage_path):
            cm_percentage_df = pd.read_csv(cm_percentage_path, index_col=0)
            plot_confusion_matrix_percentage(
                cm_percentage_df.values,
                self.class_names,
                model_name,
                dataset_type,
                cmap=cmap
            )
        else:
            print(f"未找到混淆矩阵数据: {cm_percentage_path}")

    def regenerate_tsne(self, model_name, dataset_type, cmap='tab10', figsize=(12, 10)):
        """重新生成t-SNE图"""
        tsne_path = f'{model_name}_{dataset_type}_tsne_data.csv'
        if os.path.exists(tsne_path):
            tsne_df = pd.read_csv(tsne_path)
            plot_tsne_from_data(
                tsne_df,
                self.class_names,
                model_name,
                dataset_type,
                cmap=cmap,
                figsize=figsize
            )
        else:
            print(f"未找到t-SNE数据: {tsne_path}")

    def compare_all_models(self, model_names, dataset_type='验证集'):
        """比较所有模型的混淆矩阵"""
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]

        for i, model_name in enumerate(model_names):
            cm_percentage_path = f'{model_name}_{dataset_type}_confusion_matrix_percentage.csv'
            if os.path.exists(cm_percentage_path):
                cm_percentage_df = pd.read_csv(cm_percentage_path, index_col=0)
                sns.heatmap(cm_percentage_df, annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=self.class_names, yticklabels=self.class_names,
                            cbar_kws={'label': '百分比 (%)'}, ax=axes[i])
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('预测标签')
                axes[i].set_ylabel('真实标签')
            else:
                axes[i].set_title(f'{model_name}\n(数据未找到)')

        plt.tight_layout()
        plt.savefig(f'all_models_{dataset_type}_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据文件路径 - 请根据实际情况修改
    train_path = "dataset/dataset_train_12s.npz"
    val_path = "dataset/dataset_val_12s.npz"

    # 加载数据
    print("正在加载数据...")
    data = load_and_preprocess_data(train_path, val_path)
    if data is None:
        print("数据加载失败，请检查文件路径")
        return

    X_train_time, X_train_spec, Y_train, X_val_time, X_val_spec, Y_val = data

    # 创建数据集和数据加载器
    train_dataset = FruitDataset(X_train_time, X_train_spec, Y_train)
    val_dataset = FruitDataset(X_val_time, X_val_spec, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(np.unique(Y_train))
    print(f"类别数量: {num_classes}")

    # 类别名称（10类水果）
    class_names = ["哈密瓜", "松果", "枣", "柠檬", "树莓", "核桃", "牛油果", "玉米", "草莓", "荔枝"]

    # 创建可视化管理器
    vis_manager = VisualizationManager(class_names[:num_classes])

    # 训练不同的模型
    models_config = {
        '时域模型': TimeDomainCNN(num_classes, X_train_time.shape[-1]),
        '频谱模型': SpectrumCNN(num_classes, (X_train_spec.shape[-2], X_train_spec.shape[-1])),
        '融合模型': FusionModel(num_classes, X_train_time.shape[-1], (X_train_spec.shape[-2], X_train_spec.shape[-1]))
    }

    results = {}

    for model_name, model in models_config.items():
        print(f"\n{'=' * 50}")
        print(f"训练 {model_name}")
        print(f"{'=' * 50}")

        # 创建分类器
        classifier = FruitClassifier(model, device, model_name)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 训练模型
        best_acc = classifier.train(train_loader, val_loader, criterion, optimizer, scheduler, epochs=5)

        # 最终评估
        val_loss, val_acc, val_predictions, val_true_labels = classifier.validate(val_loader, criterion)

        # 对训练集进行预测
        train_predictions, train_true_labels = classifier.predict(train_loader)

        # 提取特征用于t-SNE
        train_features, train_labels_feat = classifier.extract_features(train_loader)
        val_features, val_labels_feat = classifier.extract_features(val_loader)

        results[model_name] = {
            'classifier': classifier,
            'accuracy': val_acc,
            'train_predictions': train_predictions,
            'train_true_labels': train_true_labels,
            'val_predictions': val_predictions,
            'val_true_labels': val_true_labels,
            'best_accuracy': best_acc
        }

        # 绘制训练历史
        plot_training_history(classifier, model_name)

        # 保存并绘制训练集和验证集的混淆矩阵（百分比形式）
        train_cm, train_cm_percentage = save_confusion_matrix_data(
            train_true_labels, train_predictions,
            class_names[:num_classes], model_name, "训练集"
        )
        plot_confusion_matrix_percentage(
            train_cm_percentage, class_names[:num_classes],
            model_name, "训练集", cmap='viridis'
        )

        val_cm, val_cm_percentage = save_confusion_matrix_data(
            val_true_labels, val_predictions,
            class_names[:num_classes], model_name, "验证集"
        )
        plot_confusion_matrix_percentage(
            val_cm_percentage, class_names[:num_classes],
            model_name, "验证集", cmap='plasma'
        )

        # 计算并保存t-SNE数据
        train_features_2d, train_tsne_df = compute_tsne(
            train_features, train_labels_feat,
            class_names[:num_classes], model_name, "训练集"
        )
        plot_tsne_from_data(
            train_tsne_df, class_names[:num_classes],
            model_name, "训练集", cmap='viridis'
        )

        val_features_2d, val_tsne_df = compute_tsne(
            val_features, val_labels_feat,
            class_names[:num_classes], model_name, "验证集"
        )
        plot_tsne_from_data(
            val_tsne_df, class_names[:num_classes],
            model_name, "验证集", cmap='plasma'
        )

        print(f"{model_name} 最佳验证准确率: {best_acc:.2f}%")

    # 比较模型性能
    print(f"\n{'=' * 50}")
    print("模型性能比较")
    print(f"{'=' * 50}")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.2f}% (最佳: {result['best_accuracy']:.2f}%)")

    # 保存最佳模型
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_classifier = results[best_model_name]['classifier']
    print(f"\n最佳模型: {best_model_name}, 准确率: {results[best_model_name]['accuracy']:.2f}%")

    # 演示如何重新生成图表
    print(f"\n{'=' * 50}")
    print("重新生成图表演示")
    print(f"{'=' * 50}")

    # 使用不同的颜色映射重新生成混淆矩阵
    cmaps = ['Blues', 'viridis', 'plasma', 'coolwarm', 'YlOrRd']
    for cmap in cmaps[:2]:  # 只演示前两种
        vis_manager.regenerate_confusion_matrix(best_model_name, "验证集", cmap=cmap)

    # 使用不同的颜色映射重新生成t-SNE图
    cmaps_tsne = ['tab10', 'Set1', 'Set2', 'Set3', 'Dark2']
    for cmap in cmaps_tsne[:2]:  # 只演示前两种
        vis_manager.regenerate_tsne(best_model_name, "验证集", cmap=cmap)

    # 比较所有模型的混淆矩阵
    vis_manager.compare_all_models(list(models_config.keys()))


if __name__ == "__main__":
    main()