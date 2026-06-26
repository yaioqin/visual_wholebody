# metrics_tracker.py
import torch
import numpy as np
from collections import deque
import math
import os
import json
import csv
from datetime import datetime

class MetricsTracker:
    def __init__(self, log_interval=100, history_length=10000, save_dir=None):
        """
        初始化指标跟踪器
        :param log_interval: 每多少步输出一次日志
        :param history_length: 保存多少步的历史数据用于分析
        :param save_dir: 保存指标文件的目录，如果为None则不保存
        """
        self.log_interval = log_interval
        self.step_counter = 0
        self.save_dir = save_dir
        
        # 如果指定了保存目录，创建目录并初始化文件
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self._init_save_files()
        
        # 用于存储历史的缓冲区
        self.history = {
            'delta_p_ee': deque(maxlen=history_length),
            'A_ee': deque(maxlen=history_length),
            'eta_manip': deque(maxlen=history_length),
            'rho_align': deque(maxlen=history_length),
            'delta_p_ee_norm': deque(maxlen=history_length)
        }
        
        # 用于累计统计的变量
        self.running_stats = {
            'delta_p_ee': [],
            'A_ee': [],
            'eta_manip': [],
            'rho_align': [],
            'delta_p_ee_norm': []
        }
        
        # 用于保存完整历史记录的列表（每步都保存）
        self.full_history = {
            'step': [],
            'delta_p_ee_x': [],
            'delta_p_ee_y': [],
            'delta_p_ee_z': [],
            'delta_p_ee_norm': [],
            'A_ee': [],
            'eta_manip': [],
            'rho_align': []
        }
        
    def _init_save_files(self):
        """初始化保存文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV文件路径
        self.csv_file = os.path.join(self.save_dir, f"metrics_{timestamp}.csv")
        self.summary_file = os.path.join(self.save_dir, f"metrics_summary_{timestamp}.txt")
        
        # 创建CSV文件并写入表头
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step',
                'delta_p_ee_x', 'delta_p_ee_y', 'delta_p_ee_z',
                'delta_p_ee_norm',
                'A_ee',
                'eta_manip',
                'rho_align'
            ])
        
        print(f"Metrics will be saved to: {self.csv_file}")
    
    def compute_metrics(self, action_chunk, jacobian=None, object_pos_wrist=None, sigma_p=0.1):
        """计算5个物理量（与原代码相同）"""
        metrics = {}
        
        if torch.is_tensor(action_chunk):
            action_chunk = action_chunk.detach().cpu().numpy()
        
        if action_chunk is not None and len(action_chunk) > 0:
            delta_p = action_chunk[:, :3]
            metrics['delta_p_ee'] = np.mean(delta_p, axis=0)
            metrics['delta_p_ee_norm'] = np.sum(delta_p ** 2) / len(delta_p)
            metrics['A_ee'] = np.sum(np.sum(delta_p ** 2, axis=1))
        else:
            metrics['delta_p_ee'] = np.array([0.0, 0.0, 0.0])
            metrics['delta_p_ee_norm'] = 0.0
            metrics['A_ee'] = 0.0
        
        if jacobian is not None:
            if torch.is_tensor(jacobian):
                jacobian = jacobian.detach().cpu().numpy()
            try:
                if jacobian.shape[0] == 3:
                    jj_t = np.dot(jacobian, jacobian.T)
                    det = np.linalg.det(jj_t)
                    metrics['eta_manip'] = math.sqrt(max(0, det))
                else:
                    det = np.linalg.det(jacobian)
                    metrics['eta_manip'] = det
            except np.linalg.LinAlgError:
                metrics['eta_manip'] = 0.0
        else:
            metrics['eta_manip'] = 0.0
        
        if object_pos_wrist is not None:
            if torch.is_tensor(object_pos_wrist):
                object_pos_wrist = object_pos_wrist.detach().cpu().numpy()
            dist_sq = np.sum(object_pos_wrist ** 2)
            metrics['rho_align'] = math.exp(-dist_sq / (sigma_p ** 2))
        else:
            metrics['rho_align'] = 0.0
        
        return metrics
    
    def update(self, metrics):
        """更新跟踪器状态，并保存数据"""
        self.step_counter += 1
        
        # 存储到历史记录
        for key in self.history.keys():
            if key in metrics:
                self.history[key].append(metrics[key])
        
        # 累积统计信息
        for key in self.running_stats.keys():
            if key in metrics:
                self.running_stats[key].append(metrics[key])
        
        # 保存到完整历史（每步都保存）
        if 'delta_p_ee' in metrics:
            dp = metrics['delta_p_ee']
            self.full_history['step'].append(self.step_counter)
            self.full_history['delta_p_ee_x'].append(dp[0] if len(dp) > 0 else 0)
            self.full_history['delta_p_ee_y'].append(dp[1] if len(dp) > 1 else 0)
            self.full_history['delta_p_ee_z'].append(dp[2] if len(dp) > 2 else 0)
            self.full_history['delta_p_ee_norm'].append(metrics.get('delta_p_ee_norm', 0))
            self.full_history['A_ee'].append(metrics.get('A_ee', 0))
            self.full_history['eta_manip'].append(metrics.get('eta_manip', 0))
            self.full_history['rho_align'].append(metrics.get('rho_align', 0))
            
            # 每10步保存一次到CSV（避免IO开销太大）
            if self.step_counter % 10 == 0 and self.save_dir is not None:
                self._save_step_to_csv()
        
        # 每log_interval步输出一次
        if self.step_counter % self.log_interval == 0:
            self._log_metrics()
            # 保存summary
            if self.save_dir is not None:
                self._save_summary()
            # 清空当前累积的统计值
            for key in self.running_stats.keys():
                self.running_stats[key] = []
    
    def _save_step_to_csv(self):
        """保存单步数据到CSV"""
        if not hasattr(self, 'csv_file') or self.full_history['step'] == []:
            return
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # 获取最新的一步
            idx = -1
            writer.writerow([
                self.full_history['step'][idx],
                self.full_history['delta_p_ee_x'][idx],
                self.full_history['delta_p_ee_y'][idx],
                self.full_history['delta_p_ee_z'][idx],
                self.full_history['delta_p_ee_norm'][idx],
                self.full_history['A_ee'][idx],
                self.full_history['eta_manip'][idx],
                self.full_history['rho_align'][idx]
            ])
    
    def _log_metrics(self):
        """输出当前轮次的统计信息到终端（与原代码相同）"""
        print(f"\n[Step {self.step_counter}] Physical Metrics:")
        for key, values in self.running_stats.items():
            if values:
                avg_val = np.mean(values, axis=0)
                if isinstance(avg_val, np.ndarray):
                    if key == 'delta_p_ee':
                        print(f"  {key}: norm = {np.linalg.norm(avg_val):.4f}, "
                              f"vector = ({avg_val[0]:.4f}, {avg_val[1]:.4f}, {avg_val[2]:.4f})")
                    else:
                        print(f"  {key}: {np.linalg.norm(avg_val):.4f}")
                else:
                    print(f"  {key}: {avg_val:.4f}")
        print(f"  Effective sample count: {len(self.running_stats['delta_p_ee'])}")
        print("-" * 60)
    
    def _save_summary(self):
        """保存统计摘要到文件"""
        if not hasattr(self, 'summary_file'):
            return
        
        with open(self.summary_file, 'a') as f:
            f.write(f"\n[Step {self.step_counter}] Summary:\n")
            for key, values in self.running_stats.items():
                if values:
                    avg_val = np.mean(values, axis=0)
                    if isinstance(avg_val, np.ndarray):
                        if key == 'delta_p_ee':
                            f.write(f"  {key}: norm = {np.linalg.norm(avg_val):.4f}, "
                                    f"vector = ({avg_val[0]:.4f}, {avg_val[1]:.4f}, {avg_val[2]:.4f})\n")
                        else:
                            f.write(f"  {key}: {np.linalg.norm(avg_val):.4f}\n")
                    else:
                        f.write(f"  {key}: {avg_val:.4f}\n")
            f.write(f"  Effective sample count: {len(self.running_stats['delta_p_ee'])}\n")
            f.write("-" * 60 + "\n")
    
    def save_full_history(self):
        """训练结束后保存完整历史数据"""
        if self.save_dir is None:
            return
        
        # 保存为numpy格式
        npz_file = os.path.join(self.save_dir, "metrics_full_history.npz")
        np.savez(npz_file,
                 step=np.array(self.full_history['step']),
                 delta_p_ee_x=np.array(self.full_history['delta_p_ee_x']),
                 delta_p_ee_y=np.array(self.full_history['delta_p_ee_y']),
                 delta_p_ee_z=np.array(self.full_history['delta_p_ee_z']),
                 delta_p_ee_norm=np.array(self.full_history['delta_p_ee_norm']),
                 A_ee=np.array(self.full_history['A_ee']),
                 eta_manip=np.array(self.full_history['eta_manip']),
                 rho_align=np.array(self.full_history['rho_align'])
        )
        print(f"Full history saved to: {npz_file}")
    
    def get_recent_metrics(self, n=100):
        """获取最近n步的指标统计数据"""
        recent = {}
        for key in self.history.keys():
            if len(self.history[key]) > 0:
                recent[key] = list(self.history[key])[-n:]
        return recent