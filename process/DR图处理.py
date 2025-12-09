import numpy as np
# 在程序开头添加
import matplotlib
matplotlib.use('TkAgg')  # 明确指定使用Tkinter后端
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))
import re
import pandas as pd
from scipy.signal.windows import taylor
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog

class RadarProcessor:
    def __init__(self):
        self.stop_flag = False
        self.Fs = 20e6  # 采样率 (20 MHz)
        self.delta_R = 3e8 / 2 / self.Fs  # 距离分辨率
        
        # 初始化列索引
        self.ColPoint, self.ColTrack = self.funcColIndex()
        
        # 初始化图形
        self.mainFig = None
        self.trackPlot = None
        self.pointPlot = None
        self.mtdImage = None
        self.targetPlot = None
        
    def funcColIndex(self):
        """数据列索引"""
        class ColPoint:
            Time = 0      # 点时间
            TrackID = 1    # 航迹批号
            R = 2          # 距离
            AZ = 3         # 方位
            EL = 4         # 俯仰
            Doppler = 5    # 多普勒速度
            Amp = 6        # 和幅度
            SNR = 7        # 信噪比
            PointNum = 8   # 原始点数量

        class ColTrack:
            Time = 0       # 点时间
            TrackID = 1    # 航迹批号
            R = 2          # 滤波距离
            AZ = 3         # 滤波方位
            EL = 4         # 滤波俯仰
            Speed = 5      # 全速度
            Vx = 6         # X向速度(东)
            Vy = 7         # Y向速度(北)
            Vz = 8         # Z向速度(天)
            Head = 9       # 航向角

        return ColPoint(), ColTrack()
    
    def funcReadData(self):
        """读取数据"""
        root = tk.Tk()
        root.withdraw()
        
        # 选择数据根目录
        selectedDir = filedialog.askdirectory(title='选择数据根目录，包含原始回波、点迹、航迹文件夹')
        if not selectedDir:
            raise ValueError("错误！未选择文件路径。")
            
        IQDataDir = os.path.join(selectedDir, '原始回波')
        TrackDir = os.path.join(selectedDir, '航迹')
        PointDir = os.path.join(selectedDir, '点迹')
        #print(PointDir)
        if not all([os.path.exists(IQDataDir), os.path.exists(TrackDir), os.path.exists(PointDir)]):
            raise ValueError('错误！未选择正确路径，根目录下需包含原始回波、点迹、航迹文件夹。')
            
        # 选择原始回波文件
        rawDataFile = filedialog.askopenfilename(
            title='选择数据',
            initialdir=IQDataDir,
            filetypes=[('DAT files', '*.dat')]
        )
        if not rawDataFile:
            raise ValueError("未选择原始回波文件")
            
        fid_rawData = open(rawDataFile, 'rb')
        
        # 从文件名提取批号和标签
        fileName = os.path.basename(rawDataFile)
        match = re.match(r'^(\d+)_Label_(\d+)\.dat$', fileName)
        if not match:
            raise ValueError("文件名格式不正确")
            
        track_No = int(match.group(1))
        label = int(match.group(2))
        # 读取点迹文件
        pointPattern = f'PointTracks_{track_No}_{label}_\d+\.txt'
        pointFiles = [f for f in os.listdir(PointDir) if re.match(pointPattern, f)]
        if not pointFiles:
            raise ValueError(f"点迹文件不存在: {pointPattern}")
            
        pointFile = os.path.join(PointDir, pointFiles[0])
        pointData = pd.read_csv(pointFile, header=None,sep=',',encoding='ansi')
        
        # 读取航迹文件
        trackPattern = f'Tracks_{track_No}_{label}_\d+\.txt'
        trackFiles = [f for f in os.listdir(TrackDir) if re.match(trackPattern, f)]
        if not trackFiles:
            raise ValueError(f"航迹文件不存在: {trackPattern}")
            
        trackFile = os.path.join(TrackDir, trackFiles[0])
        trackData = pd.read_csv(trackFile, header=None,sep=',',encoding='ansi')
        
        print("已读取输入数据。")
        return fid_rawData, pointData, trackData
    
    def funcCreateFigure(self, trackData):
        """创建图窗"""
        self.mainFig = plt.figure(figsize=(14, 7))
        self.mainFig.canvas.manager.set_window_title('RDMapParser')
        
        # 创建按钮
        ax_pause = plt.axes([0.05, 0.01, 0.1, 0.05])
        btn_pause = Button(ax_pause, '暂停')
        btn_pause.on_clicked(self.on_pause)
        
        ax_resume = plt.axes([0.16, 0.01, 0.1, 0.05])
        btn_resume = Button(ax_resume, '继续')
        btn_resume.on_clicked(self.on_resume)
        
        ax_stop = plt.axes([0.27, 0.01, 0.1, 0.05])
        btn_stop = Button(ax_stop, '停止')
        btn_stop.on_clicked(self.on_stop)
        '''
        ###########################################
        print("trackData 形状:", trackData.shape)  # 查看行列数
        print("trackData 前5行:\n", trackData.head())  # 查看数据内容
        print("self.ColTrack.AZ 的值:", self.ColTrack.AZ)  # 检查列索引
        #############################################
        '''
        # 创建航迹显示区域
        ax1 = self.mainFig.add_subplot(1, 2, 1)
        
        self.trackPlot, = ax1.plot(trackData.iloc[:, self.ColTrack.AZ], 
                                  trackData.iloc[:, self.ColTrack.R], '*-')
        
        ax1.grid(True)
        ax1.set_xlabel("方位(度)")
        ax1.set_ylabel("距离(米)")
        ax1.set_title("目标航迹")
        self.pointPlot, = ax1.plot([], [], 'ro')  # 用于更新当前点
        
        # 创建MTD结果显示区域
        ax2 = self.mainFig.add_subplot(1, 2, 2)
        self.mtdImage = ax2.imshow(np.zeros((1, 1)), aspect='auto')
        plt.colorbar(self.mtdImage, ax=ax2)
        ax2.set_xlabel("多普勒速度(米/秒)")
        ax2.set_ylabel("距离(米)")
        ax2.set_title("MTD处理结果")
        self.targetPlot, = ax2.plot([], [], 'ro', markersize=8)
        
        plt.tight_layout()
        return self.mainFig
    
    def on_pause(self, event):
        plt.pause(0.1)  # 简单暂停
    
    def on_resume(self, event):
        plt.pause(0.1)  # 简单继续
    
    def on_stop(self, event):
        self.stop_flag = True
    
    def funcRawDataParser(self, fid):
        """读取解析原始回波数据"""
        frame_head = 0xFA55FA55
        frame_end = 0x55FA55FA
        
        # 读取帧头
        head_bytes = fid.read(4)
        if not head_bytes:
            return None, None
            
        head_find = int.from_bytes(head_bytes, byteorder='little', signed=False)
        
        # 查找帧头
        while head_find != frame_head and not self.stop_flag:
            fid.seek(-3, 1)  # 回退3字节
            head_bytes = fid.read(4)
            if not head_bytes:
                return None, None
            head_find = int.from_bytes(head_bytes, byteorder='little', signed=False)
        
        # 读取帧长度
        frame_length_bytes = fid.read(4)
        if not frame_length_bytes:
            return None, None
        frame_data_length = int.from_bytes(frame_length_bytes, byteorder='little', signed=False) * 4
        
        # 跳过到帧尾
        fid.seek(frame_data_length - 12, 1)
        end_bytes = fid.read(4)
        if not end_bytes:
            return None, None
        end_find = int.from_bytes(end_bytes, byteorder='little', signed=False)
        
        # 验证帧结构
        while (head_find != frame_head) or (end_find != frame_end):
            fid.seek(-frame_data_length + 1, 1)
            
            head_bytes = fid.read(4)
            if not head_bytes:
                print('未找到满足报文格式的数据')
                return None, None
            head_find = int.from_bytes(head_bytes, byteorder='little', signed=False)
            
            frame_length_bytes = fid.read(4)
            if not frame_length_bytes:
                print('未找到满足报文格式的数据')
                return None, None
            frame_data_length = int.from_bytes(frame_length_bytes, byteorder='little', signed=False) * 4
            
            fid.seek(frame_data_length - 8, 1)
            end_bytes = fid.read(4)
            if not end_bytes:
                print('未找到满足报文格式的数据')
                return None, None
            end_find = int.from_bytes(end_bytes, byteorder='little', signed=False)
            
            if self.stop_flag:
                return None, None
        
        # 回退到数据开始位置
        fid.seek(-frame_data_length + 4, 1)
        
        # 读取参数
        para = {}
        data_temp1 = np.fromfile(fid, dtype=np.uint32, count=3)
        para['E_scan_Az'] = data_temp1[1] * 0.01
        pointNum_in_bowei = data_temp1[2]
        
        # 读取航迹信息
        para['Track_No_info'] = np.fromfile(fid, dtype=np.uint32, count=pointNum_in_bowei * 4)
        
        # 读取其他参数
        other_params = np.fromfile(fid, dtype=np.uint32, count=5)
        para['Freq'] = other_params[0] * 1e6  # 频率，单位MHz
        para['CPIcount'] = other_params[1]     # CPI流水号
        para['PRTnum'] = other_params[2]       # 当前CPI内PRT数目
        para['PRT'] = other_params[3] * 0.0125e-6  # PRT时间
        para['data_length'] = other_params[4]  # 距离维采样点数
        
        # 读取数据
        data_out_temp = np.fromfile(fid, dtype=np.float32, count=para['PRTnum'] * 31 * 2)
        if len(data_out_temp) == 0:
            return None, None
            
        data_out_real = data_out_temp[::2]
        data_out_imag = data_out_temp[1::2]
        data_out_complex = data_out_real + 1j * data_out_imag
        data_out = data_out_complex.reshape(31, para['PRTnum']).T  # PRT数×距离
        
        # 跳过帧尾
        fid.seek(4, 1)
        
        return para, data_out
    
    def process(self):
        """主处理函数"""
        # 读取数据
        fid_rawData, pointData, trackData = self.funcReadData()
        
        # 创建图形
        self.funcCreateFigure(trackData)
        
        # 主循环
        while not self.stop_flag:
            para, data_out = self.funcRawDataParser(fid_rawData)
            
            if para is None or data_out is None:
                break
                
            # MTD处理
            MTD_win = taylor(data_out.shape[1], nbar=5, sll=30, norm=False)
            coef_MTD_2D = np.tile(MTD_win, (data_out.shape[0], 1))
            
            data_proc_MTD_win_out = data_out * coef_MTD_2D
            data_proc_MTD_result = np.fft.fftshift(np.fft.fft(data_proc_MTD_win_out, axis=1), axes=1)
            
            # 计算速度轴
            delta_Vr = 3e8 / (2 * data_out.shape[1] * para['PRT'] * para['Freq'])
            Vr = np.arange(-data_out.shape[1]//2, data_out.shape[1]//2) * delta_Vr
            
            # 目标检测
            Amp_max_Vr_unit = para['Track_No_info'][3]
            Amp_max_Vr_unit = np.where(
                Amp_max_Vr_unit > data_out.shape[1]//2,
                Amp_max_Vr_unit - data_out.shape[1]//2,
                Amp_max_Vr_unit + data_out.shape[1]//2
            )
            
            # 在截取的数据中，目标中心位于第16个距离单元
            center_local_bin = 16
            local_radius = 5
            
            # 计算局部检测窗口
            range_start_local = max(0, center_local_bin - local_radius - 1)
            range_end_local = min(data_out.shape[0], center_local_bin + local_radius)
            doppler_start = max(0, int(Amp_max_Vr_unit) - local_radius - 1)
            doppler_end = min(data_out.shape[1], int(Amp_max_Vr_unit) + local_radius)
            
            Target_sig = data_proc_MTD_result[range_start_local:range_end_local, 
                                            doppler_start:doppler_end]
            
            # 检测峰值
            max_val = np.max(np.abs(Target_sig))
            Amp_max_index_row, Amp_max_index_col = np.where(np.abs(Target_sig) == max_val)
            Amp_max_index_row = Amp_max_index_row[0]
            Amp_max_index_col = Amp_max_index_col[0]
            
            # 获取目标全局距离单元索引
            global_range_bin = para['Track_No_info'][2]
            
            # 计算实际距离范围（目标距离单元±15）
            range_start_bin = global_range_bin - 15  # 截取起始距离单元
            range_end_bin = global_range_bin + 15    # 截取结束距离单元
            
            # 计算真实距离轴
            Range_plot = np.arange(range_start_bin, range_end_bin + 1) * self.delta_R
            
            # 转换到全局距离位置
            detected_range_bin = range_start_local + Amp_max_index_row
            Amp_max_range = Range_plot[detected_range_bin]
            Amp_max_Vr = Vr[doppler_start + Amp_max_index_col]
            
            # 确定航迹点序号
            index_trackPointNo = min(para['Track_No_info'][1], len(trackData) - 1)
            
            # 更新航迹图
            self.pointPlot.set_data(
                trackData.iloc[index_trackPointNo, self.ColTrack.AZ],
                trackData.iloc[index_trackPointNo, self.ColTrack.R]
            )
            
            # 更新MTD结果图
            self.mtdImage.set_data(20 * np.log10(np.abs(data_proc_MTD_result) + 1e-10))
            self.mtdImage.set_extent([Vr[0], Vr[-1], Range_plot[-1], Range_plot[0]])
            self.mtdImage.autoscale()
            
            # 更新目标标记
            self.targetPlot.set_data([Amp_max_Vr], [Amp_max_range])
            
            # 设置坐标轴范围
            ax2 = self.mainFig.axes[1]
            ax2.set_xlim(-30, 30)
            ax2.set_ylim(Range_plot[0], Range_plot[-1])
            '''
            # 更新标题
            ax2.set_title(
                f"点序号: {index_trackPointNo}, "
                f"距离: {trackData.iloc[index_trackPointNo, self.ColTrack.R]:.1f} m, "
                f"多普勒速度: {pointData.iloc[index_trackPointNo, self.ColPoint.Doppler]:.2f} m/s"
            )
            '''
            ax2.set_title(
    "点序号: {}, 距离: {:.1f} m, 多普勒速度: {:.2f} m/s".format(
        int(index_trackPointNo),  # 确保为整数
        float(trackData.iloc[index_trackPointNo, self.ColTrack.R]),  # 转为浮点数
        float(pointData.iloc[index_trackPointNo, self.ColPoint.Doppler])  # 转为浮点数
    )
)
            
            plt.pause(0.5)
            
        print("已完成解析。")
        fid_rawData.close()
        plt.show()

if __name__ == "__main__":
    processor = RadarProcessor()
    processor.process()