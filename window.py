import gc
import logging
import os
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QMessageBox, QListWidgetItem
)

from UI.default import Ui_MainWindow
from processor import TextProcessor

# 配置日志系统 - 同时输出到文件和控制台
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log/log.txt",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'
)

# 添加控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# 记录应用启动信息
logger.info("=" * 50)
logger.info("应用程序启动")
logger.info(f"Python版本: {sys.version}")
logger.info(f"工作目录: {os.getcwd()}")
logger.info(f"系统路径: {sys.path}")


def handle_drag_enter(event):
    """处理拖放进入事件 - 检查是否包含有效文件URL"""
    logger.debug("拖放进入事件触发")
    if event.mimeData().hasUrls():
        logger.info("拖放内容包含URL，接受操作")
        event.acceptProposedAction()
    else:
        logger.debug("拖放内容不包含URL，忽略操作")


class FileProcessingThread(QThread):
    """
    后台文件处理线程
    负责异步处理文件，避免阻塞UI线程
    """
    # 信号定义
    progress_updated = pyqtSignal(int, str, str)  # 进度更新 (当前进度, 文件名, 结果)
    processing_completed = pyqtSignal(int, int)  # 处理完成 (成功数, 失败数)

    def __init__(self, file_paths):
        """
        初始化文件处理线程

        Args:
            file_paths: 待处理的文件路径列表
        """
        super().__init__()
        self.file_paths = file_paths
        logger.info(f"创建文件处理线程，待处理文件数: {len(file_paths)}")

    def run(self):
        """线程主执行逻辑"""
        logger.info("文件处理线程启动")
        total_files = len(self.file_paths)
        success_count = 0

        for index, file_path in enumerate(self.file_paths):
            logger.debug(f"开始处理文件 #{index + 1}/{total_files}: {file_path}")
            filename = os.path.basename(file_path)

            try:
                # 初始化处理器并处理文件
                processor = TextProcessor()
                new_name = processor.process_file(file_path)

                # 构建新文件名（添加原始文件后缀）
                file_ext = Path(file_path).suffix.lower()
                new_name_with_ext = new_name + file_ext
                logger.info(f"文件处理成功: {filename} -> {new_name_with_ext}")

                # 构建新文件路径
                file_dir = os.path.dirname(file_path)
                new_file_path = os.path.join(file_dir, new_name_with_ext)

                # 检查文件是否已存在
                if os.path.exists(new_file_path):
                    raise FileExistsError(f"文件 {new_name_with_ext} 已存在")

                # 执行文件重命名
                os.rename(file_path, new_file_path)
                success_count += 1
                logger.debug(f"文件重命名完成: {file_path} -> {new_file_path}")

                # 发送进度更新信号
                self.progress_updated.emit(index + 1, filename, new_name_with_ext)

            except Exception as e:
                logger.error(f"处理文件 {filename} 失败: {str(e)}")
                logger.debug(f"错误详情:\n{traceback.format_exc()}")
                error_msg = f"错误: {str(e)}"
                self.progress_updated.emit(index + 1, filename, error_msg)

            # 每处理5个文件执行一次垃圾回收
            if index % 5 == 0 and index > 0:
                logger.debug("触发垃圾回收")
                gc.collect()

        # 发送处理完成信号
        failed_count = total_files - success_count
        logger.info(f"文件处理完成: 成功 {success_count} 个, 失败 {failed_count} 个")
        self.processing_completed.emit(success_count, failed_count)


class SummlyApp(QMainWindow):
    """
    主应用窗口类
    负责UI交互和业务逻辑协调
    """

    def __init__(self):
        """初始化主窗口"""
        logger.info("初始化主应用窗口")
        super().__init__()

        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        logger.info("UI设置完成")

        # 初始化内部状态
        self.pending_files = []  # 待处理的文件路径列表
        self.processing_thread = None  # 当前处理线程
        self.ui.processingFileList.addItem('')
        self.ui.processingFileList.addItem('')
        self.ui.processingFileList.addItem('')
        self.ui.processingFileList.addItem('')

        # 配置拖放功能
        self._setup_drag_drop()

        # 连接UI事件
        self._connect_ui_signals()

        # 初始化UI显示
        self._init_ui_display()

        logger.info("主窗口初始化完成")

    def _setup_drag_drop(self):
        """配置拖放功能"""
        logger.debug("配置文件拖放功能")
        self.ui.fileDropArea.setAcceptDrops(True)
        self.ui.fileDropArea.dragEnterEvent = handle_drag_enter
        self.ui.fileDropArea.dropEvent = self._handle_drop_event
        self.ui.fileDropArea.mousePressEvent = self._browse_files
        logger.info("拖放功能配置完成")

    def _connect_ui_signals(self):
        """连接UI组件信号到处理函数"""
        logger.debug("连接UI信号")
        self.ui.startProcessButton.clicked.connect(self._start_file_processing)
        logger.info("UI信号连接完成")

    def _init_ui_display(self):
        """初始化UI显示状态"""
        logger.debug("初始化UI显示")
        # 添加占位符项目（可选）
        # self.ui.processingFileList.addItems([''] * 4)
        logger.info("UI显示初始化完成")

    def _handle_drop_event(self, event):
        """
        处理文件拖放事件
        Args:
            event: 拖放事件对象
        """
        logger.info("处理文件拖放事件")
        urls = event.mimeData().urls()
        local_files = [url.toLocalFile() for url in urls]
        self._add_files(local_files)

    def _browse_files(self, event):
        """
        处理鼠标点击浏览文件事件
        Args:
            event: 鼠标事件对象
        """
        logger.debug("处理文件浏览请求")
        if event.button() == Qt.MouseButton.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(self, "选择文件")
            if files:
                logger.info(f"用户选择了 {len(files)} 个文件")
                self._add_files(files)
            else:
                logger.info("用户取消了文件选择")

    def _add_files(self, file_paths):
        """
        添加文件到处理队列
        Args:
            file_paths: 文件路径列表
        """
        logger.info(f"尝试添加 {len(file_paths)} 个文件到处理队列")

        # 过滤有效文件并去重
        existing_files = set(self.pending_files)
        valid_new_files = [
            f for f in file_paths
            if f not in existing_files and os.path.isfile(f)
        ]

        if not valid_new_files:
            logger.warning("没有找到有效的新文件")
            return

        # 添加文件到队列并更新UI
        for file_path in valid_new_files:
            self.pending_files.append(file_path)
            file_name = os.path.basename(file_path)

            # 更新左侧文件列表
            self.ui.processingFileList.addItem(file_name)

            # 更新右侧处理结果列表（初始状态）
            pending_item = QListWidgetItem("等待处理...")
            pending_item.setForeground(Qt.GlobalColor.gray)
            self.ui.processLogList.addItem(pending_item)

        logger.info(f"成功添加 {len(valid_new_files)} 个文件")

    def _start_file_processing(self):
        """开始文件处理流程"""
        logger.info("开始文件处理流程")

        # 检查是否有文件待处理
        if not self.pending_files:
            logger.warning("文件处理请求被拒绝：没有待处理的文件")
            QMessageBox.warning(self, "警告", "请先添加要处理的文件！")
            return

        # 检查是否已有处理线程在运行
        if self.processing_thread and self.processing_thread.isRunning():
            logger.warning("文件处理请求被拒绝：已有处理线程在运行")
            QMessageBox.warning(self, "警告", "当前有任务正在运行！")
            return

        # 准备处理
        file_count = len(self.pending_files)
        logger.info(f"开始处理 {file_count} 个文件")

        # 更新UI状态
        self.ui.startProcessButton.setEnabled(False)
        self.ui.processProgressBar.setRange(0, file_count)
        self.ui.processProgressBar.setValue(0)

        # 更新处理状态
        for i in range(self.ui.processLogList.count()):
            item = self.ui.processLogList.item(i)
            if item.text().startswith("错误:") or item.text() == "等待处理...":
                item.setText("处理中...")
                item.setForeground(Qt.GlobalColor.gray)

        # 创建并启动处理线程
        self.processing_thread = FileProcessingThread(self.pending_files.copy())
        self.processing_thread.progress_updated.connect(self._update_processing_progress)
        self.processing_thread.processing_completed.connect(self._handle_processing_finished)
        self.processing_thread.start()
        logger.info("文件处理线程已启动")

    def _update_processing_progress(self, progress, file_name, result):
        """
        更新处理进度显示
        Args:
            progress: 当前进度值
            file_name: 文件名
            result: 处理结果文本
        """
        logger.debug(f"更新处理进度: {progress}/{self.ui.processProgressBar.maximum()}, 文件: {file_name}")

        # 更新进度条
        self.ui.processProgressBar.setValue(progress)

        # 更新结果列表
        if 0 <= progress - 1 < self.ui.processLogList.count():
            result_item = self.ui.processLogList.item(progress - 1)
            result_item.setText(result)

            # 根据结果设置文本颜色
            if result.startswith("错误:"):
                result_item.setForeground(Qt.GlobalColor.red)
            else:
                result_item.setForeground(Qt.GlobalColor.white)
        else:
            logger.warning(f"进度更新失败：索引 {progress - 1} 超出列表范围")

    def _handle_processing_finished(self, success_count, failure_count):
        """
        处理完成后的清理工作
        Args:
            success_count: 成功处理的文件数
            failure_count: 处理失败的文件数
        """
        logger.info(f"文件处理完成: 成功 {success_count}, 失败 {failure_count}")

        # 更新UI状态
        self.ui.startProcessButton.setEnabled(True)

        # 显示结果消息
        result_msg = f"处理完成！\n成功: {success_count} 个\n失败: {failure_count} 个"
        logger.info(result_msg)
        QMessageBox.information(self, "处理完成", result_msg)

        # 清理资源
        self.processing_thread = None
        gc.collect()
        logger.info("文件处理资源清理完成")


if __name__ == "__main__":
    """应用程序入口点"""
    logger.info("启动应用程序")
    try:
        # 创建应用实例
        app = QApplication(sys.argv)

        # 创建并显示主窗口
        window = SummlyApp()
        window.show()

        # 启动应用事件循环
        exit_code = app.exec()
        logger.info(f"应用程序正常退出，退出代码: {exit_code}")

    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}")
        logger.debug(f"崩溃详情:\n{traceback.format_exc()}")
        QMessageBox.critical(None, "致命错误", f"应用程序发生致命错误:\n{str(e)}")
        exit_code = 1

    finally:
        logger.info("应用程序已终止")
        sys.exit(exit_code)
