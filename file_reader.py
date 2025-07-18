import logging
import os
import re
from pathlib import Path

import olefile
from docx import Document

# 配置日志记录器
logger = logging.getLogger(__name__)


class FileReader:
    @staticmethod
    def _read_txt(file_path: str) -> str:
        """读取txt文件，自动检测编码"""
        logger.info(f"开始读取文本文件: {file_path}")
        encodings = ["utf-8", "gbk", "latin-1", "utf-16"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                    logger.info(f"成功读取文件，使用编码: {encoding}")
                    return content
            except UnicodeDecodeError as ude:
                logger.warning(f"编码 {encoding} 解码失败: {str(ude)}")
                continue

        raise ValueError(f"无法解码文件: {file_path}")

    @staticmethod
    def _read_docx(file_path: str) -> str:
        """读取docx格式的Word文档"""
        logger.info(f"开始读取DOCX文件: {file_path}")
        try:
            doc = Document(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as error:
            logger.exception(f"读取DOCX文件失败: {str(error)}")
            raise

    @staticmethod
    def _read_doc(file_path: str) -> str:
        """读取doc格式的Word文档"""
        logger.info(f"开始读取DOC文件: {file_path}")
        try:
            with olefile.OleFileIO(file_path) as ole:
                if not ole.exists('WordDocument'):
                    raise ValueError("文件不是有效的Word文档")

                doc_data = ole.openstream('WordDocument').read()
                try:
                    text = doc_data.decode('utf-16', errors='replace')
                except UnicodeDecodeError:
                    text = doc_data.decode('latin-1', errors='replace')

                text = re.sub(r'\x00', '', text)  # 移除NUL字符
                text = re.sub(r'\s+', ' ', text)  # 合并连续空格
                return text.strip()
        except Exception as error:
            logger.exception(f"读取DOC文件失败: {str(error)}")
            raise

    def read_file(self, file_path: str) -> str:
        """通用文件读取入口"""
        logger.info(f"开始读取文件: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = Path(file_path).suffix.lower()
        logger.debug(f"文件后缀: {suffix}")

        # 使用Python 3.12的match语句替代多分支if-else
        match suffix:
            case '.txt':
                logger.info("识别为文本文件，调用文本读取方法")
                return self._read_txt(file_path)
            case '.docx':
                logger.info("识别为Word文档(.docx)，调用DOCX读取方法")
                return self._read_docx(file_path)
            case '.doc':
                logger.info("识别为Word文档(.doc)，调用DOC读取方法")
                return self._read_doc(file_path)
            case _:
                supported_types = ['.txt', '.docx', '.doc']
                logger.error(f"不支持的文件类型: {suffix}，支持的类型: {supported_types}")
                raise ValueError(f"不支持的文件类型: {suffix}")
