import logging
import re
from typing import Optional

import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from file_reader import FileReader

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    文本处理引擎，支持多语言文本摘要和文件处理
    """
    # 定义需要从文件名中移除的特殊字符集合
    # 包含操作系统不允许的字符及可能引起问题的符号
    FILENAME_SPECIAL_CHARS = r'[\/:*?"<>|%#$@!^&()\[\]{};`,.~+=。，？！]'

    # 定义语言前缀映射，用于多语言摘要生成
    LANGUAGE_PREFIXES = {
        "en": "summarize English: ",
        "zh": "总结中文: ",
        "fr": "résume en français: ",
        "es": "resumir en español: ",
        "de": "fasse zusammen auf Deutsch: ",
        "ru": "сделать резюме на русском: ",
        "ja": "日本語で要約してください: ",
        "ko": "한국어로 요약: ",
        "ar": "لخص بالعربية: ",
        "it": "riassumi in italiano: "
    }

    def __init__(self):
        """
        初始化文本处理器
        """
        logger.info("初始化文本处理器")

        # 模型相关组件
        self.model: Optional[MT5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5Tokenizer] = None
        self.device: Optional[str] = None

        # 文件处理组件
        self.file_reader = FileReader()

        # 编译正则表达式模式，提高性能
        self.special_chars_pattern = re.compile(self.FILENAME_SPECIAL_CHARS)
        self.consecutive_spaces_pattern = re.compile(r'[-\s]+')

        logger.info("文本处理器初始化完成")

    def load_model(self) -> None:
        """
        加载预训练的mT5模型和分词器
        如果模型已加载，则跳过此步骤
        """
        if self.model is not None and self.tokenizer is not None:
            logger.info("模型和分词器已加载，跳过重复加载")
            return

        logger.info("开始加载模型和分词器")
        model_path = "models/mt5-small"

        try:
            # 加载分词器
            logger.debug(f"从路径加载分词器: {model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
            logger.info("分词器加载成功")

            # 加载模型
            logger.debug(f"从路径加载模型: {model_path}")
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
            logger.info("模型加载成功")

            # 自动选择计算设备（GPU或CPU）
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"选择计算设备: {self.device}")

            # 将模型移至所选设备
            self.model.to(self.device)
            logger.info(f"模型已成功加载到设备: {self.device}")

        except Exception as error:
            logger.exception(f"模型加载失败: {str(error)}")
            raise RuntimeError(f"模型加载失败: {str(error)}")

    def clean_filename(self, text: str) -> str:
        """
        清理文本，移除或替换可能影响文件命名的特殊字符

        Args:
            text: 待清理的文本

        Returns:
            清理后的文本，可安全用于文件命名
        """
        logger.debug(f"清理文件名: 原始文本长度 {len(text)}")

        # 移除标记
        cleaned_text = re.sub(r'<[^>]+>', '', text)

        # 移除特殊字符
        cleaned_text = self.special_chars_pattern.sub('', cleaned_text)

        # 合并连续空格并去除首尾空格
        cleaned_text = self.consecutive_spaces_pattern.sub(' ', cleaned_text).strip()

        # 限制最大长度，防止文件名过长
        MAX_FILENAME_LENGTH = 200
        if len(cleaned_text) > MAX_FILENAME_LENGTH:
            logger.warning(f"文件名过长 ({len(cleaned_text)} 字符)，截断为 {MAX_FILENAME_LENGTH} 字符")
            cleaned_text = cleaned_text[:MAX_FILENAME_LENGTH].rstrip()

        logger.debug(f"清理后: 文本长度 {len(cleaned_text)}")
        return cleaned_text

    def generate_summary(self,
                         text: str,
                         max_length: int = 30,
                         min_length: int = 10,
                         language: str = "en") -> str:
        """
        使用mT5模型生成文本摘要

        Args:
            text: 待摘要的原始文本
            max_length: 摘要的最大长度（token数）
            min_length: 摘要的最小长度（token数）
            language: 文本语言代码（如"en"、"zh"等）

        Returns:
            生成的摘要文本
        """
        logger.info(f"开始文本摘要处理 (语言: {language}, 文本长度: {len(text)} 字符)")

        # 确保模型已加载
        self.load_model()

        # 获取语言特定的提示前缀
        summary_prefix = self.LANGUAGE_PREFIXES.get(language, self.LANGUAGE_PREFIXES["en"])
        logger.info(f"使用语言前缀: '{summary_prefix}'")

        # 构建带前缀的输入文本
        input_text = summary_prefix + text
        logger.debug(f"构建输入文本 (总长度: {len(input_text)} 字符)")

        try:
            # 编码输入文本
            logger.info("开始编码输入文本")
            input_encoding = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            logger.info(f"文本编码完成 (input_ids 形状: {input_encoding['input_ids'].shape})")

            # 配置生成参数
            generation_params = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": 4,  # 束搜索宽度
                "early_stopping": True,  # 达到最小长度后可提前停止
                "no_repeat_ngram_size": 3,  # 避免重复3-gram
                "length_penalty": 2.0,  # 倾向于生成中等长度的摘要
                "repetition_penalty": 1.5,  # 减少重复生成的惩罚因子
                "temperature": 0.7,  # 控制生成的随机性
                "do_sample": True  # 使用采样生成
            }
            logger.debug(f"摘要生成参数: {generation_params}")

            # 生成摘要
            logger.info("开始模型摘要生成")
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids=input_encoding["input_ids"],
                    attention_mask=input_encoding["attention_mask"],
                    **generation_params
                )
            logger.info("摘要生成完成")

            # 解码生成的摘要
            logger.info("开始解码生成的摘要")
            raw_summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            logger.debug(f"原始解码摘要: '{raw_summary}' (长度: {len(raw_summary)} 字符)")

            # 移除提示前缀（如果存在）
            if raw_summary.startswith(summary_prefix):
                logger.debug("检测到前缀，正在移除")
                raw_summary = raw_summary[len(summary_prefix):].strip()
                logger.debug(f"移除前缀后摘要: '{raw_summary}'")

            # 检查摘要质量
            word_count = len(raw_summary.split())
            logger.debug(f"生成的摘要词数: {word_count}")

            # 摘要过短的回退机制
            if word_count < 3:
                logger.warning(f"摘要过短 (仅 {word_count} 词)，启动回退机制")
                try:
                    # 使用简化参数重新生成
                    with torch.no_grad():
                        fallback_ids = self.model.generate(
                            input_ids=input_encoding["input_ids"],
                            attention_mask=input_encoding["attention_mask"],
                            max_length=max_length,
                            num_beams=1,  # 禁用束搜索，使用贪心解码
                            early_stopping=True
                        )

                    fallback_summary = self.tokenizer.decode(
                        fallback_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )

                    fallback_word_count = len(fallback_summary.split())
                    logger.debug(f"回退摘要词数: {fallback_word_count}")

                    if fallback_word_count > word_count:
                        logger.info(f"采用回退摘要 (长度: {fallback_word_count} 词)")
                        raw_summary = fallback_summary
                    else:
                        logger.warning("回退摘要未提供改进，保留原始摘要")

                except Exception as fallback_error:
                    logger.error(f"回退摘要生成失败: {str(fallback_error)}", exc_info=True)

            logger.info(f"摘要生成完成 (最终长度: {len(raw_summary)} 字符)")
            return raw_summary

        except Exception as error:
            logger.exception(f"摘要生成过程中发生错误: {str(error)}")
            raise RuntimeError(f"摘要生成失败: {str(error)}")

    def process_file(self, file_path: str, language: str = "en", **summary_kwargs) -> str:
        """
        完整的文件处理流程：读取文件内容并生成安全的文件名

        Args:
            file_path: 要处理的文件路径
            language: 文件内容的语言代码
            summary_kwargs: 传递给generate_summary的额外参数

        Returns:
            清理后的摘要文本，可用作安全的文件名
        """
        logger.info(f"开始处理文件: {file_path}")
        logger.debug(f"语言设置: {language}, 额外参数: {summary_kwargs}")

        try:
            # 读取文件内容
            logger.info(f"读取文件内容: {file_path}")
            file_content = self.file_reader.read_file(file_path)
            logger.info(f"文件读取成功 (内容长度: {len(file_content)} 字符)")

            # 生成摘要
            logger.info("开始生成摘要")
            raw_summary = self.generate_summary(
                text=file_content,
                language=language,
                **summary_kwargs
            )

            # 清理摘要文本，确保可以安全用作文件名
            logger.info("清理摘要文本，确保文件名安全")
            safe_filename = self.clean_filename(raw_summary)

            logger.info(f"文件处理完成: {file_path}")
            logger.info(f"生成安全文件名 (长度: {len(safe_filename)} 字符): {safe_filename[:50]}...")

            return safe_filename

        except Exception as error:
            logger.exception(f"文件处理失败: {file_path}, 错误: {str(error)}")
            raise RuntimeError(f"文件处理失败: {str(error)}")
