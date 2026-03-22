import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的第三方库是否安装"""
    required_libs = {
        'pdfplumber': 'pdfplumber',
        'docx': 'python-docx'
    }
    missing_libs = []
    for lib, pkg in required_libs.items():
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(pkg)
    if missing_libs:
        print(f"❌ 缺少必要的库，请先运行以下命令安装：")
        print(f"   pip install {' '.join(missing_libs)}")
        sys.exit(1)

# 检查依赖后再导入
check_dependencies()
import pdfplumber
from docx import Document
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words

# ================= 核心工具模块 =================

def init_nltk_resources():
    """静默初始化NLTK资源"""
    resources = [
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('corpora/words', 'words')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"⏳ 正在下载 NLTK 资源: {name}...")
            nltk.download(name, quiet=True)

def get_wordnet_pos(nltk_tag):
    """NLTK词性转WordNet词性"""
    if nltk_tag.startswith('J'): return wordnet.ADJ
    elif nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('N'): return wordnet.NOUN
    elif nltk_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

# ================= 文件读取模块 =================

def read_txt(file_path):
    """读取TXT，自动适配编码"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f: return f.read()

def read_pdf(file_path):
    """读取PDF文本"""
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text: text_content.append(text)
    return '\n'.join(text_content)

def read_docx(file_path):
    """读取Word文档"""
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def load_file_text(file_path):
    """根据扩展名分发读取器"""
    ext = Path(file_path).suffix.lower()
    if ext == '.txt': return read_txt(file_path)
    elif ext == '.pdf': return read_pdf(file_path)
    elif ext == '.docx': return read_docx(file_path)
    else: raise ValueError(f"不支持的文件格式: {ext}")

# ================= 业务逻辑模块 =================

def extract_and_lemmatize(text):
    """流程：提取单词 -> 词性标注 -> 词形还原 -> 去重排序"""
    raw_words = re.findall(r'[a-zA-Z]+', text.lower())
    pos_tags = nltk.pos_tag(raw_words)
    
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    return sorted(set(lemmatized))

def filter_valid_words(word_list):
    """流程：基于NLTK词库过滤无效单词"""
    english_dict = set(w.lower() for w in words.words())
    valid = [w for w in word_list if w.lower() in english_dict]
    invalid = [w for w in word_list if w.lower() not in english_dict]
    return valid, invalid

def calculate_coverage_rate(dict_path, target_path):
    """计算覆盖率并返回统计数据"""
    def read_list(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    dict_set = set(read_list(dict_path))
    target_list = read_list(target_path)
    
    covered = [w for w in target_list if w in dict_set]
    uncovered = [w for w in target_list if w not in dict_set]
    
    return {
        "dict_count": len(dict_set),
        "target_count": len(target_list),
        "covered_count": len(covered),
        "uncovered_count": len(uncovered),
        "rate": (len(covered) / len(target_list) * 100) if target_list else 0,
        "uncovered_list": uncovered
    }

# ================= 工作区管理 =================

class Workspace:
    def __init__(self, source_file):
        self.source_path = Path(source_file)
        self.root_dir = self.source_path.parent / "WordTool_Workspace"
        self.source_dir = self.root_dir / "0_Source"
        self.result_dir = self.root_dir / "1_Result"
        
        self._init_dirs()
        self._copy_source()

    def _init_dirs(self):
        self.root_dir.mkdir(exist_ok=True)
        self.source_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)

    def _copy_source(self):
        shutil.copy2(self.source_path, self.source_dir / self.source_path.name)

    def save_result(self, filename, content_list):
        """保存结果文件到工作区"""
        file_path = self.result_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in content_list:
                f.write(f"{line}\n")
        return file_path

    def open_in_explorer(self):
        """跨平台打开结果文件夹"""
        if sys.platform == 'win32':
            os.startfile(self.result_dir)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', self.result_dir])
        else:
            subprocess.Popen(['xdg-open', self.result_dir])

# ================= 交互界面模块 =================

def prompt_file_path(desc="文件"):
    """提示输入文件路径"""
    while True:
        path = input(f"请拖入{desc}或输入路径: ").strip().strip('"').strip("'")
        if os.path.isfile(path): return path
        print("❌ 文件不存在，请重试。")

def prompt_yes_no(question, default="y"):
    """Yes/No询问"""
    hint = "[Y/n]" if default.lower() == "y" else "[y/N]"
    while True:
        choice = input(f"{question} {hint}: ").strip().lower()
        if not choice: return default.lower() == "y"
        if choice in ['y', 'yes']: return True
        if choice in ['n', 'no']: return False
        print("请输入 y 或 n。")

def run_text_processing():
    """流程：文本处理"""
    print("\n" + "="*40)
    print("📝 板块一：文本处理")
    print("="*40)
    
    f_path = prompt_file_path("源文件(.txt/.pdf/.docx)")
    
    print("\n处理选项：")
    do_extract = prompt_yes_no("1. 执行选词(提取/还原/去重)？", "y")
    do_filter = prompt_yes_no("2. 执行词典筛词(验证有效性)？", "y")

    if not do_extract and not do_filter:
        print("⚠️ 未选择任何操作，返回。")
        return

    try:
        ws = Workspace(f_path)
        text = load_file_text(ws.source_dir / Path(f_path).name)
        
        current_data = None
        
        if do_extract:
            print("⏳ 正在进行词形还原与去重...")
            current_data = extract_and_lemmatize(text)
            ws.save_result("1_Lemmatized_Words.txt", current_data)
            print(f"✅ 选词完成，共 {len(current_data)} 个词。")

        if do_filter:
            print("⏳ 正在进行词典校验...")
            if current_data is None:
                # 如果没选词，只做简单提取
                current_data = sorted(set(re.findall(r'[a-zA-Z]+', text.lower())))
            
            valid, invalid = filter_valid_words(current_data)
            ws.save_result("2_Valid_Words.txt", valid)
            ws.save_result("3_Invalid_Words.txt", invalid)
            print(f"✅ 筛词完成：有效 {len(valid)}，无效 {len(invalid)}。")

        print("\n🎉 处理完毕！正在打开结果文件夹...")
        ws.open_in_explorer()

    except Exception as e:
        print(f"❌ 发生错误: {e}")

def run_coverage_analysis():
    """流程：覆盖率分析"""
    print("\n" + "="*40)
    print("📊 板块二：覆盖率计算")
    print("="*40)
    
    print("> 请准备两个TXT文件，每行一个单词。")
    dict_path = prompt_file_path("【词典】文件")
    target_path = prompt_file_path("【目标】文件")

    try:
        ws = Workspace(target_path)
        # 复制词典到工作区
        shutil.copy2(dict_path, ws.source_dir / "Dictionary.txt")
        
        print("⏳ 正在计算覆盖率...")
        stats = calculate_coverage_rate(dict_path, target_path)
        
        # 保存报告
        report = [
            "="*30,
            "📊 覆盖率统计报告",
            "="*30,
            f"词典词数: {stats['dict_count']}",
            f"目标词数: {stats['target_count']}",
            f"覆盖词数: {stats['covered_count']}",
            f"未覆盖词数: {stats['uncovered_count']}",
            f"覆盖率: {stats['rate']:.2f}%",
            "="*30
        ]
        
        ws.save_result("Coverage_Report.txt", report)
        ws.save_result("Uncovered_Words.txt", stats['uncovered_list'])
        
        print("\n" + "\n".join(report))
        print("\n🎉 统计完毕！正在打开结果文件夹...")
        ws.open_in_explorer()

    except Exception as e:
        print(f"❌ 发生错误: {e}")

def main():
    init_nltk_resources()
    while True:
        print("\n" + "="*40)
        print("英语单词处理工具箱")
        print("="*40)
        print("1. 📝 文本处理 (选词/筛词)")
        print("2. 📊 覆盖率计算")
        print("0. Exit")
        print("-" * 40)
        
        choice = input("请选择功能 [0-2]: ").strip()
        
        if choice == '1': run_text_processing()
        elif choice == '2': run_coverage_analysis()
        elif choice == '0': break
        else: print("无效输入，请重试。")

if __name__ == "__main__":
    main()
