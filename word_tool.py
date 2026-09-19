import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path


def check_dependencies():
    """检查必要的第三方库是否安装（spaCy 按需加载）"""
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
        print("[错误] 缺少必要的库，请先运行以下命令安装：")
        print(f"   pip install {' '.join(missing_libs)}")
        sys.exit(1)


# 检查依赖后再导入（spaCy 不在此处导入）
check_dependencies()
import pdfplumber
from docx import Document
import nltk
from nltk.corpus import words

# ================= 核心工具模块 =================

def init_nltk_resources():
    """静默初始化NLTK资源（NLTK 用于词典筛词校验，词形还原使用 spaCy）"""
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("[下载] 正在下载 NLTK 资源: words...")
        nltk.download('words', quiet=True)


def clear_screen():
    """跨平台清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def pause_return():
    """清屏返回主菜单"""
    try:
        input("\n按回车键返回主菜单...")
    except EOFError:
        pass

# ================= 终端格式化工具 =================

def display_width(text):
    """计算字符串在终端的显示宽度（全角字符按2列计）"""
    return sum(2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1
               for ch in str(text))


def pad(text, width, align='left'):
    """按显示宽度对齐补空格"""
    text = str(text)
    gap = max(0, width - display_width(text))
    if align == 'right':
        return ' ' * gap + text
    if align == 'center':
        left = gap // 2
        return ' ' * left + text + ' ' * (gap - left)
    return text + ' ' * gap


def shorten(text, limit):
    """按显示宽度截断字符串，超出部分显示为 ..."""
    text = str(text)
    if display_width(text) <= limit:
        return text
    out = ''
    for ch in text:
        if display_width(out + ch) > limit - 3:
            break
        out += ch
    return out + '...'


def wrap_words(word_list, width):
    """把单词列表按显示宽度折行"""
    lines, cur = [], ''
    for w in word_list:
        item = f"{cur}, {w}" if cur else w
        if cur and display_width(item) > width:
            lines.append(cur)
            cur = w
        else:
            cur = item
    if cur:
        lines.append(cur)
    return lines


def format_table(headers, rows, aligns):
    """构建自动对齐的表格，返回 (行列表, 表格总宽度)"""
    n = len(headers)
    widths = [display_width(h) for h in headers]
    for row in rows:
        for i in range(n):
            widths[i] = max(widths[i], display_width(row[i]))

    def render(cells, cell_aligns):
        return '  '.join(pad(cells[i], widths[i], cell_aligns[i]) for i in range(n))

    total = sum(widths) + 2 * (n - 1)
    lines = [render(headers, ['center'] * n), '-' * total]
    for row in rows:
        lines.append(render(row, aligns))
    return lines, total


def fmt_pct(value):
    return f"{value:.2f}%"


def fmt_cost(stats):
    """均摊成本 = 词书词数 / 命中词数"""
    if stats["covered_count"] == 0:
        return "-"
    return f"{stats['dict_count'] / stats['covered_count']:.2f}"

# ================= spaCy 词形还原模块 =================

_SPACY_NLP = None

def get_spacy_nlp():
    """获取(并缓存) spaCy 英文模型；缺失时提供自动安装"""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
    except ImportError:
        print("[提示] 词形还原需要 spaCy，当前未安装。")
        if not prompt_yes_no("是否现在自动安装 spaCy？", "y"):
            raise RuntimeError("spaCy 未安装，请手动运行: pip install spacy")
        print("[安装] 正在安装 spaCy，可能需要几分钟，请稍候...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spacy'])
        except subprocess.CalledProcessError:
            raise RuntimeError("spaCy 安装失败，请手动运行: pip install spacy")
        import spacy

    if not spacy.util.is_package('en_core_web_sm'):
        print("[提示] 缺少 spaCy 英文模型 en_core_web_sm。")
        if not prompt_yes_no("是否现在自动下载模型？", "y"):
            raise RuntimeError("缺少模型，请手动运行: python -m spacy download en_core_web_sm")
        print("[下载] 正在下载 en_core_web_sm...")
        try:
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        except subprocess.CalledProcessError:
            raise RuntimeError("模型下载失败，请手动运行: python -m spacy download en_core_web_sm")

    print("[加载] 正在加载 spaCy 英文模型...")
    # 词形还原只需 tagger/attribute_ruler/lemmatizer，跳过 parser/ner 提速
    _SPACY_NLP = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    return _SPACY_NLP


def split_text_chunks(text, limit=400_000):
    """把长文本切块（尽量在空白处切断，避免截断单词，也绕过 spaCy 长度限制）"""
    if len(text) <= limit:
        return [text] if text else ['']
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            cut = max(text.rfind('\n', start, end), text.rfind(' ', start, end))
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        start = end
    return chunks


def extract_words_spacy(text):
    """spaCy 流程：分词 -> 词形还原 -> 小写 -> 去重排序

    注意：单字母噪声不在此处过滤，统一由「词典筛词」环节归类处理。
    """
    nlp = get_spacy_nlp()
    lemmas = set()
    for doc in nlp.pipe(split_text_chunks(text), batch_size=4):
        for tok in doc:
            if not tok.is_alpha:
                continue
            lemma = tok.lemma_.lower()
            if lemma == '-pron-':      # 兼容旧版 spaCy 的代词占位符
                lemma = tok.lower_
            lemmas.add(lemma)
    return sorted(lemmas)


def extract_words_simple(text):
    """备用简单提取（不依赖 spaCy，无词形还原）"""
    return sorted(set(re.findall(r'[a-zA-Z]+', text.lower())))

# ================= 小学基础词汇表 =================
ELEMENTARY_VOCAB_TEXT = """
a an about afraid after afternoon again all also always am and angry animal answer
any apple are arm art ask at aunt autumn
baby back bad bag ball banana basketball be bear beautiful bed before begin behind
beside between big bike bicycle bird birthday black blackboard blue boat body book
box boy bread breakfast bring brother brown bus busy but buy by
cake call can candy cap car card cat chair chicken child children china chinese
cinema city class clean clever clock close clothes cloudy coat cold colour color
come computer cook cool cousin cow crayon cry
dad dance day dear desk difficult dinner dirty do doctor dog door down draw dress
drink driver duck
ear early easy eat egg elephant email english evening every exercise eye
face family fan far farm farmer fast father favourite favorite feel film find fine
fish floor flower fly food foot feet football for friend from fruit
game get girl give go good goodbye bye grandfather grandpa grandmother grandma
grass great green
hair half hand happy have has he head healthy hear heavy hello help her here hi
high him his holiday home horse hospital hot hour house how hungry
i ice cream idea ill in interesting is it its
juice jump
kid kind kitchen kite know
lake late left leg lesson let library light like listen little live long look love
lunch
make man men many map maths math me meet milk minute miss monkey month moon
morning mother mouth mr mrs ms much mum mom music my
name near new next nice night no noodle nose not now nurse
of often old on open or orange our
panda parent park party pe physical education pen pencil people photo picture pig
place plane plant play playground please police potato pupil put
rain read red rice right river room ruler run
sad say school schoolbag science season see she sheep ship shirt shoe shop short
shorts sing sister sit skirt sleep slow small snow sock some sometimes song sorry
soup speak sport spring stand star stop story street strong study subject summer
sun sunny supermarket sweater swim
table take talk tall taxi tea teacher tell thank that the their them then there
these they thin think this those tiger time tired to today toilet tomato tomorrow
too toy train travel tree trousers try turn tv
umbrella uncle under up us use
vegetable very visit
wait walk want warm wash watch water way we wear weather week welcome well what
when where white who whose why window windy winter with woman women wonderful word
work worker worry write wrong
year yellow yes yesterday you young your
zoo
"""

ELEMENTARY_WORDS = frozenset(ELEMENTARY_VOCAB_TEXT.lower().split())


def filter_elementary_words(word_list):
    """从词表中移除小学基础词汇，返回 (保留列表, 移除列表)"""
    kept, removed = [], []
    for w in word_list:
        (removed if w.lower() in ELEMENTARY_WORDS else kept).append(w)
    return kept, removed

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


def load_word_list(file_path):
    """读取单词列表文件（每行一个单词），自动适配编码"""
    for enc in ('utf-8', 'gbk'):
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            continue
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return [line.strip() for line in f if line.strip()]

# ================= 业务逻辑模块 =================

_ENGLISH_DICT = None

# 视为有效单词的单字母例外：a（冠词）、i（代词）
SINGLE_LETTER_VALID = ('a', 'i')


def filter_valid_words(word_list):
    """流程：词典校验筛词

    无效词包括两类，统一归入无效清单：
    - 单字母噪声（b/x/y 等），其中仅 a、i 视为有效单词
    - 多字母但不在 NLTK 词典中的词（拼写错误、缩写噪声等）
    """
    global _ENGLISH_DICT
    if _ENGLISH_DICT is None:
        _ENGLISH_DICT = set(w.lower() for w in words.words())
    valid, invalid = [], []
    for w in word_list:
        wl = w.lower()
        if len(wl) == 1:
            (valid if wl in SINGLE_LETTER_VALID else invalid).append(w)
        elif wl in _ENGLISH_DICT:
            valid.append(w)
        else:
            invalid.append(w)
    return valid, invalid


def calculate_coverage_stats(dict_set, target_set):
    """计算单个词书对目标的覆盖统计

    覆盖率     = 命中词数 / 目标词数  （词书能覆盖目标词汇的比例）
    性价比指数 = 命中词数 / 词书词数  （词书命中词占词书总词数的比例）
    """
    covered = target_set & dict_set
    uncovered = target_set - dict_set
    redundant = dict_set - target_set
    dict_n, target_n, covered_n = len(dict_set), len(target_set), len(covered)
    return {
        "dict_set": dict_set,
        "covered_set": covered,
        "dict_count": dict_n,
        "target_count": target_n,
        "covered_count": covered_n,
        "uncovered_count": len(uncovered),
        "redundant_count": len(redundant),
        "uncovered_list": sorted(uncovered),
        "rate": (covered_n / target_n * 100) if target_n else 0.0,
        "value_index": (covered_n / dict_n * 100) if dict_n else 0.0,
    }

# ================= 覆盖率报告生成 =================

def build_coverage_report(target_name, target_raw_count, target_dedup_count,
                          elementary_removed, book_stats, combined, result_files):
    """生成覆盖率分析报告（控制台输出与保存文件内容一致）"""
    # ---- 各列最优值（仅多本词书时计算；并列最优均标记）----
    # 词书词数: 最少最优（仅统计非空词书，空词书评"最少"无意义）
    # 命中词数 / 覆盖率 / 性价比指数: 最多(高)最优
    # 均摊成本: 最低最优（仅统计有命中的词书，无命中显示 "-"）
    # 词书名: 覆盖率与性价比取均值后最高（综合最优）
    multi = len(book_stats) > 1
    best = {}
    if multi:
        nonempty = [s for s in book_stats if s["dict_count"] > 0]
        with_hits = [s for s in book_stats if s["covered_count"] > 0]
        best = {
            "dict": min((s["dict_count"] for s in nonempty), default=None),
            "covered": max((s["covered_count"] for s in book_stats), default=0),
            "rate": max((s["rate"] for s in book_stats), default=0.0),
            "value": max((s["value_index"] for s in nonempty), default=None),
            "cost": min((s["dict_count"] / s["covered_count"] for s in with_hits), default=None),
            "avg": max((s["rate"] + s["value_index"]) / 2 for s in book_stats),
        }

    def mark(cell, is_best):
        return f"[{cell}]" if is_best else cell

    # ---- 先构建总览表，报告宽度不小于表格宽度 ----
    headers = ["词书", "词书词数", "命中词数", "覆盖率", "性价比指数", "均摊成本"]
    aligns = ["left", "right", "right", "right", "right", "right"]
    rows = []
    for s in book_stats:
        cells = [
            shorten(f"{s['index']}. {s['name']}", 18),
            str(s["dict_count"]),
            str(s["covered_count"]),
            fmt_pct(s["rate"]),
            fmt_pct(s["value_index"]),
            fmt_cost(s),
        ]
        if multi:
            avg = (s["rate"] + s["value_index"]) / 2
            cells[0] = mark(cells[0], best["avg"] > 0 and avg == best["avg"])
            cells[1] = mark(cells[1], best["dict"] is not None and s["dict_count"] == best["dict"])
            cells[2] = mark(cells[2], best["covered"] > 0 and s["covered_count"] == best["covered"])
            cells[3] = mark(cells[3], best["rate"] > 0 and s["rate"] == best["rate"])
            cells[4] = mark(cells[4], best["value"] is not None and best["value"] > 0
                            and s["value_index"] == best["value"])
            cells[5] = mark(cells[5], best["cost"] is not None and s["covered_count"] > 0
                            and s["dict_count"] / s["covered_count"] == best["cost"])
        rows.append(cells)
    # 合计行为合并结果，不参与标记
    rows.append([
        shorten(f"合计({len(book_stats)}本合并去重)", 18),
        str(combined["dict_count"]),
        str(combined["covered_count"]),
        fmt_pct(combined["rate"]),
        fmt_pct(combined["value_index"]),
        fmt_cost(combined),
    ])
    table_lines, table_width = format_table(headers, rows, aligns)

    width = max(64, table_width + 2)
    bar = "=" * width
    sep = "-" * width
    L = []

    # ---- 报告头 ----
    L.append(bar)
    L.append(pad("覆盖率分析报告", width, "center"))
    L.append(bar)
    L.append(f"{pad('目标文件', 10)}: {target_name}")
    parts = [f"原始 {target_raw_count} 行", f"去重后 {target_dedup_count}"]
    if elementary_removed:
        parts.append(f"已移除小学基础词 {len(elementary_removed)} 个")
    L.append(f"{pad('目标词数', 10)}: {combined['target_count']} ({'，'.join(parts)})")
    L.append(f"{pad('词书数量', 10)}: {len(book_stats)}")
    L.append(f"{pad('生成时间', 10)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(sep)

    # ---- 一、词书总览 ----
    L.append("一、词书总览")
    L.append(sep)
    L.extend("  " + line for line in table_lines)
    L.append("注：所有词数均按去重后统计。")
    if elementary_removed or any(s["elem_removed_count"] for s in book_stats):
        L.append("      已开启基础词过滤：目标与各词书中的小学基础词均不参与计算。")
    if multi:
        L.append("[] 标记说明：")
        L.append("  - 各列数值被 [] 包裹 = 该列最优 (词书词数最少 / 命中词数最多 / 覆盖率最高 / 性价比指数最高 / 均摊成本最低)")
        L.append("  - 词书名被 [] 包裹 = 覆盖率与性价比取均值后最高 (综合最优)")
        L.append("  - 并列最优均标记；合计行为合并结果，不参与标记")
        L.append("  - 均摊成本与性价比指数互为倒数，这两列的标记总是落在同一本词书上")
    L.append(bar)

    # ---- 二、各词书详细数据 ----
    L.append("二、各词书详细数据")
    L.append(sep)
    for s in book_stats:
        L.append(f"[词书 {s['index']}] {s['source_name']}")
        L.append(f"  {pad('词书词数', 12)}: {s['dict_count']}")
        if s["elem_removed_count"]:
            L.append(f"  {pad('移除基础词', 12)}: {s['elem_removed_count']}  (原始 {s['dict_original_count']} 词，移除后按 {s['dict_count']} 词参与计算)")
        L.append(f"  {pad('命中词数', 12)}: {s['covered_count']}")
        L.append(f"  {pad('目标未覆盖', 12)}: {s['uncovered_count']}  (目标中该词书未收录的词)")
        L.append(f"  {pad('词书冗余词', 12)}: {s['redundant_count']}  (词书中目标文本用不到的词)")
        L.append(f"  {pad('独有命中词', 12)}: {s['unique_count']}  (仅被该词书覆盖的目标词)")
        L.append(f"  {pad('覆盖率', 12)}: {fmt_pct(s['rate'])}  = 命中 {s['covered_count']} / 目标 {s['target_count']}")
        if s["dict_count"]:
            L.append(f"  {pad('性价比指数', 12)}: {fmt_pct(s['value_index'])}  = 命中 {s['covered_count']} / 词书 {s['dict_count']}")
        else:
            L.append(f"  {pad('性价比指数', 12)}: -  (词书为空)")
        if s["covered_count"]:
            L.append(f"  {pad('均摊成本', 12)}: {fmt_cost(s)} 词/命中  = 词书 {s['dict_count']} / 命中 {s['covered_count']}")
        else:
            L.append(f"  {pad('均摊成本', 12)}: -  (该词书没有命中任何目标词)")
        L.append(sep)

    # ---- 三、综合分析 ----
    L.append("三、综合分析")
    L.append(sep)
    if multi:
        nonempty = [s for s in book_stats if s["dict_count"] > 0]
        with_hits = [s for s in book_stats if s["covered_count"] > 0]
        if nonempty:
            b = min(nonempty, key=lambda s: s["dict_count"])
            L.append(f"  {pad('词书词数最少', 12)}: 词书{b['index']} {b['name']} {b['dict_count']} 词")
        if with_hits:
            b = max(book_stats, key=lambda s: s["covered_count"])
            L.append(f"  {pad('命中词数最多', 12)}: 词书{b['index']} {b['name']} {b['covered_count']} 词")
        b = max(book_stats, key=lambda s: s["rate"])
        L.append(f"  {pad('覆盖率最高', 12)}: 词书{b['index']} {b['name']} {fmt_pct(b['rate'])}")
        b = max(book_stats, key=lambda s: s["value_index"])
        L.append(f"  {pad('性价比最高', 12)}: 词书{b['index']} {b['name']} {fmt_pct(b['value_index'])}")
        if with_hits:
            b = min(with_hits, key=lambda s: s["dict_count"] / s["covered_count"])
            L.append(f"  {pad('均摊成本最低', 12)}: 词书{b['index']} {b['name']} {fmt_cost(b)} 词/命中")
        b = max(book_stats, key=lambda s: (s["rate"] + s["value_index"]) / 2)
        avg_val = (b["rate"] + b["value_index"]) / 2
        L.append(f"  {pad('综合最优', 12)}: 词书{b['index']} {b['name']} 覆盖率与性价比均值 {fmt_pct(avg_val)}")
    L.append(f"  {pad('合并覆盖率', 12)}: {fmt_pct(combined['rate'])}  (全部词书合并去重后)")
    L.append(f"  {pad('剩余未覆盖', 12)}: {combined['uncovered_count']} 个目标词未被任何词书覆盖")
    L.append(sep)
    if combined["uncovered_list"]:
        preview = combined["uncovered_list"][:30]
        L.append(f"未覆盖词预览 (前 {len(preview)} / 共 {combined['uncovered_count']})：")
        for line in wrap_words(preview, width - 4):
            L.append(f"  {line}")
        L.append(sep)

    # ---- 指标说明 ----
    L.append("指标说明")
    L.append(sep)
    L.append("  覆盖率     = 命中词数 / 目标词数 : 词书能覆盖目标词汇的比例")
    L.append("  性价比指数 = 命中词数 / 词书词数 : 词书命中词占词书总词数的比例，衡量选该词书的性价比，越高越划算")
    L.append("  均摊成本   = 词书词数 / 命中词数 : 每命中 1 个目标词平均需掌握的词数，越低越划算")
    L.append("  独有命中词 = 仅被该词书覆盖的目标词数，衡量词书的不可替代性")
    L.append(bar)

    # ---- 结果文件清单 ----
    L.append("结果文件 (位于 1_Result 文件夹)")
    L.append(sep)
    for fn, desc in result_files:
        L.append(f"  {pad(shorten(fn, 34), 34)} {desc}")
    L.append(bar)
    return L

# ================= 工作区管理 =================

class Workspace:
    WORKSPACE_NAME = "WordTool_Workspace"

    def __init__(self, source_file):
        self.source_path = Path(source_file).resolve()
        # 若源文件本身位于某个工作区内部（如使用上次的结果文件），则复用该工作区，避免嵌套
        self.root_dir = None
        for parent in self.source_path.parents:
            if parent.name == self.WORKSPACE_NAME:
                self.root_dir = parent
                break
        if self.root_dir is None:
            self.root_dir = self.source_path.parent / self.WORKSPACE_NAME

        self.source_dir = self.root_dir / "0_Source"
        self.result_dir = self.root_dir / "1_Result"

        self._init_dirs()
        self._copy_source()

    def _init_dirs(self):
        self.root_dir.mkdir(exist_ok=True)
        self.source_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)

    def _copy_source(self):
        dest = self.source_dir / self.source_path.name
        if self.source_path != dest.resolve():
            shutil.copy2(self.source_path, dest)

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
    """提示输入单个文件路径"""
    while True:
        path = input(f"请拖入{desc}或输入路径: ").strip().strip('"').strip("'")
        if os.path.isfile(path): return path
        print("[错误] 文件不存在，请重试。")


def prompt_wordbook_paths(exclude=None):
    """提示输入多个词书文件：直接回车结束，输入 - 撤销上一个"""
    exclude = os.path.abspath(exclude) if exclude else None
    paths = []
    print("[提示] 可添加多个词书进行对比；输入 - 撤销上一个；直接回车结束添加。")
    while True:
        if paths:
            names = ", ".join(Path(p).name for p in paths)
            print(f"[已添加] {len(paths)} 个: {names}")
            hint = "回车=开始计算，或继续添加"
        else:
            hint = "至少需要 1 个词书"
        raw = input(f"请拖入【词书 {len(paths) + 1}】或输入路径 ({hint}): ").strip().strip('"').strip("'")

        if not raw:
            if paths:
                return paths
            print("[提示] 尚未添加任何词书，请先输入至少一个。")
            continue

        if raw in ('-', '--'):
            if paths:
                removed = paths.pop()
                print(f"[撤销] 已移除: {Path(removed).name}")
            else:
                print("[提示] 当前没有可撤销的词书。")
            continue

        if not os.path.isfile(raw):
            print("[错误] 文件不存在，请重试。")
            continue

        if Path(raw).suffix.lower() != '.txt':
            print("[提示] 词书建议使用 TXT 格式（每行一个单词）。")

        p = os.path.abspath(raw)
        if exclude and p == exclude:
            print("[提示] 该文件与目标文件相同，不能作为词书。")
        elif p in paths:
            print("[提示] 该词书已添加过。")
        else:
            paths.append(p)


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
    print("=" * 40)
    print("[板块一] 文本处理")
    print("=" * 40)

    f_path = prompt_file_path("源文件(.txt/.pdf/.docx)")

    print("\n处理选项：")
    do_extract = prompt_yes_no("1. 执行选词（spaCy 词形还原/去重）？", "y")
    do_filter = prompt_yes_no("2. 执行词典筛词（校验有效性，单字母噪声 b/x/z 等归入无效，保留 a/i）？", "y")
    do_elementary = prompt_yes_no("3. 过滤小学基础词汇（移除 apple/dog 等过于基础的词）？", "n")

    if not (do_extract or do_filter or do_elementary):
        print("[警告] 未选择任何操作，返回。")
        return

    try:
        ws = Workspace(f_path)
        text = load_file_text(ws.source_dir / Path(f_path).name)

        summary = []
        result_files = []
        current = None

        if do_extract:
            print("[处理中] spaCy 正在分词与词形还原...")
            try:
                current = extract_words_spacy(text)
            except (RuntimeError, OSError) as e:
                print(f"[提示] {e}")
                if prompt_yes_no("是否改用简单提取（仅小写化/去重，无词形还原）继续？", "n"):
                    current = extract_words_simple(text)
                else:
                    raise
            if current is not None:
                ws.save_result("1_Lemmatized_Words.txt", current)
                print(f"[完成] 选词完成，共 {len(current)} 个词。")
                summary.append(f"选词: 共 {len(current)} 词（spaCy 词形还原）")
                result_files.append(("1_Lemmatized_Words.txt", f"词形还原后的全部单词 {len(current)} 个"))

        if do_filter:
            if current is None:
                print("[处理中] 未启用选词，先做简单提取（无词形还原）...")
                current = extract_words_simple(text)
            total = len(current)
            print(f"[处理中] 共 {total} 个词，正在进行词典校验...")
            valid, invalid = filter_valid_words(current)
            ws.save_result("2_Valid_Words.txt", valid)
            ws.save_result("3_Invalid_Words.txt", invalid)
            n_single = sum(1 for w in invalid if len(w) == 1)
            detail = f"（其中单字母噪声 {n_single} 个）" if n_single else ""
            print(f"[完成] 筛词完成：共 {total} 词，有效 {len(valid)}，无效 {len(invalid)}{detail}。")
            summary.append(f"筛词: 共 {total} 词，有效 {len(valid)} 个 / 无效 {len(invalid)} 个{detail}")
            result_files.append(("2_Valid_Words.txt", f"词典校验有效单词 {len(valid)} 个"))
            result_files.append(("3_Invalid_Words.txt", f"词典校验无效单词 {len(invalid)} 个（含单字母噪声）"))
            current = valid

        if do_elementary:
            if current is None:
                print("[处理中] 未启用选词，先做简单提取（无词形还原）...")
                current = extract_words_simple(text)
            total = len(current)
            print(f"[处理中] 共 {total} 个词，正在过滤小学基础词汇...")
            kept, removed = filter_elementary_words(current)
            ws.save_result("4_Filtered_Words.txt", kept)
            ws.save_result("5_Removed_Elementary_Words.txt", removed)
            print(f"[完成] 基础词过滤完成：共 {total} 词，移除 {len(removed)} 个，剩余 {len(kept)} 个。")
            current = kept
            summary.append(f"基础词过滤: 共 {total} 词，移除 {len(removed)} 个，剩余 {len(kept)} 个")
            result_files.append(("4_Filtered_Words.txt", f"过滤基础词后的单词 {len(kept)} 个"))
            result_files.append(("5_Removed_Elementary_Words.txt", f"被移除的小学基础词 {len(removed)} 个"))

        # 清屏后输出干净的完成摘要
        clear_screen()
        width = 60
        bar = "=" * width
        print(bar)
        print(pad("文本处理完成", width, "center"))
        print(bar)
        print(f"源文件: {Path(f_path).name}")
        for line in summary:
            print(f"- {line}")
        print("-" * width)
        print("结果文件 (位于 1_Result 文件夹):")
        for fn, desc in result_files:
            print(f"  {pad(shorten(fn, 32), 32)} {desc}")
        print(bar)

        ws.open_in_explorer()

    except Exception as e:
        print(f"[错误] 发生错误: {e}")


def run_coverage_analysis():
    """流程：覆盖率计算（一个目标 x 多个词书）"""
    print("=" * 40)
    print("[板块二] 覆盖率计算 (支持多词书)")
    print("=" * 40)

    print("> 目标文件与词书均为 TXT 格式，每行一个单词。")
    print("> 同一目标可添加多个词书，将逐一对比覆盖率与性价比指数。")

    target_path = prompt_file_path("【目标】文件")
    dict_paths = prompt_wordbook_paths(exclude=target_path)
    filter_elem = prompt_yes_no(
        "计算前是否移除目标与词书中的小学基础词汇？\n"
        "(目标与词书中的 apple/dog 等基础词均不参与计算，覆盖率与性价比更真实)", "y")

    try:
        print("[处理中] 正在读取目标文件...")
        target_words = load_word_list(target_path)
        target_raw_count = len(target_words)
        target_set = set(target_words)
        target_dedup_count = len(target_set)

        elementary_removed = []
        if filter_elem:
            elementary_removed = sorted(w for w in target_set
                                        if w.lower() in ELEMENTARY_WORDS)
            target_set -= set(elementary_removed)
            if elementary_removed:
                print(f"[处理中] 已从目标移除 {len(elementary_removed)} 个小学基础词。")

        if not target_set:
            print("[警告] 过滤后目标中没有单词，已取消。")
            return

        ws = Workspace(target_path)

        # 清理上次运行留下的旧结果，避免新旧文件混淆
        for pattern in ("Coverage_Report.txt", "Uncovered_*.txt", "Removed_Elementary.txt"):
            for old in ws.result_dir.glob(pattern):
                try:
                    old.unlink()
                except OSError:
                    pass

        # 备份词书到工作区（与已有文件同名的自动加序号）
        # 注意：备份保留原始内容，基础词过滤仅发生在计算中
        used_names = {ws.source_path.name.lower()}
        for dp in dict_paths:
            src = Path(dp).resolve()
            name, n = src.name, 1
            while name.lower() in used_names:
                n += 1
                name = f"{src.stem}_{n}{src.suffix}"
            dest = ws.source_dir / name
            if src != dest.resolve():
                shutil.copy2(src, dest)
            used_names.add(name.lower())

        print(f"[处理中] 正在计算 {len(dict_paths)} 本词书的覆盖率...")
        book_stats = []
        for idx, dp in enumerate(dict_paths, 1):
            dict_set = set(load_word_list(dp))
            dict_original_count = len(dict_set)
            elem_removed_count = 0
            if filter_elem:
                book_elem = {w for w in dict_set if w.lower() in ELEMENTARY_WORDS}
                elem_removed_count = len(book_elem)
                dict_set -= book_elem
                if elem_removed_count:
                    print(f"[处理中] 词书 {idx} [{Path(dp).name}]: "
                          f"移除基础词 {elem_removed_count} 个 ({dict_original_count} -> {len(dict_set)})")
            if not dict_set:
                if dict_original_count:
                    print(f"[警告] 词书 {idx} [{Path(dp).name}] 过滤基础词后没有剩余单词。")
                else:
                    print(f"[警告] 词书 {idx} [{Path(dp).name}] 中没有读到任何单词。")
            stats = calculate_coverage_stats(dict_set, target_set)
            stats["index"] = idx
            stats["name"] = Path(dp).stem
            stats["source_name"] = Path(dp).name
            stats["dict_original_count"] = dict_original_count
            stats["elem_removed_count"] = elem_removed_count
            book_stats.append(stats)

        # 独有命中词数：仅被该词书覆盖的目标词
        for i, s in enumerate(book_stats):
            others = set()
            for j, t in enumerate(book_stats):
                if j != i:
                    others |= t["covered_set"]
            s["unique_count"] = len(s["covered_set"] - others)

        # 全部词书合并（去重）后的统计（同样基于过滤后的词书）
        union_dicts = set()
        for s in book_stats:
            union_dicts |= s["dict_set"]
        combined = calculate_coverage_stats(union_dicts, target_set)

        # 保存基础词移除清单
        if elementary_removed:
            ws.save_result("Removed_Elementary.txt", elementary_removed)

        # 结果文件清单（写入报告尾部）
        result_files = [("Coverage_Report.txt", "本报告")]
        if elementary_removed:
            result_files.append(("Removed_Elementary.txt",
                                 f"从目标移除的小学基础词 {len(elementary_removed)} 个"))
        for s in book_stats:
            fn = f"Uncovered_{s['index']}_{s['name']}.txt"
            result_files.append((fn, f"词书{s['index']} [{s['name']}] 未覆盖的目标词，共 {s['uncovered_count']} 个"))
        result_files.append(("Uncovered_All.txt", f"所有词书合并后仍未覆盖的词，共 {combined['uncovered_count']} 个"))

        # 生成报告
        report = build_coverage_report(
            Path(target_path).name, target_raw_count, target_dedup_count,
            elementary_removed, book_stats, combined, result_files)

        # 保存结果文件
        for s in book_stats:
            ws.save_result(f"Uncovered_{s['index']}_{s['name']}.txt", s["uncovered_list"])
        ws.save_result("Uncovered_All.txt", combined["uncovered_list"])
        ws.save_result("Coverage_Report.txt", report)

        # 清屏后从顶部完整展示报告
        clear_screen()
        print("\n".join(report))

        print("\n[完成] 统计完毕！正在打开结果文件夹...")
        ws.open_in_explorer()

    except Exception as e:
        print(f"[错误] 发生错误: {e}")


def main():
    init_nltk_resources()
    while True:
        clear_screen()
        print("=" * 40)
        print("英语单词处理工具箱")
        print("=" * 40)
        print("1. [文本处理] 选词 / 筛词 / 基础词过滤")
        print("2. [覆盖率] 多词书对比统计")
        print("0. [退出]")
        print("-" * 40)

        choice = input("请选择功能 [0-2]: ").strip()

        if choice == '1':
            clear_screen()
            run_text_processing()
            pause_return()
        elif choice == '2':
            clear_screen()
            run_coverage_analysis()
            pause_return()
        elif choice == '0':
            print("[退出] 再见!")
            break
        else:
            print("无效输入，请重试。")
            time.sleep(1)


if __name__ == "__main__":
    main()
