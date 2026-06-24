import re
import json
import torch
import jieba
import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
基于大语言模型的产品评论细粒度情感分析系统

系统流程：
1. 输入产品评论
2. 输入有效性检测
3. TF-IDF + SVM baseline 输出整体情感：正面 / 负面
4. 将“原始评论 + baseline 整体情感结果”构造成 Prompt
5. Qwen3-8B 输出 JSON 格式的方面级结果
6. 程序解析 JSON
7. 输出：
   - Baseline 整体情感
   - 方面情感摘要
   - 大模型细粒度分析
   - 综合总结

测试语句：
质量太差了，用了两天就坏了，物流也很慢
这个电脑运行速度很快，屏幕也很清晰，但是价格有点贵，物流太慢了
这款手机是黑色的，内存是128G，屏幕尺寸是6.5英寸
这款手机是黑色的，拍照很清晰，但是电池不耐用
"""


# =========================
# 1. 路径配置
# =========================

SVM_MODEL_PATH = "svm_weight/baseline_all_cats/svm_sentiment_model.pkl"
TFIDF_PATH = "svm_weight/baseline_all_cats/tfidf_vectorizer.pkl"

QWEN_MODEL_PATH = "/data2/nfs_node1/gb/gb_project/Project/EmotionAnalysis/models/Qwen3-8B"


# =========================
# 2. 加载 baseline
# =========================

print("正在加载 SVM baseline...")
svm_model = joblib.load(SVM_MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


# =========================
# 3. 加载 Qwen3-8B
# =========================

print("正在加载 Qwen3-8B tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    QWEN_MODEL_PATH,
    trust_remote_code=True
)

print("正在加载 Qwen3-8B 模型...")
llm = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

llm.eval()

print("模型加载完成。")


# =========================
# 4. 输入有效性检测
# =========================

def validate_product_review(text):
    """
    判断输入是否像一条产品评论。
    """

    text = str(text).strip()

    if len(text) == 0:
        return False, "输入为空"

    if len(text) < 4:
        return False, "文本过短，不像完整产品评论"

    chinese_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    if len(chinese_chars) < 2:
        return False, "中文内容过少"

    non_review_keywords = [
        "你是谁", "你好", "在吗", "天气", "几点", "新闻",
        "帮我写", "写代码", "翻译", "总结", "讲个笑话",
        "生成图片", "打开文件", "运行代码", "怎么安装",
        "什么是", "介绍一下", "帮我分析代码"
    ]

    for word in non_review_keywords:
        if word in text:
            return False, "输入内容更像普通问答或指令，不像产品评论"

    review_keywords = [
        # 商品类别词
        "手机", "电脑", "平板", "书", "书籍", "水果", "衣服",
        "酒店", "洗发水", "热水器", "牛奶", "商品", "产品", "东西",

        # 通用评论词
        "质量", "做工", "材质", "价格", "价钱", "物流", "快递",
        "发货", "包装", "客服", "服务", "售后", "退货", "换货",
        "购买", "买", "收到", "使用", "用了", "用起来", "体验",
        "效果", "安装", "师傅", "配送",

        # 数码
        "屏幕", "性能", "速度", "电池", "续航", "拍照", "内存",
        "配置", "系统", "运行", "清晰",

        # 食品
        "味道", "口感", "新鲜", "保质期",

        # 服装
        "尺码", "面料", "大小", "合身",

        # 酒店
        "房间", "卫生", "环境", "位置", "隔音", "床"
    ]

    sentiment_keywords = [
        "好", "很好", "不错", "满意", "喜欢", "推荐", "划算",
        "实惠", "清晰", "流畅", "舒服", "新鲜", "快",
        "差", "很差", "失望", "不好", "垃圾", "后悔",
        "贵", "慢", "坏", "破", "脏", "吵", "卡", "难用"
    ]

    has_review_word = any(word in text for word in review_keywords)
    has_sentiment_word = any(word in text for word in sentiment_keywords)

    if not has_review_word and not has_sentiment_word:
        return False, "未检测到明显的产品评价内容"

    return True, "有效产品评论"


# =========================
# 5. baseline 文本处理
# =========================

def clean_text(text):
    text = str(text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cut_words(text):
    text = clean_text(text)
    words = jieba.lcut(text)
    return " ".join(words)


def predict_overall_sentiment(review):
    """
    Baseline 整体情感预测。
    只返回整体情感，不输出得分和置信度。
    """

    cut_review = cut_words(review)
    review_tfidf = tfidf.transform([cut_review])

    pred = svm_model.predict(review_tfidf)[0]

    if pred == 1:
        return "正面"
    else:
        return "负面"


# =========================
# 6. 构造大模型 Prompt
# =========================

def build_prompt(review, baseline_sentiment):
    prompt = f"""
你是一个产品评论细粒度情感分析助手。

请对下面的产品评论进行方面级细粒度情感分析。

已知传统情感分类模型 TF-IDF + SVM 给出的整体情感结果为：
整体情感：{baseline_sentiment}

请注意：
1. baseline 结果只作为整体参考；
2. baseline 只能判断整体正面或负面，不代表每个方面的情感；
3. 最终判断必须以原始评论内容为准；
4. 一条评论中可以同时包含正面、负面和中性方面；
5. 不要因为整体情感是正面或负面，就强行让所有方面都相同；
6. evidence 必须是原评论中的原文片段，不要自己编造。

情感判断标准：
1. 如果用户对某个方面表达满意、喜欢、好用、划算、快、清晰、舒服等积极态度，标为“正面”；
2. 如果用户对某个方面表达不满、差、贵、慢、坏、卡顿、失望、难用等消极态度，标为“负面”；
3. 如果用户只是客观描述某个方面，没有明显好坏评价，标为“中性”；
4. 如果同一条评论中同时有正面和负面内容，不要把它整体标为中性，而是分别抽取不同方面。

任务要求：
1. 抽取评论中涉及的评价方面；
2. 判断每个方面的情感倾向，情感只能是：正面、负面、中性；
3. 提取对应的原文依据短句；
4. 只输出 JSON 数组；
5. 不要输出任何解释文字；
6. 不要使用 Markdown 代码块；
7. 如果没有明确评价方面，输出 []。

可选方面包括：
质量、价格、物流、服务、包装、性能、屏幕、外观、电池、拍照、内容、印刷、口感、新鲜度、尺码、面料、卫生、环境、位置、安装、存储、其他。

输出示例1：
[
  {{"aspect": "性能", "sentiment": "正面", "evidence": "运行速度很快"}},
  {{"aspect": "价格", "sentiment": "负面", "evidence": "价格有点贵"}}
]

输出示例2：
[
  {{"aspect": "外观", "sentiment": "中性", "evidence": "手机是黑色的"}},
  {{"aspect": "存储", "sentiment": "中性", "evidence": "内存是128G"}}
]

产品评论：
{review}
"""
    return prompt.strip()


# =========================
# 7. Qwen 推理
# =========================

def qwen_generate(prompt, max_new_tokens=1024):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    with torch.no_grad():
        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return response.strip()


# =========================
# 8. 方面名称归一化
# =========================

def normalize_aspect(aspect):
    """
    统一方面名称，避免大模型输出同义方面导致展示混乱。
    """

    aspect = str(aspect).strip()

    aspect_alias_map = {
        "快递": "物流",
        "配送": "物流",
        "发货": "物流",
        "送货": "物流",
        "运输": "物流",

        "售后服务": "服务",
        "客服服务": "服务",
        "客服": "服务",

        "做工": "质量",
        "材质": "质量",
        "耐用性": "质量",

        "价钱": "价格",
        "性价比": "价格",
        "优惠": "价格",

        "运行速度": "性能",
        "配置": "性能",
        "系统": "性能",

        "显示": "屏幕",
        "显示效果": "屏幕",
        "屏幕尺寸": "屏幕",

        "续航": "电池",
        "电量": "电池",

        "相机": "拍照",
        "摄像头": "拍照",

        "味道": "口感",

        "新鲜": "新鲜度",

        "内存": "存储",
        "容量": "存储",

        "衣服尺码": "尺码",
        "尺寸": "尺码",

        "布料": "面料",

        "酒店卫生": "卫生",
        "房间卫生": "卫生",

        "酒店环境": "环境",
        "房间环境": "环境",

        "地理位置": "位置",

        "安装服务": "安装"
    }

    return aspect_alias_map.get(aspect, aspect)


# =========================
# 9. 大模型 JSON 输出解析
# =========================

def remove_markdown_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_array(text):
    """
    从大模型输出中提取 JSON 数组。
    """

    text = remove_markdown_code_fence(text)

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return None

    json_text = text[start:end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    results = []

    for item in data:
        if not isinstance(item, dict):
            continue

        aspect = item.get("aspect") or item.get("方面") or item.get("评价方面")
        sentiment = item.get("sentiment") or item.get("情感") or item.get("情感倾向")
        evidence = item.get("evidence") or item.get("依据") or item.get("原文依据")

        if aspect is None or sentiment is None or evidence is None:
            continue

        aspect = normalize_aspect(aspect)
        sentiment = str(sentiment).strip()
        evidence = str(evidence).strip()

        if sentiment not in ["正面", "负面", "中性"]:
            continue

        if len(aspect) == 0 or len(evidence) == 0:
            continue

        results.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "evidence": evidence
        })

    return results


def repair_json_output(raw_output):
    """
    如果第一次输出不是合法 JSON，则让大模型修复一次。
    """

    repair_prompt = f"""
下面是一段产品评论细粒度情感分析结果，但它可能不是严格的 JSON 数组格式。

请你将其转换为严格合法的 JSON 数组。

要求：
1. 只保留 aspect、sentiment、evidence 三个字段；
2. sentiment 只能是：正面、负面、中性；
3. 只输出 JSON 数组；
4. 不要输出解释文字；
5. 不要使用 Markdown 代码块；
6. 如果无法解析，输出 []。

待转换内容：
{raw_output}
"""
    repaired_output = qwen_generate(repair_prompt, max_new_tokens=512)
    return extract_json_array(repaired_output)


def parse_llm_output(raw_output):
    results = extract_json_array(raw_output)

    if results is not None:
        return results

    repaired_results = repair_json_output(raw_output)

    if repaired_results is not None:
        return repaired_results

    return None


# =========================
# 10. 方面情感摘要
# =========================

def build_aspect_sentiment_summary(results):
    """
    生成方面情感摘要。

    输出示例：
    质量：负面
    物流：负面
    价格：正面
    """

    if results is None:
        return "未能解析出方面级情感结果。"

    if len(results) == 0:
        return "未识别到明确的评价方面。"

    aspect_order = [
        "质量", "价格", "物流", "服务", "包装",
        "性能", "屏幕", "外观", "电池", "拍照",
        "内容", "印刷", "口感", "新鲜度",
        "尺码", "面料", "卫生", "环境", "位置", "安装",
        "存储", "其他"
    ]

    aspect_sentiments = {}

    for item in results:
        aspect = normalize_aspect(item["aspect"])
        sentiment = item["sentiment"]

        if aspect not in aspect_sentiments:
            aspect_sentiments[aspect] = []

        aspect_sentiments[aspect].append(sentiment)

    summary_lines = []

    for aspect in aspect_order:
        if aspect not in aspect_sentiments:
            continue

        sentiments = aspect_sentiments[aspect]

        if "负面" in sentiments:
            final_sentiment = "负面"
        elif "正面" in sentiments:
            final_sentiment = "正面"
        else:
            final_sentiment = "中性"

        summary_lines.append(f"{aspect}：{final_sentiment}")

    for aspect, sentiments in aspect_sentiments.items():
        if aspect in aspect_order:
            continue

        if "负面" in sentiments:
            final_sentiment = "负面"
        elif "正面" in sentiments:
            final_sentiment = "正面"
        else:
            final_sentiment = "中性"

        summary_lines.append(f"{aspect}：{final_sentiment}")

    if len(summary_lines) == 0:
        return "未识别到明确的评价方面。"

    return "\n".join(summary_lines)


# =========================
# 11. 细粒度分析文字
# =========================

def format_finegrained_text(results):
    """
    将结构化结果转换成自然语言细粒度分析文字。
    """

    if results is None:
        return "大模型返回的内容未能解析为有效结果，建议重新输入评论或调整提示词。"

    if len(results) == 0:
        return "该评论中未识别到明确的评价方面，因此未生成细粒度情感分析结果。"

    details = []

    for item in results:
        aspect = normalize_aspect(item["aspect"])
        sentiment = item["sentiment"]
        evidence = item["evidence"]

        details.append(
            f"在“{aspect}”方面，用户表现出{sentiment}情感，依据是“{evidence}”"
        )

    text = "具体来看，"
    text += "；".join(details)
    text += "。"

    return text


# =========================
# 12. 综合总结
# =========================

def build_final_summary(baseline_sentiment, results):
    """
    生成综合总结。
    """

    if results is None or len(results) == 0:
        return f"综合来看，该评论整体为{baseline_sentiment}，但大模型未能提取出明确的方面级评价内容。"

    positive_aspects = []
    negative_aspects = []
    neutral_aspects = []

    for item in results:
        aspect = normalize_aspect(item["aspect"])
        sentiment = item["sentiment"]

        if sentiment == "正面" and aspect not in positive_aspects:
            positive_aspects.append(aspect)
        elif sentiment == "负面" and aspect not in negative_aspects:
            negative_aspects.append(aspect)
        elif sentiment == "中性" and aspect not in neutral_aspects:
            neutral_aspects.append(aspect)

    summary = f"综合来看，该评论整体为{baseline_sentiment}。"

    if positive_aspects:
        summary += "用户正面评价主要集中在" + "、".join(positive_aspects) + "等方面。"

    if negative_aspects:
        summary += "用户负面评价主要集中在" + "、".join(negative_aspects) + "等方面。"

    if neutral_aspects:
        summary += "其中" + "、".join(neutral_aspects) + "等方面主要属于客观描述，情感倾向较弱。"

    if baseline_sentiment == "负面" and negative_aspects:
        summary += "因此，该评论整体偏负面的主要原因是这些方面存在明显不满。"
    elif baseline_sentiment == "正面" and positive_aspects:
        summary += "因此，该评论整体偏正面的主要原因是这些方面获得了较积极评价。"

    return summary


# =========================
# 13. 完整预测流程
# =========================

def finegrained_sentiment_analysis(review):
    baseline_sentiment = predict_overall_sentiment(review)

    prompt = build_prompt(
        review=review,
        baseline_sentiment=baseline_sentiment
    )

    raw_llm_output = qwen_generate(prompt)
    parsed_results = parse_llm_output(raw_llm_output)

    aspect_summary = build_aspect_sentiment_summary(parsed_results)
    detail_text = format_finegrained_text(parsed_results)
    final_summary = build_final_summary(baseline_sentiment, parsed_results)

    return baseline_sentiment, aspect_summary, detail_text, final_summary


# =========================
# 14. 主程序
# =========================

if __name__ == "__main__":
    print("=" * 80)
    print("基于大语言模型的产品评论细粒度情感分析系统")
    print("输入 q 退出")
    print("=" * 80)

    while True:
        review = input("\n请输入产品评论：\n")

        if review.lower() == "q":
            print("系统已退出。")
            break

        is_valid, reason = validate_product_review(review)

        if not is_valid:
            print("\n【输入有效性检测】")
            print("该输入不是有效的产品评论。")
            print("原因：", reason)
            print("请重新输入一条产品评论。")
            print("-" * 80)
            continue

        baseline_sentiment, aspect_summary, detail_text, final_summary = finegrained_sentiment_analysis(review)

        print("\n【Baseline 整体情感分析】")
        print("整体情感：", baseline_sentiment)

        print("\n【方面情感摘要】")
        print(aspect_summary)

        print("\n【大模型细粒度情感分析】")
        print(detail_text)

        print("\n【综合总结】")
        print(final_summary)

        print("-" * 80)