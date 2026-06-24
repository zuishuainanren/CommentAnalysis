import html
import gradio as gr

from llm_finegrained_predict import (
    validate_product_review,
    finegrained_sentiment_analysis
)


# =========================
# 1. 情感标签样式
# =========================

def sentiment_badge(sentiment):
    sentiment = str(sentiment).strip()

    if sentiment == "正面":
        return '<span class="badge badge-positive">正面</span>'
    elif sentiment == "负面":
        return '<span class="badge badge-negative">负面</span>'
    elif sentiment == "中性":
        return '<span class="badge badge-neutral">中性</span>'
    else:
        return f'<span class="badge badge-neutral">{html.escape(sentiment)}</span>'


# =========================
# 2. 输入检测展示
# =========================

def format_status_html(status_text, valid=True):
    status_text = html.escape(str(status_text))

    if valid:
        return """
        <div class="result-card status-card status-valid">
            <div class="card-title">输入有效性检测</div>
            <div class="status-text">有效产品评论</div>
        </div>
        """
    else:
        return f"""
        <div class="result-card status-card status-invalid">
            <div class="card-title">输入有效性检测</div>
            <div class="status-text">{status_text}</div>
        </div>
        """


# =========================
# 3. Baseline 展示
# =========================

def format_baseline_html(baseline_sentiment):
    baseline_sentiment = str(baseline_sentiment).strip()

    if baseline_sentiment == "":
        badge = '<span class="badge badge-neutral">待分析</span>'
    else:
        badge = sentiment_badge(baseline_sentiment)

    return f"""
    <div class="result-card">
        <div class="card-title">Baseline 整体情感分析</div>
        <div class="baseline-row">
            <span class="baseline-label">整体情感</span>
            {badge}
        </div>
    </div>
    """


# =========================
# 4. 方面摘要展示
# =========================

def format_aspect_summary_html(aspect_summary):
    aspect_summary = str(aspect_summary).strip()

    if not aspect_summary:
        return """
        <div class="result-card aspect-card">
            <div class="card-title">方面情感摘要</div>
            <div class="empty-text">等待分析结果</div>
        </div>
        """

    rows = []

    for line in aspect_summary.splitlines():
        line = line.strip()
        if not line:
            continue

        if "：" in line:
            aspect, sentiment = line.split("：", 1)
        elif ":" in line:
            aspect, sentiment = line.split(":", 1)
        else:
            aspect, sentiment = line, ""

        aspect = html.escape(aspect.strip())
        sentiment = sentiment.strip()

        rows.append(
            f"""
            <div class="aspect-row">
                <span class="aspect-name">{aspect}</span>
                {sentiment_badge(sentiment)}
            </div>
            """
        )

    if not rows:
        rows.append('<div class="empty-text">暂无方面级结果</div>')

    return f"""
    <div class="result-card aspect-card">
        <div class="card-title">方面情感摘要</div>
        <div class="aspect-list">
            {''.join(rows)}
        </div>
    </div>
    """


# =========================
# 5. 分析区重点标签
# =========================

def build_aspect_tags(aspect_summary):
    aspect_summary = str(aspect_summary).strip()

    if not aspect_summary:
        return ""

    tags = []

    for line in aspect_summary.splitlines():
        line = line.strip()
        if not line:
            continue

        if "：" in line:
            aspect, sentiment = line.split("：", 1)
        elif ":" in line:
            aspect, sentiment = line.split(":", 1)
        else:
            aspect, sentiment = line, ""

        aspect = html.escape(aspect.strip())
        sentiment = sentiment.strip()

        if sentiment == "正面":
            cls = "tag-positive"
        elif sentiment == "负面":
            cls = "tag-negative"
        else:
            cls = "tag-neutral"

        tags.append(
            f'<span class="aspect-tag {cls}">{aspect} · {html.escape(sentiment)}</span>'
        )

    return "".join(tags)


# =========================
# 6. 细粒度分析 + 综合总结展示
# =========================

def format_analysis_html(detail_text, final_summary, aspect_summary=""):
    detail_text = html.escape(str(detail_text)).replace("\n", "<br>")
    final_summary = html.escape(str(final_summary)).replace("\n", "<br>")

    aspect_tags_html = build_aspect_tags(aspect_summary)

    focus_block = ""
    if aspect_tags_html:
        focus_block = f"""
        <div class="focus-block">
            <div class="mini-title">重点方面</div>
            <div class="tag-container">
                {aspect_tags_html}
            </div>
        </div>
        """

    return f"""
    <div class="result-card analysis-card">
        <div class="card-title">大模型细粒度情感分析与综合总结</div>

        {focus_block}

        <div class="analysis-section analysis-detail">
            <div class="analysis-header">
                <span class="analysis-icon">📌</span>
                <span class="analysis-title">具体分析</span>
            </div>
            <div class="analysis-content">{detail_text}</div>
        </div>

        <div class="soft-divider"></div>

        <div class="analysis-section analysis-summary">
            <div class="analysis-header">
                <span class="analysis-icon">📝</span>
                <span class="analysis-title">综合总结</span>
            </div>
            <div class="analysis-content">{final_summary}</div>
        </div>
    </div>
    """


def format_empty_analysis(text="请先输入产品评论并点击“开始分析”。"):
    text = html.escape(text)

    return f"""
    <div class="result-card analysis-card">
        <div class="card-title">大模型细粒度情感分析与综合总结</div>
        <div class="empty-analysis">
            <div class="empty-icon">💬</div>
            <div class="empty-text">{text}</div>
        </div>
    </div>
    """


# =========================
# 7. Gradio 预测函数
# =========================

def web_predict(review):
    review = str(review).strip()

    if len(review) == 0:
        return (
            format_status_html("请输入一条产品评论。", valid=False),
            format_baseline_html(""),
            format_aspect_summary_html(""),
            format_empty_analysis("请输入一条产品评论后再进行分析。")
        )

    is_valid, reason = validate_product_review(review)

    if not is_valid:
        return (
            format_status_html(f"无效输入：{reason}", valid=False),
            format_baseline_html(""),
            format_aspect_summary_html(""),
            format_empty_analysis(
                "该输入不像产品评论，请重新输入关于商品质量、价格、物流、服务、性能、口感、卫生、环境等方面的评价。"
            )
        )

    try:
        baseline_sentiment, aspect_summary, detail_text, final_summary = finegrained_sentiment_analysis(review)

        return (
            format_status_html("有效产品评论", valid=True),
            format_baseline_html(baseline_sentiment),
            format_aspect_summary_html(aspect_summary),
            format_analysis_html(detail_text, final_summary, aspect_summary)
        )

    except Exception as e:
        return (
            format_status_html("系统运行出错", valid=False),
            format_baseline_html(""),
            format_aspect_summary_html(""),
            format_empty_analysis(f"错误信息：{str(e)}")
        )


def clear_all():
    return (
        "",
        format_status_html("等待输入", valid=False),
        format_baseline_html(""),
        format_aspect_summary_html(""),
        format_empty_analysis()
    )


# =========================
# 8. 示例评论
# =========================

examples = [
    ["质量太差了，用了两天就坏了，物流也很慢"],
    ["这个电脑运行速度很快，屏幕也很清晰，但是价格有点贵，物流太慢了"],
    ["这款手机是黑色的，内存是128G，屏幕尺寸是6.5英寸"],
    ["这款手机是黑色的，拍照很清晰，但是电池不耐用"],
    ["水果很新鲜，味道也很甜，就是包装有点破"],
    ["酒店位置很好，交通方便，但是房间卫生一般，隔音也不好"],
    ["你好，今天天气怎么样"]
]


# =========================
# 9. 页面 CSS
# =========================

custom_css = """
body {
    background: #f4f7fb !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    background: #f4f7fb !important;
}

#main-title {
    text-align: center;
    margin-top: 16px;
    margin-bottom: 6px;
}

#main-title h1 {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #16233f !important;
    letter-spacing: 0.4px;
}

#subtitle {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-bottom: 24px;
}

.panel-card {
    background: #ffffff;
    border: 1px solid #e6eaf2;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}

.input-card textarea {
    border-radius: 14px !important;
    border: 1px solid #d9e0ec !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

.input-card textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
}

button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    height: 42px !important;
}

button.secondary {
    border-radius: 12px !important;
    height: 42px !important;
    font-weight: 700 !important;
}

.result-card {
    background: #ffffff;
    border: 1px solid #e6eaf2;
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}

.card-title {
    font-size: 14px;
    font-weight: 800;
    color: #334155;
    margin-bottom: 13px;
    display: flex;
    align-items: center;
}

.card-title::before {
    content: "";
    width: 4px;
    height: 16px;
    background: #2563eb;
    border-radius: 4px;
    margin-right: 8px;
}

.status-text {
    font-size: 16px;
    font-weight: 800;
}

.status-valid .status-text {
    color: #047857;
}

.status-invalid .status-text {
    color: #b91c1c;
}

.baseline-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.baseline-label {
    font-size: 15px;
    color: #475569;
    font-weight: 600;
}

.badge {
    display: inline-block;
    min-width: 52px;
    text-align: center;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
}

.badge-positive {
    color: #047857;
    background: #d1fae5;
    border: 1px solid #a7f3d0;
}

.badge-negative {
    color: #b91c1c;
    background: #fee2e2;
    border: 1px solid #fecaca;
}

.badge-neutral {
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
}

.aspect-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.aspect-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #eef2f7;
    border-radius: 13px;
    padding: 9px 12px;
}

.aspect-name {
    color: #1e293b;
    font-size: 15px;
    font-weight: 700;
}

.analysis-card {
    margin-top: 10px;
    padding: 20px;
}

.focus-block {
    background: linear-gradient(180deg, #f8fbff 0%, #f4f8ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 15px;
    padding: 14px 16px;
    margin-bottom: 16px;
}

.mini-title {
    font-size: 13px;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 10px;
}

.tag-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.aspect-tag {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    line-height: 1.2;
}

.tag-positive {
    color: #047857;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
}

.tag-negative {
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
}

.tag-neutral {
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.analysis-section {
    border-radius: 15px;
    padding: 16px 18px;
}

.analysis-detail {
    background: #f8fbff;
    border: 1px solid #dbeafe;
}

.analysis-summary {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
}

.analysis-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.analysis-icon {
    font-size: 18px;
}

.analysis-title {
    font-size: 16px;
    font-weight: 800;
    color: #1e293b;
}

.analysis-content {
    font-size: 15px;
    line-height: 1.9;
    color: #243044;
    text-align: justify;
}

.soft-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #cbd5e1, transparent);
    margin: 18px 0;
}

.empty-analysis {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 15px;
    padding: 26px 18px;
    text-align: center;
}

.empty-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.empty-text {
    font-size: 15px;
    color: #94a3b8;
    line-height: 1.8;
}

footer {
    visibility: hidden;
}
"""


# =========================
# 10. 构建 Gradio 页面
# =========================

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=["Microsoft YaHei", "Arial", "sans-serif"]
)

with gr.Blocks(
    css=custom_css,
    theme=theme,
    title="产品评论细粒度情感分析系统"
) as demo:

    gr.Markdown(
        """
        # 基于大语言模型的产品评论细粒度情感分析系统
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        <div id="subtitle">输入一条产品评论，系统自动识别整体情感、方面情感和细粒度原因。</div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel-card", "input-card"]):
                review_input = gr.Textbox(
                    label="请输入产品评论",
                    placeholder="例如：质量太差了，用了两天就坏了，物流也很慢",
                    lines=7
                )

                with gr.Row():
                    submit_btn = gr.Button("开始分析", variant="primary")
                    clear_btn = gr.Button("清空", variant="secondary")

                gr.Examples(
                    examples=examples,
                    inputs=review_input,
                    label="示例评论"
                )

        with gr.Column(scale=1):
            valid_output = gr.HTML(
                value=format_status_html("等待输入", valid=False)
            )

            baseline_output = gr.HTML(
                value=format_baseline_html("")
            )

            aspect_output = gr.HTML(
                value=format_aspect_summary_html("")
            )

    analysis_output = gr.HTML(
        value=format_empty_analysis()
    )

    submit_btn.click(
        fn=web_predict,
        inputs=review_input,
        outputs=[
            valid_output,
            baseline_output,
            aspect_output,
            analysis_output
        ]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            review_input,
            valid_output,
            baseline_output,
            aspect_output,
            analysis_output
        ]
    )


# =========================
# 11. 启动页面
# =========================

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )