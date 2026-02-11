import logging
import random

from llm import query_llm


def generate_segment_templates(segments_df, segment_col, strategy_map, segment_examples):
    logger = logging.getLogger(__name__)
    segment_templates = {}

    for segment, segment_data in segments_df.groupby(segment_col):
        strategy = strategy_map.get(segment)
        if not strategy:
            logger.warning("No strategy found for segment '%s'", segment)
            continue

        # Insert one Urdu example into the prompt when available for this segment
        segment_example_list = segment_examples.get(segment, [])
        if segment_example_list:
            example_text = random.choice(segment_example_list)
            example_block = f"""
# Refer to the following Urdu examples for this segment
{example_text}
"""
            style_instructions = """
Imitate the spoken Pakistani Urdu style and sentence rhythm in the example.
Do not generate formal or notice-style Urdu.
Rewrite the message to match the example's tone and style.
"""
        else:
            example_block = ""
            style_instructions = ""

        prompt = f"""
آپ پاکستانی صارفین کے لیے ایک اردو SMS ٹیمپلیٹ بنا رہے ہیں۔ سیگمنٹ: {segment}
تمام قواعد سختی سے فالو کریں۔ اگر دو قواعد ٹکرائیں تو سخت تر قاعدہ لاگو ہوگا۔

--- سیگمنٹ ڈیٹا ---
سیگمنٹ: {segment}
انداز: {strategy.get('Tone', '')}
اہم الفاظ/جملے: {strategy.get('Example Keywords/Phrases', '')}
مشغولیت کا انداز: {strategy.get('Engagement Style', '')}
فوکس: {strategy.get('Message Focus', '')}
{example_block}

--- اہم: پلیس ہولڈرز استعمال کریں ---
{style_instructions}
• پیغام میں لازمی طور پر `{{name}}` پلیس ہولڈر شامل ہو جہاں صارف کا اردو نام آئے گا۔
• پیغام میں لازمی طور پر `{{products}}` پلیس ہولڈر شامل ہو، جو نئی لائنز میں مصنوعات دکھائے گا۔
• مثال ڈھانچہ:
  "محترم {{name}}، ...
  ہماری مندرجہ ذیل مصنوعات پسند آئیں گی:
  {{products}}
  ... نیک تمناؤں کے ساتھ، باخبر کسان ٹیم"
• ٹیمپلیٹ میں کوئی مخصوص نام یا مصنوعات نہ لکھیں۔

--- زبان کا قاعدہ (سب سے اہم) ---
• SMS صرف اردو میں ہو۔
• سادہ، روزمرہ اردو استعمال کریں۔
• انگریزی، رومن اردو یا مخلوط زبان استعمال نہ کریں۔
• اگر مصنوعات کے نام proper nouns ہیں تو وہ جوں کے توں رہ سکتے ہیں۔

--- حکمتِ عملی ---
انداز: {strategy.get('Tone', '')}
اہم الفاظ/جملے: {strategy.get('Example Keywords/Phrases', '')}
مشغولیت کا انداز: {strategy.get('Engagement Style', '')}
فوکس: {strategy.get('Message Focus', '')}
ممنوع الفاظ: discerning

--- سخت قواعد (لازمی) ---
1. `{{products}}` شامل ہونے سے پہلے ٹیمپلیٹ 200 حروف سے کم ہو۔
   • حروف = حروف + اعداد + اسپیس + رموز + `{{name}}` + `{{products}}`
   • `{{products}}` بعد میں حقیقی مصنوعات سے بدلا جائے گا اور ہر آئٹم نئی لائن پر ہوگا
   • اگر ٹیمپلیٹ 100 حروف سے زیادہ ہو تو مختصر کریں۔

2. ترتیب لازمی:
   a) `{{name}}` کے ساتھ سلام
   b) سیگمنٹ کے مطابق پیغام
   c) مصنوعات کا تعارفی جملہ (مثلاً "ہماری مندرجہ ذیل مصنوعات پسند آئیں گی:")
   d) `{{products}}` نئی لائن پر
   e) اختتامی جملہ
   f) دستخط: "نیک تمناؤں کے ساتھ، باخبر کسان ٹیم"

3. `{{products}}` لازماً اپنی لائن پر ہو اور اس سے پہلے نئی لائن ہو۔

4. ایموجی نہ ہوں، بناوٹی/اشتہاری زبان نہ ہو، طویل جملے نہ ہوں۔

5. ذکر نہ کریں:
   • قیمت/رقم
   • "precision engineered", "exclusive range", "premium quality" جیسے الفاظ

6. کوئی وضاحت، نوٹس یا پروسیس بیان نہ کریں۔

7. پیغام قدرتی، دوستانہ اور پاکستانی انداز میں ہو۔

8. کوئی لنک شامل نہ کریں۔ لنک بعد میں خود شامل ہوگا۔

--- آؤٹ پٹ فارمیٹ ---
صرف SMS ٹیمپلیٹ لکھیں جس میں `{{name}}` اور `{{products}}` ہوں۔
کوئی اضافی متن، تبصرہ، کوٹس یا مارک ڈاؤن نہیں۔

--- اب اردو SMS ٹیمپلیٹ بنائیں ---
"""

        response = query_llm(prompt)

        if response.startswith("[Error]") or not any("\u0600" <= c <= "\u06FF" for c in response):
            logger.warning("LLM response invalid for segment %s, retrying with fallback prompt", segment)
            fallback_prompt = f"""
You are writing an Urdu SMS template for Pakistani farmers, segment: {segment}.

STRICT RULES:
- Only Urdu (no English words).
- Must include {{name}} and {{products}} placeholders.
- Keep it under 100 characters before products.
- End with: "نیک تمناؤں کے ساتھ، باخبر کسان ٹیم"
- Make it simple and natural.
"""
            response = query_llm(fallback_prompt)

        if response.startswith("[Error]"):
            logger.error("All LLM attempts failed for segment %s", segment)
            fallback_template = f"محترم {{name}}، {segment} کے لئے خصوصی پیشکش!\nہماری مندرجہ ذیل مصنوعات پسند آئیں گی:\n{{products}}\nنیک تمناؤں کے ساتھ، باخبر کسان ٹیم"
            response = fallback_template

        response = response.strip()
        if "{name}" not in response and "{{name}}" not in response:
            response = f"محترم {{name}}، {response}"

        if "{products}" not in response and "{{products}}" not in response:
            if "نیک تمناؤں کے ساتھ" in response:
                parts = response.split("نیک تمناؤں کے ساتھ")
                response = f"{parts[0]}\nہماری مصنوعات:\n{{products}}\nنیک تمناؤں کے ساتھ{parts[1] if len(parts) > 1 else ''}"
            else:
                response = f"{response}\nہماری مصنوعات:\n{{products}}"

        segment_templates[segment] = response
        logger.info("Generated template for segment '%s': %s", segment, response[:50])

    return segment_templates
