from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
import gradio as gr
import torch

# تحميل النموذج المحفوظ
model_path = "saved_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# إعدادات التوليد
generation_config = GenerationConfig(
    max_length=128,
    num_beams=4,
    length_penalty=1.0,
    early_stopping=True
)

# دالة الترجمة
def generate_translation(texts, batch_size=4):
    model.eval()
    translations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations

# دالة تستخدم في الواجهة
def translate_and_evaluate(input_text):
    translated = generate_translation([input_text])[0]
    return translated

# Gradio Interface
interface = gr.Interface(
    fn=translate_and_evaluate,
    inputs=gr.Textbox(lines=3, placeholder="اكتب الجملة الإنجليزية الموجودة في مجموعة الاختبار..."),
    outputs=gr.Textbox(label="🔸 Predict Translation"),
    title="🔤 Translation English ➜ Arabic",
    description="Enter an English sentence to get its Arabic translation."
)

interface.launch()
