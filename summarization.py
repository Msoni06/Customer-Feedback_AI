from transformers import T5ForConditionalGeneration, T5Tokenizer

print("Loading T5 summarization model (t5-small)...")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def get_summary(text, max_len=150, min_len=30):
    input_text = f"summarize: {text}"

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512, 
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


long_feedback_1 = """
The new dashboard update is visually appealing, and I appreciate the new color scheme.
However, the key metrics I used to track on the main page have been moved to a
sub-menu, which now takes three clicks to access instead of one. This significantly
slows down my workflow. Furthermore, the 'Export to CSV' feature seems to be broken
for date ranges longer than 30 days. I filed a support ticket (T-12345)
about this four days ago but have not received any response beyond the
automated confirmation. Please fix the export and reconsider the placement
of the key metrics.
"""

print("\n--- Summarization Examples ---")

print("\n[Input 1]")
print(long_feedback_1)
print("\n[Short Summary (max_len=50)]")
print(get_summary(long_feedback_1, max_len=50, min_len=15))
print("\n[Detailed Summary (max_len=150)]")
print(get_summary(long_feedback_1, max_len=150, min_len=40))