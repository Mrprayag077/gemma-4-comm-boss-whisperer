from llama_cpp import Llama
import gradio as gr

MODEL_PATH = "/home/prayag/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
print("Model loaded.")


def text_fixer(raw_text: str) -> str:
    prompt = (
        "You are a professional editor. Fix all typos, spelling mistakes, grammar errors, and punctuation issues "
        "in the following text. Keep the original meaning and tone exactly as-is — only fix mistakes. "
        "Respond only with the corrected text, nothing else.\n\n"
        f"Original Text: \"{raw_text}\"\n\nCorrected Text:"
    )
    output = llm(prompt, max_tokens=512, temperature=0.3, echo=False)
    return output["choices"][0]["text"].strip()


demo = gr.Interface(
    fn=text_fixer,
    inputs=gr.Textbox(lines=5, placeholder="Paste your text here — typos, grammar mistakes, anything...", label="Your Text"),
    outputs=gr.Textbox(lines=8, label="Fixed Text"),
    title="Text Fixer",
    description="Instantly fixes typos, spelling, grammar, and punctuation — running fully local using Gemma 4.",
    examples=[
        ["i dont knw waht im doign with this proyect its a complet mess"],
        ["Their going to the store tommorow to bye some grocerys"],
        ["the meeting was schedueld for thurdsay but nobdy showed up"],
    ],
)
demo.launch()
