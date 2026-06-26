from llama_cpp import Llama
import gradio as gr

MODEL_PATH = "/home/prayag/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
print("Model loaded.")


def corporate_translator(rude_text: str) -> str:
    prompt = (
        "You are a corporate communications expert. Translate the following blunt, emotional, or rude text "
        "into a professional, polite, and diplomatic business email while keeping the original intent. "
        "Respond only with the revised professional email.\n\n"
        f"Original Rude Text: \"{rude_text}\"\n\nProfessional Business Email:"
    )
    output = llm(prompt, max_tokens=512, temperature=0.7, echo=False)
    return output["choices"][0]["text"].strip()


demo = gr.Interface(
    fn=corporate_translator,
    inputs=gr.Textbox(lines=5, placeholder="Type your blunt/rude text here...", label="Rude Input"),
    outputs=gr.Textbox(lines=8, label="Professional Corporate Email"),
    title="Corporate Translator",
    description="Turns blunt or rude text into a polished business email using Gemma 4.",
    examples=[
        ["This deadline is stupid and I'm not going to meet it."],
        ["Your code is terrible and broke everything."],
        ["I told you this wouldn't work. Why didn't you listen?"],
    ],
)
demo.launch()
