from llama_cpp import Llama
import gradio as gr

MODEL_PATH = "/home/prayag/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
print("Model loaded.")

STYLE_PROMPTS = {
    "Normal": (
        "You are a professional editor. Fix all typos, spelling mistakes, grammar errors, and punctuation issues "
        "in the following text. Keep the original meaning and tone exactly as-is — only fix mistakes. "
        "Respond only with the corrected text, nothing else."
    ),
    "Teams": (
        "You are a professional editor. Fix all typos, spelling mistakes, grammar errors, and punctuation issues "
        "in the following text, then format it as a clear, concise Microsoft Teams message — short sentences, "
        "professional but friendly tone, no unnecessary fluff. Respond only with the corrected message, nothing else."
    ),
    "Friends Chat": (
        "You are an editor. Fix the typos and spelling mistakes in the following text but keep it casual, "
        "natural, and conversational — like texting a friend. Don't make it formal. "
        "Respond only with the corrected text, nothing else."
    ),
    "Email": (
        "You are a professional editor. Fix all typos, spelling mistakes, grammar errors, and punctuation issues "
        "in the following text, then format it as a proper business email with a greeting, body, and sign-off. "
        "Keep the original intent. Respond only with the finished email, nothing else."
    ),
}


def text_fixer(raw_text: str, style: str) -> str:
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["Normal"])
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original Text: {raw_text}"},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return output["choices"][0]["message"]["content"].strip()


with gr.Blocks(title="Text Fixer") as demo:
    gr.Markdown("# Text Fixer")
    gr.Markdown("Fix typos, grammar, and spelling — then format it for the right context. Runs fully local using Gemma 4.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=6, placeholder="Paste your text here — typos, grammar mistakes, anything...", label="Your Text")
            style = gr.Radio(
                choices=["Normal", "Teams", "Friends Chat", "Email"],
                value="Normal",
                label="Format for",
            )
            submit_btn = gr.Button("Fix Text", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(lines=10, label="Fixed Text")

    gr.Examples(
        examples=[
            ["i dont knw waht im doign with this proyect its a complet mess", "Normal"],
            ["hey cn we sync tmrw abt the relese blockers?", "Teams"],
            ["omg i 4got to tell u abt wat happend at the party lol", "Friends Chat"],
            ["hi just wanted to folllow up on the propsal i sent last weak did u get a chance to look at it", "Email"],
        ],
        inputs=[input_text, style],
    )

    submit_btn.click(fn=text_fixer, inputs=[input_text, style], outputs=output_text)

demo.launch()
