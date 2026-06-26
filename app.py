import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- STEP 1: Load the Model and Tokenizer ---
# NOTE: Replace 'google/gemma-4b' with the specific model identifier you plan to use.
# The actual size (e.g., 2B, 7B) will depend on your hardware.
MODEL_NAME = "google/gemma-2b" # Example placeholder

try:
    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the Model (adjust device if using a GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Use float16 for efficiency
        device_map="auto"           # Automatically map model layers to available devices
    )
    print(f"Model {MODEL_NAME} loaded successfully on {device}.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have the necessary hardware and libraries installed.")
    exit()


# --- STEP 2: Define the Magic Prompt Function ---
def corporate_translator(rude_text: str) -> str:
    """
    Translates rude/blunt text into a professional, diplomatic email.
    """
    # The "Magic Prompt" that guides the model's behavior
    magic_prompt = (
        "You are a corporate communications expert. Translate the following blunt, emotional, or rude text "
        "into a professional, polite, and diplomatic business email while keeping the original intent. "
        "Respond only with the revised professional email."
    )

    # Construct the full input for the model
    full_prompt = f"{magic_prompt}\n\nOriginal Rude Text: \"{rude_text}\"\n\nProfessional Business Email:"

    # Tokenize the input text
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate the response
    print("\n--- Generating Response ---")
    output_ids = model.generate(
        input_ids,
        max_length=512,
        num_beams=4,
        temperature=0.7, # A balance between creativity and adherence to prompt
        do_sample=True
    )

    # Decode the output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response

# --- STEP 3: Run the Application ---
if __name__ == "__main__":
    import gradio as gr

    def translate(rude_text):
        if not rude_text.strip():
            return "Please enter some text."
        return corporate_translator(rude_text)

    demo = gr.Interface(
        fn=translate,
        inputs=gr.Textbox(lines=5, placeholder="Type your blunt/rude text here...", label="Rude Input"),
        outputs=gr.Textbox(lines=8, label="Professional Corporate Email"),
        title="Corporate Translator",
        description="Turns blunt or rude text into a polished business email using Gemma.",
        examples=[
            ["This deadline is stupid and I'm not going to meet it."],
            ["Your code is terrible and broke everything."],
            ["I told you this wouldn't work. Why didn't you listen?"],
        ],
    )
    demo.launch()

