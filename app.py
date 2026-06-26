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
    print("--- Corporate Translator Initialized ---")
    
    # Example Input from the user (the rude version)
    rude_input = input("Enter the blunt/rude text you want to translate: \n> ")

    if rude_input.strip():
        print("\nProcessing...")
        professional_output = corporate_translator(rude_input)

        print("\n=============================================")
        print("✅ PROFESSIONAL CORPORATE TRANSLATION:")
        print(professional_output)
        print("=============================================")


