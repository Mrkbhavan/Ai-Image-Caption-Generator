import gradio as gr
from PIL import Image
import random
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Caption styles with 20 unique variations each
caption_styles = {
    "Emotional": [
        "This moment touches the soul.", "Every pixel carries emotion.", "Overflowing with feeling.",
        "A heart whisper in every frame.", "Captured tears and smiles.", "Raw feelings, frozen.",
        "Echoes of the heart.", "Silent emotions scream beauty.", "A pulse in each pixel.",
        "Feeling speaks louder here.", "Waves of heartfelt memory.", "Love and longing collide.",
        "Tears and joy held still.", "Captured soul spark.", "Sentiment stitched in pixels.",
        "Hearts don't lie, pictures neither.", "Melancholy beauty lingers.", "An emotional echo.",
        "Memory wrapped in light.", "Touch of inner storm."
    ],
    "Aesthetic": [
        "A visual poetry.", "Elegance frozen in time.", "Stunning frame of beauty.",
        "Art in its purest form.", "Color and calm meet.", "A touch of visual grace.",
        "Minimal and magical.", "Where light meets soul.", "Refined in pixels.",
        "A painter's dream.", "Chic yet simple.", "Sleek serenity.",
        "Timeless charm.", "Visually therapeutic.", "Muted but meaningful.",
        "Delicately beautiful.", "Ethereal essence.", "Whispers of beauty.",
        "Elegance reborn.", "Finesse in every frame."
    ],
    "Poetic": [
        "Whispers in pixel rain.", "A silent verse in color.", "The sky wrote a poem here.",
        "When light rhymes with shadow.", "A haiku in hues.", "Stilled like a sonnet.",
        "Pixel symphony of emotion.", "A stanza of solitude.", "Framed thoughts linger.",
        "Love letters in contrast.", "Verses trapped in light.", "Ink of dreams in pixels.",
        "A canvas of soft musings.", "Serenading silence.", "Echoes of metaphors.",
        "Fleeting time, frozen rhyme.", "Imagery as poetry.", "Dreams caught in frame.",
        "Soft stanzas in silence.", "The camera wrote a poem."
    ],
    "Funny": [
        "Even the camera laughed!", "Caption loading... humor found!", "Pixel comedy show!",
        "LOL in HD.", "When pixels joke.", "Caught mid-laugh.", "This image cracks up!",
        "High-res humor.", "Say cheese... or pizza?", "Pixelated punchline incoming!",
        "Giggle-certified!", "100% comedy in frame.", "Lens has jokes too.",
        "Zoom in for more fun.", "Mood: Meme-worthy.", "Laughs in every pixel.",
        "Snap and giggle.", "This image has dad jokes.", "Certified funboi moment.",
        "Trolled by a camera."
    ],
    "Dramatic": [
        "Lights. Camera. Emotions!", "Pixel power unleashed.", "Dramatic enough to win an Oscar!",
        "Cinematic chaos!", "Frame of tension.", "A storm in silence.",
        "So intense, it stares back.", "Zoom into drama.", "Still screams passion.",
        "Bold and brooding.", "A saga in still.", "Explosive calm.",
        "Edge of emotion.", "When frames fight feelings.", "Dark mode activated.",
        "Peak drama alert.", "Unfiltered tension.", "Frame of fate.",
        "The plot thickens here.", "Theatrical vibes only."
    ],
    "Inspirational": [
        "Dream big, capture the moment.", "Every image tells a story of hope.", "Inspire through visuals.",
        "Motivation in every pixel.", "Shine on, silently.", "Frame your future.",
        "Vision becomes reality here.", "Faith in focus.", "Pixel-powered dreams.",
        "Belief captured boldly.", "Hope glows quietly.", "You got this... and that frame!",
        "Rise, always rise.", "Stillness fuels strength.", "A spark to go on.",
        "Visual victory vibes.", "Shoot your shot – and dream.", "Picture says: 'Keep going!'",
        "Encouragement in high res.", "Courage framed beautifully."
    ],
    "Romantic": [
        "Love captured in every frame.", "Every pixel whispers 'I love you'.", "Love through the lens.",
        "Hearts aligned here.", "A kiss in color.", "Romance in resolution.",
        "Soft stares and warmth.", "You & me: framed forever.", "Light loves you too.",
        "A feeling, not a photo.", "Held by hues of affection.", "Cupid clicked this!",
        "Camera crush moment.", "Soft hearts, sharp focus.", "Shot through the heart.",
        "Forever in pixels.", "Blushing shadows.", "In love with this moment.",
        "Your eyes clicked this.", "A dreamy affection burst."
    ]
}

# Stylish font transformations
font_styles = {
    "Normal": lambda x: x,
    "Underline": lambda x: ''.join(c + '\u0332' for c in x),
    "Strikethrough": lambda x: ''.join(c + '\u0336' for c in x),
    "Circled": lambda x: ''.join(c + '\u20dd' if c != ' ' else ' ' for c in x),
    "Glitch": lambda x: ''.join(c + '\u0489' for c in x),
    "Bold": lambda x: ''.join([chr(0x1D400 + ord(c) - 65) if 'A' <= c <= 'Z' else chr(0x1D41A + ord(c) - 97) if 'a' <= c <= 'z' else c for c in x]),
    "Italic": lambda x: ''.join([chr(0x1D434 + ord(c) - 65) if 'A' <= c <= 'Z' else chr(0x1D44E + ord(c) - 97) if 'a' <= c <= 'z' else c for c in x]),
    "Funky": lambda x: ''.join(random.choice([c.upper(), c.lower()]) for c in x),
    "Shadow": lambda x: f"\u0336{x}"
}

# Language support
languages = {
    "English": "en", "Telugu": "te", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh-CN",
    "Arabic": "ar", "Russian": "ru", "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Punjabi": "pa", "Tamil": "ta", "Urdu": "ur"
}

def generate_caption(image, style, font_style, language):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=30)
    base_caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Choose styled caption if available
    if style in caption_styles and caption_styles[style]:
        styled = random.choice(caption_styles[style])
    else:
        styled = base_caption

    emoji = random.choice(["✨", "❤️", "🔥", "🌟", "😊", "📸"])
    caption = f"{styled} {emoji}"

    if language != "English":
        try:
            caption = GoogleTranslator(source='auto', target=languages[language]).translate(caption)
        except:
            pass

    # Apply font style
    caption = font_styles.get(font_style, lambda x: x)(caption)

    return caption

with gr.Blocks(css="""
#title {font-size: 28px; text-align: center; color: #ff6b6b;}
.gr-button {background-color: #222; color: white;}
.gr-radio-item label {font-weight: bold;}
#madeby {text-align: center; font-size: 16px; margin-top: 20px; color: #ffdd57;}
#style label {color: #00f2ff;}
#font label {color: #ff66c4;}
#lang label {color: #8aff8a;}
#caption {background-color: #000; color: white; border: 1px solid #444;}
""") as demo:
    gr.Markdown('<div id="title">Stylish AI Image Caption Generator</div>')

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Upload Image")
            style = gr.Radio(list(caption_styles.keys()), label="Caption Style", value="Aesthetic", elem_id="style")
            font = gr.Radio(list(font_styles.keys()), label="Font Style", value="Normal", elem_id="font")
            language = gr.Radio(list(languages.keys()), label="Language", value="English", elem_id="lang")

            caption_output = gr.Textbox(label="Generated Caption", lines=2, elem_id="caption")

            generate_button = gr.Button("Generate Caption")

            generate_button.click(generate_caption, inputs=[image, style, font, language], outputs=caption_output)

    gr.Markdown('<div id="madeby">Made with ❤️ Munna</div>')

demo.launch()