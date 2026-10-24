import streamlit as st

from inference import generate_text, load_model


st.set_page_config(
    page_title="Story Generator",
    page_icon="📖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at top, #263653 0%, #101827 55%, #0b101a 100%);
        color: #f4efe4;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(12, 18, 30, 0.92);
        border-right: 1px solid rgba(218, 183, 112, 0.25);
    }

    h1, h2, h3, label, p {
        color: #f4efe4 !important;
    }

    .story-card {
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(218, 183, 112, 0.35);
        border-radius: 14px;
        background: rgba(248, 242, 226, 0.07);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.22);
        margin: 1rem 0;
    }

    div.stButton > button {
        width: 100%;
        border: 1px solid #d8b875;
        border-radius: 10px;
        background: linear-gradient(135deg, #b88b43, #d8b875);
        color: #172033;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    div.stButton > button:hover {
        border-color: #f0d79d;
        background: linear-gradient(135deg, #d1aa63, #f0d79d);
        color: #101827;
    }

    [data-testid="stTextArea"] textarea {
        background-color: rgba(248, 242, 226, 0.95);
        color: #172033;
        border-radius: 10px;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        accent-color: #d8b875;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📖 Story Generator")
st.caption("A little magic, one prompt at a time.")
st.write("Enter a story opening and let the model continue it.")

@st.cache_resource
def get_model():
    return load_model()


with st.sidebar:
    st.header("✨ Generation settings")

    top_k = st.slider(
        "top_k",
        min_value=1,
        max_value=8000,
        value=40,
        step=1,
        help="Lower values keep the next word choices more focused.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Lower values make generation more predictable.",
    )

    max_new_tokens = st.slider(
        "max_new_tokens",
        min_value=1,
        max_value=512,
        value=300,
        step=1,
        help="Maximum number of tokens to generate.",
    )


prompt = st.text_area(
    "Your prompt",
    placeholder="For example: The castle was unusually quiet that evening...",
    height=140,
)

if st.button("✨ Generate story", type="primary", disabled=not prompt.strip()):
    try:
        with st.spinner("The quill is at work..."):
            model, tokenizer, device = get_model()
            top_k = min(top_k, tokenizer.get_vocab_size())

            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        st.subheader("Your story")
        st.markdown(
            f'<div class="story-card">{output}</div>',
            unsafe_allow_html=True,
        )

    except FileNotFoundError as error:
        st.error(str(error))