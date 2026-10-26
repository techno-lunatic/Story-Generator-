import streamlit as st

from inference import generate_text, load_model


st.set_page_config(
    page_title="Story Laboratory",
    page_icon="📖",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #172329;
        --ink-soft: #43504c;
        --paper: #f4efe2;
        --paper-light: #fbf8ef;
        --brass: #a77d3d;
        --brass-light: #d4b574;
        --line: rgba(69, 59, 40, 0.18);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(212, 181, 116, 0.12), transparent 42%),
            linear-gradient(180deg, #e8e0cf 0%, #f4efe2 38%, #ece5d6 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 239, 226, 0.72);
    }

    [data-testid="stSidebar"] {
        background: #172329;
        border-right: 1px solid rgba(212, 181, 116, 0.35);
    }

    [data-testid="stSidebar"] * {
        color: #f4efe2;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        color: var(--brass) !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.19em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        color: var(--ink) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.7rem, 6vw, 4.6rem);
        font-weight: 500;
        letter-spacing: -0.045em;
        line-height: 1;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--ink-soft) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.2rem;
        line-height: 1.55;
        max-width: 720px;
        margin-top: 1rem;
    }

    .rule {
        border: 0;
        border-top: 1px solid var(--line);
        margin: 1.8rem 0;
    }

    .section-kicker {
        color: var(--brass) !important;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .section-title {
        color: var(--ink) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.8rem;
        font-weight: 500;
        margin: 0 0 0.35rem 0;
    }

    .section-copy {
        color: var(--ink-soft) !important;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .paper-card {
        background: rgba(251, 248, 239, 0.88);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 14px 36px rgba(47, 39, 26, 0.08);
        padding: 1.25rem 1.4rem;
        height: 100%;
    }

    .paper-card h3 {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0.15rem 0 0.5rem 0;
    }

    .paper-card p {
        color: var(--ink-soft);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .card-mark {
        color: var(--brass);
        font-size: 1.25rem;
    }

    [data-testid="stTextArea"] textarea {
        background: var(--paper-light);
        border: 1px solid rgba(69, 59, 40, 0.25);
        border-radius: 10px;
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--brass);
        box-shadow: 0 0 0 1px var(--brass);
    }

    div.stButton > button {
        border: 1px solid #765c32;
        border-radius: 9px;
        background: var(--ink);
        color: var(--paper-light);
        font-weight: 700;
        min-height: 2.8rem;
        padding: 0.55rem 1.1rem;
        transition: all 150ms ease;
    }

    div.stButton > button:hover {
        border-color: var(--brass);
        background: #26383e;
        color: #fff8e9;
        transform: translateY(-1px);
    }

    div.stButton > button:disabled {
        background: #7c817b;
        color: #e8e3d8;
        border-color: #7c817b;
    }

    .passage {
        background:
            linear-gradient(90deg, rgba(167, 125, 61, 0.08), transparent 18%),
            var(--paper-light);
        border: 1px solid rgba(167, 125, 61, 0.35);
        border-left: 4px solid var(--brass);
        border-radius: 4px 13px 13px 4px;
        box-shadow: 0 14px 36px rgba(47, 39, 26, 0.09);
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.08rem;
        line-height: 1.85;
        padding: 1.6rem 1.8rem;
        white-space: pre-wrap;
    }

    .prompt-label {
        color: var(--brass) !important;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .prompt-context {
        color: var(--ink-soft) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-style: italic;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--ink-soft);
        font-weight: 700;
        padding: 0.7rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        border-bottom-color: var(--brass) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.2rem;
        }

        .paper-card {
            margin-bottom: 0.8rem;
        }

        .passage {
            padding: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


st.markdown('<p class="eyebrow">A small experimental writing room</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Story Laboratory</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="hero-subtitle">
        Enter a Harry Potter–inspired story opening, and the model will continue it.
        This model was trained on Harry Potter books.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="rule">', unsafe_allow_html=True)


write_tab, method_tab = st.tabs(["✍️  Writing desk", "⌘  How it works"])


with write_tab:
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        st.markdown('<p class="section-kicker">Begin with a few words</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Set the scene</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Give the model a beginning, a character, or a moment. '
            'It will use your prompt as the opening context for a new passage.</p>',
            unsafe_allow_html=True,
        )

        prompt = st.text_area(
            "Story opening",
            placeholder=(
                "The castle was unusually quiet that evening, "
                "until a strange silver light appeared in the corridor..."
            ),
            height=190,
            label_visibility="collapsed",
        )

        generate_clicked = st.button(
            "✦  Continue the story",
            type="primary",
            disabled=not prompt.strip(),
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="paper-card">
                <div class="card-mark">✧</div>
                <h3>A prompt is a doorway</h3>
                <p>
                    A short opening gives the model a setting and a direction.
                    Try a character, a place, or an unusual event, then see where
                    the next words lead.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="paper-card">
                <div class="card-mark">“</div>
                <h3>Prompt fragment</h3>
                <p>
                    “The portrait on the third-floor landing had always been asleep.
                    Tonight, it was wide awake.”
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if generate_clicked:
        try:
            with st.spinner("The next passage is taking shape..."):
                model, tokenizer, device = get_model()
                top_k = min(st.session_state["top_k"], tokenizer.get_vocab_size())

                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    top_k=top_k,
                    temperature=st.session_state["temperature"],
                    max_new_tokens=st.session_state["max_new_tokens"],
                )

            st.markdown('<hr class="rule">', unsafe_allow_html=True)
            st.markdown('<p class="section-kicker">Fresh from the writing room</p>', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">Generated passage</h2>', unsafe_allow_html=True)
            st.markdown('<p class="prompt-label">Your opening</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="prompt-context">{prompt}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="passage">{output}</div>',
                unsafe_allow_html=True,
            )

        except FileNotFoundError as error:
            st.error(str(error))


with method_tab:
    st.markdown('<p class="section-kicker">Behind the page</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How it works</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">This is a compact language model trained on Harry Potter books. '
        'It generates a continuation one token at a time.</p>',
        unsafe_allow_html=True,
    )

    how_col1, how_col2, how_col3 = st.columns(3, gap="medium")

    with how_col1:
        st.markdown(
            """
            <div class="paper-card">
                <div class="card-mark">01</div>
                <h3>Split into tokens</h3>
                <p>
                    The prompt is encoded into smaller text pieces using the
                    model’s Byte Pair Encoding tokenizer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with how_col2:
        st.markdown(
            """
            <div class="paper-card">
                <div class="card-mark">02</div>
                <h3>Predict the next piece</h3>
                <p>
                    The transformer reads the recent context and calculates
                    probabilities for what token could come next.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with how_col3:
        st.markdown(
            """
            <div class="paper-card">
                <div class="card-mark">03</div>
                <h3>Repeat and decode</h3>
                <p>
                    It samples a token, adds it to the context, and repeats.
                    The tokenizer then turns the generated tokens back into text.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<p class="section-kicker">A note on the controls</p>', unsafe_allow_html=True)

    note_col1, note_col2, note_col3 = st.columns(3, gap="medium")

    with note_col1:
        st.markdown(
            """
            <div class="paper-card">
                <h3>top_k</h3>
                <p>
                    Limits each choice to the most likely next tokens.
                    Smaller values make choices more focused.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with note_col2:
        st.markdown(
            """
            <div class="paper-card">
                <h3>Temperature</h3>
                <p>
                    Adjusts how sharply the model favors likely tokens.
                    Lower values are more predictable; higher values are more varied.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with note_col3:
        st.markdown(
            """
            <div class="paper-card">
                <h3>max_new_tokens</h3>
                <p>
                    Sets the maximum number of new tokens in the continuation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown('<p class="eyebrow">The instrument panel</p>', unsafe_allow_html=True)
    st.markdown("### Generation settings")
    st.caption("Adjust how the next passage is sampled.")

    st.slider(
        "top_k",
        min_value=1,
        max_value=8000,
        value=40,
        step=1,
        key="top_k",
        help="Keep sampling among the most likely next tokens.",
    )

    st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        key="temperature",
        help="Lower is more predictable; higher is more varied.",
    )

    st.slider(
        "max_new_tokens",
        min_value=1,
        max_value=512,
        value=300,
        step=1,
        key="max_new_tokens",
        help="Maximum number of generated tokens.",
    )

    st.markdown("---")
    st.caption("The trained model and tokenizer are loaded from `artifacts/final_model/`.")
