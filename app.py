# =============================================================
# MedBot — Final Clean UI + Branding + PDF Export
# =============================================================

import uuid
import streamlit as st
from io import BytesIO
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from config import DOMAIN_DESCRIPTION
from agent import app as agent_app

st.set_page_config(page_title="MedBot", layout="wide")

# =============================================================
# STATE
# =============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# =============================================================
# HELPER
# =============================================================

def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })

# =============================================================
# 🎨 STYLE
# =============================================================

st.markdown("""
<style>

.chat-container {
    max-width: 800px;
    margin: auto;
}

/* Header branding */
.brand {
    text-align: center;
    margin-bottom: 10px;
}

.brand-title {
    font-size: 28px;
    font-weight: 600;
}

.brand-sub {
    color: gray;
    font-size: 14px;
    margin-top: 4px;
}

/* Chat bubbles */
.user {
    background: #2563eb;
    color: white;
    padding: 10px 14px;
    border-radius: 16px;
    margin: 8px 0;
    max-width: 70%;
    margin-left: auto;
}

.bot {
    background: #f1f5f9;
    padding: 10px 14px;
    border-radius: 16px;
    margin: 8px 0;
    max-width: 70%;
}

/* Timestamp */
.timestamp {
    font-size: 10px;
    color: gray;
    margin-top: 2px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:

    st.markdown("## 🩺 MedBot")

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    # SAVE CHAT
    if st.session_state.messages:

        def generate_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                name="Title",
                fontSize=18,
                textColor=colors.HexColor("#2563eb"),
                spaceAfter=10
            )

            user_style = ParagraphStyle(
                name="User",
                backColor=colors.HexColor("#2563eb"),
                textColor=colors.white,
                spaceAfter=6,
                leftIndent=200
            )

            bot_style = ParagraphStyle(
                name="Bot",
                backColor=colors.HexColor("#e2e8f0"),
                textColor=colors.black,
                spaceAfter=6
            )

            time_style = ParagraphStyle(
                name="Time",
                fontSize=8,
                textColor=colors.grey
            )

            content = []

            content.append(Paragraph("🩺 MedBot Chat Report", title_style))
            content.append(Spacer(1, 10))

            for msg in st.session_state.messages:

                role = msg["role"]
                text = msg["content"]
                time = msg.get("time", "")

                if role == "user":
                    content.append(Paragraph(f"You: {text}", user_style))
                else:
                    content.append(Paragraph(f"MedBot: {text}", bot_style))

                content.append(Paragraph(time, time_style))
                content.append(Spacer(1, 10))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf_data = generate_pdf()

        if st.download_button(
            "📄 Save Chat",
            data=pdf_data,
            file_name="medbot_chat.pdf",
            mime="application/pdf"
        ):
            st.success("Chat saved successfully!")

    st.divider()

    st.markdown("### History")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.caption(msg["content"][:40])

# =============================================================
# MAIN PAGE (BRANDING + CHAT)
# =============================================================

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# 🔥 BRAND HEADER (ALWAYS ON TOP)
st.markdown("""
<div class='brand'>
    <div class='brand-title'>🩺 MedBot</div>
    <div class='brand-sub'>Ask anything about health</div>
</div>
""", unsafe_allow_html=True)

# Landing
if not st.session_state.messages:

    st.caption(DOMAIN_DESCRIPTION)
    st.caption("This is general information, not professional medical advice.")

    sample = [
        "What are symptoms of diabetes?",
        "How much exercise per week?",
        "How to improve sleep?",
        "Side effects of antibiotics?"
    ]

    cols = st.columns(2)

    for i, q in enumerate(sample):
        with cols[i % 2]:
            if st.button(q):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                add_message("user", q)
                result = agent_app.invoke({"question": q}, config=config)
                add_message("assistant", result.get("answer", "Error"))
                st.rerun()

# Chat display
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f"<div class='user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='timestamp'>{msg.get('time','')}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================
# INPUT
# =============================================================

if prompt := st.chat_input("Ask a medical question..."):

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    add_message("user", prompt)
    result = agent_app.invoke({"question": prompt}, config=config)
    add_message("assistant", result.get("answer", "Error"))

    st.rerun()

# =============================================================
# FOOTER
# =============================================================

st.caption("MedBot provides general information. Seek professional medical advice.")