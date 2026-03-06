import streamlit as st
from datetime import datetime

def apply_custom_css():
    """Applies custom CSS for professional styling."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #2E86AB 0%, #06BCC1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
        }
        .node-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.25rem;
        }
        .node-guard { background-color: #E3F2FD; color: #1976D2; }
        .node-context { background-color: #F3E5F5; color: #7B1FA2; }
        .node-classify { background-color: #E8F5E9; color: #388E3C; }
        .node-planner { background-color: #FFF3E0; color: #F57C00; }
        .node-quick { background-color: #E0F2F1; color: #00796B; }
        .node-quickmode { background-color: #E0F2F1; color: #00796B; }
        .node-deep { background-color: #FCE4EC; color: #C2185B; }
        .node-deepresearch { background-color: #FCE4EC; color: #C2185B; }
        .node-gap { background-color: #FFF9C4; color: #F57F17; }
        .node-gapanalysis { background-color: #FFF9C4; color: #F57F17; }
        .node-synthesize { background-color: #E1BEE7; color: #6A1B9A; }
        .node-formatter { background-color: #C5CAE9; color: #303F9F; }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the main application header."""
    st.markdown('<h1 class="main-header">🔍 Developer Research AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask complex technical questions and get detailed, research-backed answers</p>', unsafe_allow_html=True)

def render_node_badge(node_name):
    """Renders a styled badge for a node."""
    return f'<span class="node-badge node-{node_name.lower().replace("_", "")}">{node_name}</span>'

def render_execution_path(nodes):
    """Renders the execution path of nodes."""
    if not nodes:
        return ""
    
    nodes_html = " → ".join([render_node_badge(node) for node in nodes])
    st.markdown(f"**Execution Path:** {nodes_html}", unsafe_allow_html=True)

def render_message_metadata(mode, confidence, tokens):
    """Renders metadata for a message."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🎯 Mode: {mode}")
    with col2:
        st.caption(f"📊 Confidence: {confidence:.2f}")
    with col3:
        st.caption(f"🔢 Tokens: {tokens}")


def render_chat_history(messages):
    """Renders the chat history."""
    for message in messages:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"**{message['timestamp']}**")
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant"):
                # Show node execution path
                if 'nodes' in message and message['nodes']:
                    render_execution_path(message['nodes'])
                
                # Show report content
                st.markdown(message['content'])
                
                # Show metadata
                render_message_metadata(
                    message.get('mode', 'N/A'),
                    message.get('confidence', 0),
                    message.get('tokens', 0)
                )

def render_sidebar_header():
    """Renders the sidebar header."""
    st.sidebar.markdown("### 💬 Chat Threads")

def render_thread_list(threads, current_thread_id, on_switch, on_delete):
    """Renders the list of threads in the sidebar."""
    sorted_threads = sorted(
        threads.items(),
        key=lambda x: x[1]['created'],
        reverse=True
    )
    
    for thread_id, thread in sorted_threads:
        is_active = thread_id == current_thread_id
        
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"{'📌 ' if is_active else '💬 '}{thread['title']}",
                key=f"thread_{thread_id}",
                use_container_width=True,
                type="secondary" if is_active else "tertiary"
            ):
                on_switch(thread_id)
        
        with col2:
            if st.button("🗑️", key=f"del_{thread_id}", help="Delete thread"):
                on_delete(thread_id)
        
        st.caption(f"📅 {thread['created']} | 💬 {len(thread['messages'])} msgs")
        st.markdown("---")

def render_footer():
    """Renders the application footer."""
    st.markdown("---")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            Powered by LangGraph • Qdrant • Groq
        </div>
        """, unsafe_allow_html=True)
