import streamlit as st
import uuid
import threading
import time
from datetime import datetime
from main import build_agent
from config import Config
from utils.streaming import get_streaming_buffer, clear_streaming_buffer
import ui

# --- State Management ---
def init_state():
    """Initialize session state variables."""
    if 'threads' not in st.session_state:
        st.session_state.threads = {}
        create_new_thread()
    
    if 'agent' not in st.session_state:
        st.session_state.agent = build_agent()
        
    if 'current_thread_id' not in st.session_state:
         if st.session_state.threads:
             st.session_state.current_thread_id = list(st.session_state.threads.keys())[0]
         else:
             create_new_thread()

def create_new_thread():
    """Create a new chat thread."""
    new_thread_id = str(uuid.uuid4())
    st.session_state.threads[new_thread_id] = {
        'title': 'New Chat',
        'created': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'messages': [],
        'total_tokens': 0
    }
    st.session_state.current_thread_id = new_thread_id

def get_current_thread():
    """Get the current active thread."""
    return st.session_state.threads[st.session_state.current_thread_id]

def update_thread_title(thread_id, query):
    """Update thread title based on first query."""
    thread = st.session_state.threads[thread_id]
    if thread['title'] == 'New Chat' and query:
        thread['title'] = query[:50] + "..." if len(query) > 50 else query

def switch_thread(thread_id):
    """Switch to a different thread."""
    st.session_state.current_thread_id = thread_id
    st.rerun()

def delete_thread(thread_id):
    """Delete a thread."""
    if len(st.session_state.threads) > 1:
        del st.session_state.threads[thread_id]
        if st.session_state.current_thread_id == thread_id:
            st.session_state.current_thread_id = list(st.session_state.threads.keys())[0]
        st.rerun()

# --- Agent Interaction ---
def run_agent_in_thread(agent, query, status_container, nodes_container, report_container, conversation_history):
    """Runs the agent in a separate thread to allow UI updates."""
    
    initial_state = {"query": query, "history": conversation_history}
    
    # Shared state for communication between agent thread and UI
    shared_state = {
        'nodes_executed': [],
        'final_report': '',
        'final_state': {},
        'query_id': None,
        'streaming_content': '',
        'agent_complete': False,
        'error': None
    }
    
    def target():
        try:
            # Use the passed agent object directly, avoiding st.session_state in thread
            for output in agent.stream(initial_state):
                for key, value in output.items():
                    shared_state["nodes_executed"].append(key)
                     # Capture query_id from guard for streaming
                    if key == "guard" and "query_id" in value:
                        shared_state["query_id"] = value["query_id"]
                    
                    # Capture final state
                    if key == "formatter":
                        shared_state["final_report"] = value.get("final_report", "")
                        shared_state["final_state"] = value
            
            shared_state["agent_complete"] = True
        except Exception as e:
            shared_state["error"] = str(e)
            shared_state["agent_complete"] = True

    # Start the agent thread
    agent_thread = threading.Thread(target=target)
    agent_thread.start()
    
    # Wait for query_id to setup streaming
    time.sleep(0.3)
    
    # UI Loop: Update while agent runs
    while not shared_state["agent_complete"]:
        # Update Execution Path
        if shared_state["nodes_executed"]:
             with nodes_container.container():
                ui.render_execution_path(shared_state["nodes_executed"])

        # Update Status
        if shared_state["nodes_executed"]:
             status_container.info(f"⚡ Processing: **{shared_state['nodes_executed'][-1]}**")

        # Handle Streaming
        if shared_state["query_id"]:
            buffer = get_streaming_buffer(shared_state["query_id"])
            try:
                # Non-blocking get
                chunk = buffer.queue.get(timeout=0.05)
                if chunk is not None:
                    shared_state["streaming_content"] += chunk
                    report_container.markdown(shared_state["streaming_content"] + " ▌")
            except:
                pass
        
        time.sleep(0.05)
        
    agent_thread.join(timeout=1)
    
    # Post-processing
    if shared_state["error"]:
        st.error(f"❌ Error: {shared_state['error']}")
        return None, None, []

    # Cleanup Streaming
    if shared_state["query_id"]:
        clear_streaming_buffer(shared_state["query_id"])
        
    return shared_state["final_report"] or shared_state["streaming_content"], shared_state["final_state"], shared_state["nodes_executed"]


# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Developer Research AI Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    ui.apply_custom_css()
    init_state()
    
    # Sidebar
    with st.sidebar:
        ui.render_sidebar_header()
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_thread()
            st.rerun()
            
        st.markdown("---")
        
        ui.render_thread_list(
            st.session_state.threads,
            st.session_state.current_thread_id,
            switch_thread,
            delete_thread
        )
        
        st.markdown("### ⚙️ Configuration")
        st.info(f"**Model**: {Config.MODEL_NAME}")
        st.info(f"**Max Iterations**: {Config.MAX_ITERATIONS_DEEP_MODE}")
        
    # Main Content
    ui.render_header()
    
    current_thread = get_current_thread()
    st.markdown(f"### 💬 {current_thread['title']}")
    st.markdown("---")
    
    # Render History
    ui.render_chat_history(current_thread['messages'])
    
    # Input
    query = st.chat_input("Enter your technical research query...")
    
    if query:
        update_thread_title(st.session_state.current_thread_id, query)
        
        # Add user message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_thread['messages'].append({
            'role': 'user',
            'timestamp': timestamp,
            'content': query
        })
        
        # Rerun to show user message immediately (optional, might flicker)
        # But we want to show the user message in the list
        with st.chat_message("user"):
             st.markdown(f"**{timestamp}**")
             st.markdown(query)

        # Container for assistant response
        with st.chat_message("assistant"):
            nodes_container = st.empty()
            status_container = st.empty()
            report_container = st.empty()
            
            
            # Build conversation history in the format the agent expects
            # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            conversation_history = [
                {"role": msg['role'], "content": msg['content']} 
                for msg in current_thread['messages']
            ]
            
            final_report, final_state, executed_nodes = run_agent_in_thread(
                st.session_state.agent, 
                query, 
                status_container, 
                nodes_container, 
                report_container,
                conversation_history
            )
            
            if final_report:
                # Clear moving parts
                status_container.empty()
                
                # Show final result
                report_container.markdown(final_report)
                
                # Show metadata
                tokens = final_state.get('token_usage', 0)
                ui.render_message_metadata(
                    final_state.get('mode', 'N/A'),
                    final_state.get('confidence_score', 0),
                    tokens
                )
                
                # Save to history
                current_thread['messages'].append({
                    'role': 'assistant',
                    'timestamp': timestamp,
                    'content': final_report,
                    'nodes': executed_nodes,
                    'mode': final_state.get('mode', 'N/A'),
                    'confidence': final_state.get('confidence_score', 0),
                    'tokens': tokens
                })
                current_thread['total_tokens'] += tokens

    ui.render_footer()

if __name__ == "__main__":
    main()
