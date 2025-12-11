import json
import os

def clean_and_parse_json(file_path):
    """
    Reads a file, handles comments, and returns parsed JSON.
    """
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None, False
    except json.JSONDecodeError:
        pass 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//"): continue 
            if "//" in line: line = line.split("//")[0]
            clean_lines.append(line)
            
        fixed_json = "".join(clean_lines)
        data = json.loads(fixed_json)
        msg = f"⚠️ **Format Warning ({file_name}):** Auto-repaired comments."
        return data, msg, True
    except Exception as e:
        return None, f"❌ **Error ({file_name}):** {str(e)}", False

def load_data(chat_path, context_path):
    logs = []
    
    # 1. Validation
    if not os.path.exists(chat_path): return [], [f"❌ Error: {chat_path} not found"]
    if not os.path.exists(context_path): return [], [f"❌ Error: {context_path} not found"]

    chat_data, chat_msg, _ = clean_and_parse_json(chat_path)
    ctx_data, ctx_msg, _ = clean_and_parse_json(context_path)
    
    if chat_msg: logs.append(chat_msg)
    if ctx_msg: logs.append(ctx_msg)
    if not chat_data or not ctx_data: return [], logs

    try:
        # 2. Extract Context
        vector_data = ctx_data.get('data', {}).get('vector_data', [])
        if not vector_data: vector_data = ctx_data.get('vector_data', [])
        context_text = "\n\n".join([v.get('text', '') for v in vector_data if v.get('text')])

        # 3. Extract Target Response (Source of Truth)
        sources = ctx_data.get('data', {}).get('sources', {})
        target_res_raw = sources.get('final_response', "")
        
        # Normalize target response to string
        target_response_str = ""
        if isinstance(target_res_raw, list):
            target_response_str = " ".join([str(x) for x in target_res_raw])
        else:
            target_response_str = str(target_res_raw)

        # 4. Find Match in Chat History
        user_query = "Unknown Query"
        found_response = None
        
        turns = chat_data.get('conversation_turns', [])
        
        # Attempt to find the AI turn that matches the context response
        for i, turn in enumerate(turns):
            if turn['role'] == 'AI/Chatbot':
                msg = turn.get('message', '')
                # Check match (first 30 chars)
                if len(target_response_str) > 10 and (target_response_str[:30] in msg or msg[:30] in target_response_str):
                    found_response = msg
                    if i > 0: user_query = turns[i-1]['message']
                    break
        
        # 5. Handle "Missing Turn" Case (Common in Set 02)
        # If we didn't find the answer in the chat log, it means the chat log ended early.
        # We must construct the pair using the Context Response + Last User Query.
        if not found_response:
            if turns:
                # Find last user message
                for i in range(len(turns)-1, -1, -1):
                    if turns[i]['role'] == 'User':
                        user_query = turns[i]['message']
                        break
                
                # Use the response from the Context file directly
                found_response = target_response_str
                logs.append("ℹ️ **Note:** AI response missing from Chat Log. Extracted directly from Vector Data.")
            else:
                logs.append("❌ Error: Chat log is empty.")
                return [], logs

        return [{
            "query": user_query,
            "response": found_response,
            "context": context_text
        }], logs

    except Exception as e:
        logs.append(f"❌ **Processing Error:** {str(e)}")
        return [], logs