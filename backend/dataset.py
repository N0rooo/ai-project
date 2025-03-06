import kagglehub
import os


def load_movie_dialogs(max_sequences=50000): 
    path = kagglehub.dataset_download("rajathmc/cornell-moviedialog-corpus")
    
    movie_lines_path = os.path.join(path, "movie_lines.txt")
    movie_conversations_path = os.path.join(path, "movie_conversations.txt")
    
    lines = {}
    with open(movie_lines_path, 'r', encoding='iso-8859-1') as f:
        for i, line in enumerate(f):
            if i >= max_sequences:  
                break
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                line_id, _, _, _, text = parts
                lines[line_id] = text.lower()
    
    # Read conversations
    conversations = []
    with open(movie_conversations_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            if len(conversations) >= max_sequences:  
                break
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:
                line_ids = eval(parts[3])
                conversation = [lines[line_id] for line_id in line_ids if line_id in lines]
                conversations.extend(conversation[:max_sequences - len(conversations)])
    
    print(f"Loaded {len(conversations)} sequences")
    return " ".join(conversations)
