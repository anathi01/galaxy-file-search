import os
import fnmatch
import tkinter as tk
from tkinter import filedialog, messagebox
from sentence_transformers import SentenceTransformer, util

def read_text_files(root_dir, pattern):
    files_content = []
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, pattern):
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    files_content.append((path, content))
            except Exception:
                continue
    return files_content

def find_most_similar_file(query, files_content, model):
    query_emb = model.encode(query, convert_to_tensor=True)
    contents = [content[:1000] for _, content in files_content]  # Limit to 1000 chars
    content_embs = model.encode(contents, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_emb, content_embs)[0]
    best_idx = similarities.argmax().item()
    return files_content[best_idx][0], similarities[best_idx].item()

def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_dir.delete(0, tk.END)
        entry_dir.insert(0, folder_selected)

def search_file():
    global last_found_path
    search_dir = entry_dir.get()
    file_pattern = entry_pattern.get()
    if not file_pattern.startswith("*."):
        file_pattern = "*." + file_pattern
    query = entry_query.get()

    if not os.path.isdir(search_dir):
        messagebox.showerror("Error", "Please select a valid directory.")
        return

    btn_search.config(state=tk.DISABLED)
    lbl_result.config(text="Loading model (this may take a few seconds)...")
    btn_copy.config(state=tk.DISABLED)
    root.update()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    files_content = read_text_files(search_dir, file_pattern)
    if not files_content:
        lbl_result.config(text="No files found matching the pattern.")
        btn_copy.config(state=tk.DISABLED)
    else:
        best_path, score = find_most_similar_file(query, files_content, model)
        lbl_result.config(text=f"Most relevant file:\n{best_path}\n(similarity score: {score:.2f})")
        last_found_path = best_path
        btn_copy.config(state=tk.NORMAL)
    btn_search.config(state=tk.NORMAL)

def copy_path():
    if last_found_path:
        root.clipboard_clear()
        root.clipboard_append(last_found_path)
        messagebox.showinfo("Copied", "File path copied to clipboard!")

# --- Tkinter GUI ---
last_found_path = None
root = tk.Tk()
root.title("Galaxy Search - Contextual File Finder")

tk.Label(root, text="Directory:").grid(row=0, column=0, sticky="e")
entry_dir = tk.Entry(root, width=50)
entry_dir.grid(row=0, column=1, padx=5, pady=5)
btn_browse = tk.Button(root, text="Browse", command=browse_folder)
btn_browse.grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="File Pattern (e.g., txt):").grid(row=1, column=0, sticky="e")
entry_pattern = tk.Entry(root, width=20)
entry_pattern.insert(0, "txt")
entry_pattern.grid(row=1, column=1, sticky="w", padx=5, pady=5)

tk.Label(root, text="Describe the file:").grid(row=2, column=0, sticky="e")
entry_query = tk.Entry(root, width=50)
entry_query.grid(row=2, column=1, padx=5, pady=5)

btn_search = tk.Button(root, text="Search", command=search_file)
btn_search.grid(row=3, column=1, pady=10)

btn_copy = tk.Button(root, text="Copy Path", command=copy_path, state=tk.DISABLED)
btn_copy.grid(row=3, column=2, pady=10, padx=5)

lbl_result = tk.Label(root, text="", wraplength=500, justify="left")
lbl_result.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()