def count_tokens(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    tokens = text.split()
    num_tokens = len(tokens)

    print(f"Number of tokens in the prompt: {num_tokens}")

#file_path = 'prompt_longer_def.txt'
file_path = 'new_prompt.txt'
count_tokens(file_path)
