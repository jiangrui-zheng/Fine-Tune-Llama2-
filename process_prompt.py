with open('prompt.txt', 'r') as file:

    content = file.read()
new_content = content.replace('\\', '')
#new_content = content.replace('"', '\\"')
with open('prompt.txt', 'w') as file:

    file.write(new_content)