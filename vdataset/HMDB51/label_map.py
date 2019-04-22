import os

label_map = dict()

dir_path = os.path.dirname(os.path.realpath(__file__))
list_file = os.path.join(dir_path, "hmdb_labels.txt")
f = open(list_file, "r")
for _line in f:
    text = _line.split('\n')[0]
    text = text.split(' ')
    label_map[text[1]] = int(text[0])
f.close()

if __name__ == "__main__":
    print(label_map)