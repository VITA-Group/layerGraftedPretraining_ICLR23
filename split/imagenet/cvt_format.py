import re

# cvt the split file from https://github.com/google-research/simclr/blob/master/imagenet_subsets to compatible format

def cvt_txt_format():
  # readFiles = ["1percent.txt", "10percent.txt"]
  readFiles = ["1imgs_class.txt", "2imgs_class.txt", "5imgs_class.txt"]
  # writeFiles = ["imagenet_1percent.txt", "imagenet_10percent.txt"]
  writeFiles = ["imagenet_1imgs_class.txt", "imagenet_2imgs_class.txt", "imagenet_5imgs_class.txt"]

  refer_file = "imagenet_train.txt"
  f = open(refer_file, "r")
  folder2id = dict()

  for line in f.readlines():
    address = line.split(' ')[0]
    classId = int(line.split(' ')[1][:-1])
    folder = line.split('/')[1]
    folder2id[folder] = classId

  print("folder2id is {}".format(folder2id))

  for readFile, writeFile in zip(readFiles, writeFiles):
    f = open(readFile, "r")
    # print(f.readlines())

    linesToWrite = []

    for line in f.readlines():
      group = re.search(f"^(n[0-9]+)_", line)
      assert group is not None
      linesToWrite.append("train/{}/{} {}\n".format(group[1], line.strip(), folder2id[group[1]]))

    with open(writeFile, 'w') as the_file:
      # print("{} {}".format(join('train', folder, path), classId))
      the_file.writelines(linesToWrite)


if __name__ == "__main__":
  cvt_txt_format()
