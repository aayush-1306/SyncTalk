import glob
all_files = glob.glob("*/*.py")
for file in all_files:
  with open(file) as f:
    lines = f.readlines()
    lines = [line.replace('-std=c++14', '-std=c++17') for line in lines]
    if 'backend' in file:
      print(lines)
  with open(file, 'w') as f:
    f.writelines(lines)