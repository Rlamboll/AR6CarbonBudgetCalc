import os
## Not of general use, used to move around setup files

folder = "../InputData/fair163_sr15/SR15_all_temps/"

files = os.listdir(folder)
for file in files:
    if file[-4:] == ".hdf":
        reprstr = file.replace("+", "-").replace("％", "-").replace("&", "-").replace(
            "$", "-").replace("(", "-").replace(")", "-").replace(",", "-").replace("°", "-").replace("$", "-")
        if reprstr != file:
            print(f"Rename {file}")
            os.rename(folder + file, folder + reprstr)
"""
for f in os.walk(folder):
    file = f[-1]
    if (len(file) != 0):
        os.rename(f[0].replace("\\", "/")+"/"+file[0], folder+file[0])
"""