import os
import FinderZ

org_path = ""

#Path to move all files to (mixed is the class)
mv_path = ""

org_path_dirs = FinderZ.GatherInfo.readDir(org_path, returnFiles = False)

for dir in org_path_dirs:
    #Get the files:
    dir_path = os.path.join(org_path, dir)
    dir_files = FinderZ.GatherInfo.readDir(dir_path, returnDirectories = False)
    for file in dir_files:
        try:
            FinderZ.fileOperands.moveFile(os.path.join(dir_path, file), mv_path)
        except:
            pass