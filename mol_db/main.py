import glob as glob

if __name__ == '__main__':
    files = glob.glob('chks*/*')
    print(len(files))
