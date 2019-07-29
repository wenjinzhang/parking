import termios, sys , tty

p = 'Rainy'

if not os.path.exists("./testImages"):
    os.mkdir("./testImages")

if not os.path.exists("./testImages/" + p):
    os.mkdir("./testImages/" + p)

rootDir = './PKLot/PKLot/PUCPR/' + p
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname[-4:] == ".jpg":
            image = cv2.imread(dirName + "/" + fname)
            pts = np.array([(343, 501), (1166, 492), (1061, 182), (466, 190)], dtype = "float32")
            warped = four_point_transform(image, pts)
            
            cv2.imwrite("./testImages" + "/" + p + "/" + fname, warped)

def _getch():
   fd = sys.stdin.fileno()
   old_settings = termios.tcgetattr(fd)
   try:
      tty.setraw(fd)
      ch = sys.stdin.read(1)     #This number represents the length
   finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
   return ch

getch = _getch()
print(getch)

