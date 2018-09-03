from PIL import Image

path ='/home/ubuntu/Desktop/socket.txt'

for i in range (1000):
    try:
        infile = r'/home/ubuntu/Desktop/Untitled/%d'% i
        im = Image.open(infile)
        (x, y) = im.size
        x_s = 32
        y_s = 32
        outfile = r'/home/ubuntu/Desktop/Untitled/%d.jpg' % i
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        out.save(outfile)

    except:
        pass


