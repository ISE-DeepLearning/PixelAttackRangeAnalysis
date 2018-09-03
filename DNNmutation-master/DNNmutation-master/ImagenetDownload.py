# from urllib import request
import urllib2, urllib
import signal

path ='/home/ubuntu/Desktop/socket.txt'


def handler(signum, frame):
    raise AssertionError

file = open(path)
count = 0
for line in file:
    count += 1
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)

        print(line)
        # # fake header
        # headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
        # req = urllib.request.Request(url=line, headers=headers)
        # urllib.request.urlopen(req).read()

        try:
            f = urllib2.urlopen(line)
            data = f.read()
            tmpPath = r'/home/ubuntu/Desktop/Untitled/%d'% count
            with open(tmpPath, "wb") as code:
                code.write(data)
        except:
            pass

        #pic_link = line
        #save_path = r'/home/ubuntu/Desktop/Untitled/%s.JPG '% line.split('/')[-1]
        #urllib.urlretrieve(pic_link, save_path)
    except AssertionError:
        print("%s timeout " % line)
        continue

file.close()