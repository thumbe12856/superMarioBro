from imgpy import Img
from PIL import Image

img = Image.open('pigs.gif')

counter = 0
collection = []
current = img.convert('RGBA')
while True:
    try:
        current.save('original%d.png' % counter)
        img.seek(img.tell()+1)
        current = Image.alpha_composite(current, img.convert('RGBA'))
        counter += 1
    except EOFError:
        break

"""
with Img(fp='m.gif') as im:
    im.load(limit=100000000, first=True)
    a = im.seek(1)
    print(a)
"""