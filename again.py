
from PIL import Image, ImageDraw
import math
import numpy as np
import time
from pprint import pprint
from scipy.fftpack import dct, idct
from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
global header
mquant = np.array([[3,5,7,9,11,13,15,17],
[5,7,11,13,15,17,19,21],
[7,11,13,15,17,19,21,23],
[9,11,13,15,17,19,21,23],
[11,13,15,17,19,21,23,25],
[13,15,17,19,21,23,25,27],
[15,17,19,21,23,25,27,29],
[17,19,21,23,25,27,29,31]])
M = 8
N = 8
#np.hsplit(a, 3) 

def split_2d(array, splits):
    x, y = splits
    return np.split(np.concatenate(np.split(array, y, axis=1)), x*y)



def Cu(idx):
    '''
    coeficiente U
    '''
    if idx>0:
        return math.sqrt(2/M)
    return 1/math.sqrt(M)

def Cv(idx):
    '''
    coeficiente V
    '''
    if idx>0:
        return math.sqrt(2/N)
    return 1/math.sqrt(N)
def get_dct_matrix(m):

    dct_matrix = np.empty((m, m))

    for p in range(m):
        for q in range(m):
            if p == 0:  # Case: First row
                dct_matrix[p, q] = 1/math.sqrt(m)
            else:  # Every other row
                dct_matrix[p, q] = (math.sqrt(2/m))*math.cos((math.pi*((2*q)+1)*p)/(2*m))
    return dct_matrix
def isum_of_sum(matrix, M,N,r,s):
    '''
    Soma ponto a ponto, seguida da formula da iDCT2D
    '''
    return np.sum([np.sum([Cu(i)*Cv(j)*matrix[i,j]*math.cos(((2*r+1)*i*math.pi)/(2*M)) * math.cos(((2*s+1)*j*math.pi)/(2*N)) for j in range(N)]) for i in range(M)])
def idct2(matrix_after_dct):
    idct_matrix = np.zeros([M,N])
    for r in range(M):
        for s in range(N):
            idct_matrix[r,s] = int(isum_of_sum(matrix_after_dct, M, N, r, s))
    return idct_matrix
# Input Matrix = A
# DCT Matrix = T
# Transpose of DCT Matrix  = T'
# Computes T * A * T' to calculate DCT of A
def dctt(a):
    if len(a) == len(a[0]):
        t = get_dct_matrix(len(a))
        b = np.matmul(t, a)
        out = np.matmul(b, t.T)
    else:
        raise InvalidMatrixSizeError("Matrix must be a square matrix")
    out = np.matmul(out,mquant)
    return out
def idctt(a):
    if len(a) == len(a[0]):
        t = get_dct_matrix(len(a))
        b = np.matmul(t.T, a)
        out = np.matmul(b, t)
    else:
        raise InvalidMatrixSizeError("Matrix must be a square matrix")
    
    return out

def divide_np(matrix, n_parts, m_parts):
    N, M = len(matrix), len(matrix[0])
    n, m = N // n_parts, M // m_parts
    
    return [matrix[i:i+n, j:j+m] for i in range(0, N, n) for j in range(0, M, m)]

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))
def get_chunks(M, n, p):
    n = len(M)//n
    p = len(M[0])//p
    for i in range(0, len(M), n):
        for j in range(0, len(M[0]), p):
            yield M[i:i+n,j:j+p]

def get_chunks2(M, n, p):
        for i in range(0, len(M), n):
            for j in range(0, len(M[0]), p):
                yield M[i:i+n,j:j+p]

def read_rows(path):
    image_file = open(path, "rb")
    # Blindly skip the BMP header.
    global header
    header = image_file.read(54)
    image_file.close()
def writeycbcr(comp, name):
    image_file = open(name+'.bmp', 'wb')
    image_file.write(bytearray(header))
    res = []
    for (i,j),el in np.ndenumerate(comp):
            res.append(int(el))
            res.append(int(el))
            res.append(int(el))
    image_file.write(bytearray(res))
    image_file.close()
def main():
    read_rows('lenna.png')
    image = Image.open("lenna.png")
    w,h = image.size
    image = image.rotate(90)
    imagered = image.copy()
    imagegreen = image.copy()
    imageblue = image.copy()
    blue = []
    green = []
    red = []
    pix = image.load()
    width = image.size[0] #Определяем ширину.
    height = image.size[1] #Определяем высоту.
    draw = ImageDraw.Draw(image) #Создаем инструмент для рисования.
    drawred = ImageDraw.Draw(imagered)
    drawgreen = ImageDraw.Draw(imagegreen)
    drawblue = ImageDraw.Draw(imageblue)
    tos = []
    for i in range(width):
           for j in range(height):
                     a = pix[i, j][0]
                     b = pix[i, j][1]
                     c = pix[i, j][2]
                     #draw.point((i, j), (0, 0, c))
                     blue.append(c)
                     red.append(a)
                     green.append(b)
      
    ycomp = [int(0.299*red[i]+0.587*green[i]+0.114*blue[i]) for i in range(len(green))]
    nnn = 7
    # for i in range(60):
    #     nnn+=8
    #     print(nnn,end=', ')
    #writeycbcr(ycomp,'dfs')
    y = np.array(ycomp)
    y = y.reshape(h,w)
    y = np.fliplr(y)
    #plt.gray()
    #plt.imshow(y)
    writeycbcr(y,'last')
    
    # list(get_chunks2(M, 9, 5)) same result more faster
    allmx = blockshaped(y,8,8)
    final = unblockshaped(allmx,h,w)
    # allmx = np.hsplit(allmx, (15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231))
    # for i in allmx:
    #     i = np.vsplit(i, (15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231))
    pprint(np.stack(allmx).shape) 
    #allmx = view_as_blocks(y, (8, 8))
    rd = np.dstack(allmx).reshape(h,w)
    # flatten_view = allmx.reshape(allmx.shape[0], allmx.shape[1], -1)
    # flatten_view = allmx.reshape(h,w)
   
    res = []
    maxx = 0
    #resmx = np.array(900,8,8)
    for i in allmx:
        i = dct(i)
        print(i)
    for i in allmx:
        i = idct(i)
        print(i)
    for i in allmx:
        res.append(np.max(i))
    print(len(res))
    #for i in allmx:
        #i[i!=np.max(i)] = np.max(i)/2
    for i in allmx:
        print(i)
    for i in res:
        if i<0:
            i = 0
        if i >255:
            i = 255
    out = unblockshaped(allmx,h,w)
    print(allmx)
    imm = Image.fromarray(out.astype('uint8'))
    imm.show()
    #print(res)
if __name__ == '__main__':
    main()