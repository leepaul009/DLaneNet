import os,sys
import cv2
import json
import numpy as np
import json
from matplotlib import pyplot as plt

def spline_interp_step(pts, step=1):
    res = []
    if len(pts) <= 2:
        res = pts
        return res
    tmp_param = cal_params(pts)

    if len(tmp_param) == 0:
        print('error during cal spline param')
        return res
    for f in tmp_param:
        st = []
        tmp = 0
        while tmp < f['h']:
            st.append(tmp)
            tmp += step

        for t in st[:]:
            x = getX(f, t)
            y = getY(f, t)
            # res.append({'x': x, 'y': y})
            res.append([x, y])
    res.append(pts[len(pts)-1])
    return res

def spline_interp_step_int(pts, step=1):
    res = []
    if len(pts) <= 2:
        res = pts
        res = [[int(x), int(y)] for x,y in pts]
        return res
    tmp_param = cal_params(pts)

    if len(tmp_param) == 0:
        print('error during cal spline param')
        return res
    for f in tmp_param:
        st = []
        tmp = 0
        while tmp < f['h']:
            st.append(tmp)
            tmp += step

        for t in st[:]:
            x = getX(f, t)
            y = getY(f, t)
            # res.append({'x': x, 'y': y})
            res.append([int(x), int(y)])
    last_pt = pts[ len(pts)-1 ]
    res.append([int(last_pt[0]), int(last_pt[1])])
    return res

def cal_params(pts):
    params = []
    if len(pts) <= 2:
        return params

    h=[]
    for i in range(0,len(pts)-1):
        dx = pts[i][0] - pts[i+1][0]
        dy = pts[i][1] - pts[i+1][1]
        dis = np.sqrt(dx*dx + dy*dy)
        h.append(dis)

    A=[]
    B=[]
    C=[]
    Dx=[]
    Dy=[]
    for i in range(0,len(pts)-2):
        A.append(h[i])
        B.append( 2*(h[i]+h[i+1]) )
        C.append(h[i+1])

        dx1 = (pts[i+1][0] - pts[i][0]) / h[i]
        dx2 = (pts[i+2][0] - pts[i+1][0]) / h[i+1]
        Dx.append(6*(dx2-dx1))

        dy1 = (pts[i+1][1] - pts[i][1]) / h[i]
        dy2 = (pts[i+2][1] - pts[i+1][1]) / h[i+1]
        Dy.append(6*(dy2-dy1))

    C[0] /= B[0]
    Dx[0] /= B[0]
    Dy[0] /= B[0]

    for i in range(1,len(pts)-2):
        tmp = B[i] - A[i] *C[i-1]
        C[i] /= tmp
        Dx[i] = (Dx[i] - A[i]*Dx[i-1]) / tmp
        Dy[i] = (Dy[i] - A[i]*Dy[i-1]) / tmp

    Mx = np.zeros(len(pts))
    My = np.zeros(len(pts))
    Mx[len(pts)-2] = Dx[len(pts)-3]
    My[len(pts)-2] = Dy[len(pts)-3]
    for i in range(len(pts)-4,-1,-1):
        Mx[i+1] = Dx[i] - C[i] * Mx[i+2]
        My[i+1] = Dy[i] - C[i] * My[i+2]

    Mx[0] = 0
    Mx[-1] = 0
    My[0] = 0
    My[-1] = 0

    for i in range(0, len(pts)-1):
        param = {}
        param['a_x'] = pts[i][0]
        param['b_x'] = (pts[i+1][0] - pts[i][0]) / h[i] -(2*h[i]*Mx[i] + h[i]*Mx[i+1])/6
        param['c_x'] = Mx[i] / 2
        param['d_x'] = (Mx[i+1]-Mx[i])/(6*h[i])

        param['a_y'] = pts[i][1]
        param['b_y'] = (pts[i+1][1] - pts[i][1]) / h[i] - (2*h[i]*My[i]+h[i]*My[i+1])/6
        param['c_y'] = My[i]/2
        param['d_y'] = (My[i+1] - My[i]) / (6*h[i])
        param['h'] = h[i]
        params.append(param)

    return params


def getX(f, t):
    return f['a_x'] + f['b_x']*t + f['c_x']*t*t + f['d_x']*t*t*t


def getY(f, t):
    return f['a_y'] + f['b_y']*t + f['c_y']*t*t + f['d_y']*t*t*t

'''
color_dic={\
        "normal_line":          (254, 254, 0), \
        "discarded_line":       (250, 100, 0), \
        "ignore_line":          (44,  92,  255), \
        "not_available":        (250, 26,  255), \
        "barrier_line":         (144, 144, 255), \
        "speed_reduction_line": (29,  91,  185), \
        "edge_line":            (43,  131, 94), \
        "brick_line":           (27,  21,  230) \
        }
'''
color_dic={\
        "solid": (254, 254, 0),\
        "dashed": (250, 100, 0),\
        "unknown": (27,21,230),\
        }

