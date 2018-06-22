import numpy as np
from collections import deque

class Line:
    def __init__(self, pos):
        self.pos = pos
        self.found = False
        self.X = None
        self.Y = None
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)
        self.lastx_int = None
        self.last_top = None
        self.radius = None
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []
        self.count = 0

    def blind_search(self, x, y, img):
        xvals = []
        yvals = []
        if self.found == False:
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(img[j:i, :], axis = 0)
                if self.pos == 'left':
                    peak = np.argmax(histogram[:640])
                else:
                    peak = np.argmax(histogram[640:]) + 640
                x_idx = np.where((((peak - 25) < x) & (x < (peak + 25)) & ((y < j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X
        return xvals, yvals, self.found

    def found_search(self, x, y):
        xvals = []
        yvals = []
        if self.found == True:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i, j])
                xval = (np.mean(self.fit0))*yval**2 + (np.mean(self.fit1))*yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x) & (x < (xval + 25)) & ((y < j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.found = False
        return xvals, yvals, self.found

    def radius_of_curvature(self, xvals, yvals):
        ym_per_pix = 30./720
        xm_per_pix = 3.7/700
        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*np.max(yvals) + fit_cr[1])**2)**1.5) / mp.absolute(2*fit_cr[0])
        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top
