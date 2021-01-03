#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:24:31 2021

@author: welliving

"""
class PensionFactor():
    def __init__(self, r, t):
        self.r = r
        self.t = t

    # 終価係数
    def spcaf(self):
        """
        終価係数        
        """
        return (1 + self.r)**self.t

    # 現価係数
    def sppwf(self):
        """
        現価係数        
        """
        return 1 / (1 + self.r)**self.t

    # 年金終価係数
    def uscaf(self):
        """
        年金終価係数        
        """
        return ((1 + self.r)**self.t - 1) / self.r

    # 減債基金係数
    def sff(self):
        """
        減債基金係数        
        """
        return self.r/ ((1 + self.r)**self.t - 1)

    # 年金現価係数
    def uspwf(self):
        """
        年金現価係数        
        """
        return - ((1 + self.r)**(-self.t) - 1) / self.r

    # 資本回収係数
    def crf(self):
        """
        資本回収係数        
        """
        return -self.r / ((1 + self.r)**(-self.t) - 1)


if __name__ == "__main__":
    pf = PensionFactor(0.01, 10)
    print('終価係数:', pf.spcaf(),
          '\n現価係数:', pf.sppwf(),
          '\n年金終価係数:', pf.uscaf(),
          '\n減債基金係数:', pf.sff(),
          '\n年金現価係数:', pf.uspwf(),
          '\n資本回収係数:', pf.crf())
