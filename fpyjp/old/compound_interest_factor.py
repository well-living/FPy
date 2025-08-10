# -*- coding: utf-8 -*-
"""
Compound Interest Fomulas

@author: WeLLiving@well-living

"""
import pandas as pd

class InterestFactor:
    def __init__(self, r, t):
        """
        Parameters
        ----------
        r : float
            interest rate.
        t : int
            point of time.
        """
        self.r = r
        self.t = t

    # 終価係数
    def spcaf(self):
        """
        Single Payment Compound Amount Factor: 終価係数        
        """
        return (1 + self.r)**self.t

    # 現価係数
    def sppwf(self):
        """
        Single Payment Present Worth Factor: 現価係数        
        """
        return 1 / (1 + self.r)**self.t

    # 年金終価係数
    def uscaf(self):
        """
        Uniform Series Compound Amount Factor: 年金終価係数        
        """
        return ((1 + self.r)**self.t - 1) / self.r

    # 減債基金係数
    def sff(self):
        """
        Sinking Fund Factor: 減債基金係数        
        """
        return self.r / ((1 + self.r)**self.t - 1)

    # 年金現価係数
    def uspwf(self):
        """
        Uniform Series Present Worth Factor: 年金現価係数        
        """
        return - ((1 + self.r)**(-self.t) - 1) / self.r

    # 資本回収係数
    def crf(self):
        """
        Capital Recovery Factor: 資本回収係数        
        """
        return -self.r / ((1 + self.r)**(-self.t) - 1)


class CompoundInterestSimulation:    
    def __init__(self, r, t):
        """
        Parameters
        ----------
        r : float
            interest rate.
        t : int
            point of time.
        """
        self.r = r
        self.t = t
    
    # 終価係数
    def spcaf(self):
        """
        Single Payment Compound Amount Factor: 終価係数        
        """
        df = pd.DataFrame(0, index=range(0, self.t+1), columns=["現在資産", "将来資産"])
        df.loc[0, "現在資産"]= 1
        df.loc[self.t, "将来資産"] = (1 + self.r)**self.t
        return df

    # 現価係数
    def sppwf(self):
        """
        Single Payment Present Worth Factor: 現価係数        
        """
        df = pd.DataFrame(0, index=range(0, self.t+1), columns=["現在資産", "将来資産"])
        df.loc[self.t, "将来資産"]= 1      
        df.loc[0, "現在資産"] = 1 / (1 + self.r)**self.t
        return df
    
    # 年金終価係数
    def uscaf(self):
        """
        Uniform Series Compound Amount Factor: 年金終価係数        
        """
        df = pd.DataFrame(0, index=range(1, self.t+1), columns=["積立前期末資産", "期末積立額", "積立後期末資産"])
        df["期末積立額"] = 1
        df.loc[1, "積立後期末資産"] = df.loc[1, "期末積立額"]
        for i in range(2, self.t+1):
            df.loc[i, "積立前期末資産"] = df.loc[i-1, "積立後期末資産"] * (1 + self.r)
            df.loc[i, "積立後期末資産"] = df.loc[i, "積立前期末資産"] + df.loc[i, "期末積立額"]
        return df, 1, df.loc[self.t, "積立後期末資産"]

    # 減債基金係数
    def sff(self):
        """
        Sinking Fund Factor: 減債基金係数        
        """
        df = pd.DataFrame(0, index=range(1, self.t+1), columns=["積立前期末資産", "期末積立額", "積立後期末資産"])
        df["期末積立額"] = InterestFactor(self.r, self.t).sff()
        df.loc[1, "積立後期末資産"] = df.loc[1, "期末積立額"]
        for i in range(2, self.t+1):
            df.loc[i, "積立前期末資産"] = df.loc[i-1, "積立後期末資産"] * (1 + self.r)
            df.loc[i, "積立後期末資産"] = df.loc[i, "積立前期末資産"] + df.loc[i, "期末積立額"]
        return df, InterestFactor(self.r, self.t).sff(), 1

    # 年金現価係数
    def uspwf(self):
        """
        Uniform Series Present Worth Factor: 年金現価係数        
        """
        df = pd.DataFrame(0, index=range(0, self.t+1), columns=["受給前期末資産","期末年金", "受給後期末資産"])
        df.loc[0, "受給後期末資産"] = InterestFactor(self.r, self.t).uspwf()
        df.iloc[1:11, 1] = 1
        for i in range(1, self.t+1):
            df.loc[i, "受給前期末資産"] = df.loc[i-1, "受給後期末資産"] * (1 + self.r)
            df.loc[i, "受給後期末資産"] = df.loc[i, "受給前期末資産"] - df.loc[i, "期末年金"]
        return df, 1, df.loc[0, "受給後期末資産"]
    
    # 資本回収係数
    def crf(self):
        """
        Capital Recovery Factor: 資本回収係数        
        """
        df = pd.DataFrame(0, index=range(0, self.t+1), columns=["受給前期末資産","期末年金", "受給後期末資産"])
        df.loc[0, "受給後期末資産"] = 1
        df.iloc[1:11, 1] =  InterestFactor(self.r, self.t).crf()
        for i in range(1, self.t+1):
            df.loc[i, "受給前期末資産"] = df.loc[i-1, "受給後期末資産"] * (1 + self.r)
            df.loc[i, "受給後期末資産"] = df.loc[i, "受給前期末資産"] - df.loc[i, "期末年金"]
        return df, InterestFactor(self.r, self.t).crf(), 1


if __name__ == "__main__":
    factor = InterestFactor(0.01, 10)
    print('終価係数:', factor.spcaf(),
          '\n現価係数:', factor.sppwf(),
          '\n年金終価係数:', factor.uscaf(),
          '\n減債基金係数:', factor.sff(),
          '\n年金現価係数:', factor.uspwf(),
          '\n資本回収係数:', factor.crf())
