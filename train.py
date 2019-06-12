#!/usr/bin/env python
#coding=utf-8
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import beta_VAE.beta_VAE as beta_VAE

def main():
    beta_VAE = beta_VAE()
    beta_VAE.fit(50)
if __name__ == '__main__':
    main()