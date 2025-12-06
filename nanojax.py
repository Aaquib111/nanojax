"""Small implementation of functional auto-differentiation in Python

Usage:
import nanojax as nj

f = lambda x: nj.sin(nj.pow(x, 2))
df_dx = grad(f)
df2_d2x = grad(df_dx)
"""
