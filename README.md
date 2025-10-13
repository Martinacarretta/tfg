# tfg


png and npy files are saved with the correct orientation. 
    Correct orientation (from axial view):
    Anterior is up
    posterior is down

I flipped the x and y axis with transpose function ONCE during data loading to avoid having to transpose it when plotting...
To plot, display normally, no .t and no origin = lower. 

BUT REMEMBER THE AXIS FOLLOW THE IMAGE y-axis direction CONVENTION (Y AXIS GOES DOWN, NOT UP)

  0  1  2  3
0
1
2
3

NOT 

3
2
1
0
  0  1  2  3