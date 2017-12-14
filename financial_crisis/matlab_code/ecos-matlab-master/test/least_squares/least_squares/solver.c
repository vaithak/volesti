#include "solver.h"

pwork *setup(params *p)
{
  idxint i = 0;
  double *ptr;
  static double b[48] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  ptr = (double *) p->d;
  for(i = 6; i < 9; ++i) {
    b[i] = *ptr++;
  }
  b[15] = p->e;
  ptr = (double *) p->b;
  for(i = 38; i < 48; ++i) {
    b[i] = *ptr++;
  }
  static double c[58] = {1}; /* rest = {0.0}; */
  static double h[30]; /* = {0.0}; */
  static idxint q[7] = {2, 2, 2, 2, 2, 3, 11};
  static idxint Gjc[59] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30};
  static idxint Gir[30] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  static double Gpr[30] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
  static idxint Ajc[59] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 39, 52, 65, 78, 79, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 94, 96, 98, 100, 102, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150};
  static idxint Air[150] = {16, 17, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0, 3, 1, 4, 2, 5, 0, 6, 1, 7, 2, 8, 9, 10, 11, 12, 13, 14, 9, 15, 18, 28, 19, 29, 20, 30, 21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37, 18, 38, 19, 39, 20, 40, 21, 41, 22, 42, 23, 43, 24, 44, 25, 45, 26, 46, 27, 47};
  static double Apr[150] = {0.5,-0.5,1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0};
  static const idxint A_ind_map[50] = {16, 29, 42, 55, 68, 17, 30, 43, 56, 69, 18, 31, 44, 57, 70, 19, 32, 45, 58, 71, 20, 33, 46, 59, 72, 21, 34, 47, 60, 73, 22, 35, 48, 61, 74, 23, 36, 49, 62, 75, 24, 37, 50, 63, 76, 25, 38, 51, 64, 77};
  ptr = (double *) p->A;
  for (i = 0; i < 50; ++i) {
    Apr[ A_ind_map[i] ] = *ptr++;
  }
  static const idxint C_ind_map[15] = {13, 26, 39, 52, 65, 14, 27, 40, 53, 66, 15, 28, 41, 54, 67};
  ptr = (double *) p->C;
  for (i = 0; i < 15; ++i) {
    Apr[ C_ind_map[i] ] = *ptr++;
  }
  return ECOS_setup(58 /* num vars */, 30 /* num cone constraints */, 48 /* num eq constraints */, 6 /* num linear cones */, 7 /* num second-order cones */, q, Gpr, Gjc, Gir, Apr, Ajc, Air, c, h ,b);
}

int solve(pwork *w, vars *sol)
{
  idxint i = 0;
  int exitflag = ECOS_solve(w);
  for(i = 0; i < 5; ++i) {
    sol->x[i] = w->x[i + 12];
  }
  return exitflag;
}

void cleanup(pwork *w)
{
  ECOS_cleanup(w,0);
}

