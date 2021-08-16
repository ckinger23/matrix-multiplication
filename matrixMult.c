/*
 *  Matrix Multiplication
 *  Carter King
 *  Dr. Larkins
 *  CS 380 Parallel & Distributed Systems
 *  January, 23, 2019
 *
 *
 *  This program is capable of initializing square matrices and vectors into a 1-D array,
 *  and allows one to test the time it takes to run matrix-matrix multiplication, matrix-
 *  vector multiplication, testing the Frobenius Norm, and transposing a matrix. 
 *
 *
 *  Sources used:
 *  https://stackoverflow.com/questions/13105056/allocate-contiguous-memory
 *  https://stackoverflow.com/questions/15062718/allocate-memory-2d-array-in-function-c
 *  https://www.programiz.com/c-programming/examples/matrix-transpose
 *  http://www.cs.umsl.edu/~sanjiv/classes/cs5740/lectures/mvm.pdf
 */


#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>


typedef double fp_t;

/*
 * prototypes
 */
double get_wctime(void);
void init2d(double** m, int numRaC, double base);
void init1d(double** m, int vecSize, double base);
void mxm(double* source1, double* source2, double** result, int RaCSize);
void mxm2(double* source1, double* source2, double** result, int RaCSize);
void mxv(double* source1, double* sourceVec, double** resultVec, int VecSize);
void mmT(double* readMat, int RaCSize);
double normf( double* mat, int RaCSize);
void printMatrix(double * mat, int numRaC);
void printVector(double * vec, int numItems);


/*
 * Function: get_wctime
 *  This function gets the time and returns it as a double
 * parameters: void
 * returns: double value of the time
 */


double get_wctime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}


/*
 * Function: init2D
 *  This function takes in a pointer to a pointer to a double, or can be thought of as a pointer to a double array, 
 *  and initializes a 2-D array into a 1-D array of space.The function allocates enough memory to hold the given 
 *  amount of data, and initializes the data using the base.
 * parameters:
 *  double** m: a pointer to a pointer to a double
 *  int numRaC: number of Rows and Columns of the 2-D array
 *  double base: The double number used to initialize the content of the array
 * returns; void
 */


void init2d(double** m, int numRaC, double base){
  int total = numRaC * numRaC;
  *m = (double *)malloc(total * sizeof(double));
  if(*m == NULL){
    printf("error allocating mem: ");
    exit(1);
  }
  for(int i = 0; i < total; i++){
    /* Access the every row and column and initialize to a 1-D array */
    int x = i / numRaC; 
    int y = i % numRaC;
    (*m)[i] = base * (x + (0.1 * y));
  }
}


/*
 * Function: init1D
 *  This function initializes a vector or 1-D array into a pointer to a pointer to a double.
 *  A certain amount of memory is allocated for the array based on the amount of items to 
 *  be held in the vector. The vector's contents are then initialized using the base double.
 * parameters:
 *  double** m: The pointer to the double array, vector.
 *  vecSize: the number of items to be held in the vector
 *  base: used to initialize the elements in the vector.
 * returns: void
 */


void init1d(double** m, int vecSize, double base){
  *m =(double *)malloc(vecSize * sizeof(double));
  if(*m == NULL){
    printf("error allocating mem: ");
    exit(1);
  }
  for(int i = 0; i < vecSize; i++){
    (*m)[i] = base * (i+1);
  }  
}


/*
 * Function: mxm
 *  This function performs matrix-matrix multiplication on two 2-D arrays and stores the results
 *  in  a third 2-D array, result. 
 * parameters:
 *  double* source1: the first 2-D array to be multiplied
 *  double* source2: the second 2-D array to be multiplied
 *  double** result: the 2-D array for the multiplication to be stored
 *  int RaCSize: the number of rows and columns in each matrix
 * returns: void
 */


void mxm(double* source1, double* source2, double** result, int RaCSize){
  for(int i = 0; i < RaCSize; i++){    
    for(int j = 0; j < RaCSize; j++){
      for(int k = 0; k < RaCSize; k++){
        (*result)[i * RaCSize + j] += source1[(i * RaCSize) + k] * source2[(k * RaCSize) + j];
        /* C[i][j] += A[i][k] * B[k][j]*/
      }
    }
  }
}


/*
 * Function: mxm2
 *  This function performs matrix-matrix multiplication on two 2-D arrays and stores the results
 *  in  a third 2-D array, result, but the access of source 2 is reversed into row-order 
 * parameters:
 *  double* source1: the first 2-D array to be multiplied
 *  double* source2: the second 2-D array to be multiplied
 *  double** result: the 2-D array for the multiplication to be stored
 *  int RaCSize: the number of rows and columns in each matrix
 * returns: void
 */


void mxm2(double* source1, double* source2, double** result, int RaCSize){
  for(int i = 0; i < RaCSize; i++){
    for(int j = 0; j < RaCSize; j++){
      for(int k = 0; k < RaCSize; k++){
        (*result)[i * RaCSize + j] += source1[(i * RaCSize) + k] * source2[(j * RaCSize) + k];
        /* C[i][j] += A[i][k] * B[j][k] */
      }
    }
  }
}


/*
 * Function: mxv
 *  This function performs matrix-vector multiplication on a 2-D array and a vector  and stores the results
 *  in a vector, resultVec. 
 * parameters:
 *  double* source1: the 2-D array to be multiplied
 *  double* sourceVec: the vector to be used in the multiplication
 *  double** resultVec: the vector for the multiplication to be stored
 *  int VecSize: the number of rows and columns in each matrix
 * returns: void
 */


void mxv(double* source1, double* sourceVec, double** resultVec, int VecSize){
  for( int i = 0; i < VecSize; i++){
    for( int j = 0; j < VecSize; j++){
      (*resultVec)[i] += source1[(i * VecSize) + j] * sourceVec[j];
      /*  resultVec[i] += source1[i][j] * sourceVec[j]; */
    }
  }
}


/*
 * Function: mmT
 *  This function computes the transpose of a square matrix in-place.
 * parameters: 
 *  double* readMat: the Matrix to be transposed
 *  int RaCSize: the number of rows and columns in this matrix
 * returns: void
 */


void mmT(double* readMat, int RaCSize){  
  double trans = 0;
  for(int i = 0; i < RaCSize; i++){
    /* access elements on the diagonal to not overwrite, so j = i */
    for(int j = i; j < RaCSize; j++){
      /* swap method */
      trans = readMat[(i * RaCSize) + j];
      readMat[(i * RaCSize) + j] = readMat[(j * RaCSize) + i];  
      readMat[(j * RaCSize) + i] = trans;
    }
  }
}


/*
 * Function: normf
 *  This function computes the Frobenius norm of a matrix
 * parameters: 
 *  double* mat: this is the matrix to have the norm computed on
 *  int RaCSize: the number of rows and columns of this matrix
 * returns:
 *  a double of the sqrt of the total sum of absolute squares
 */


double normf(double* mat, int RaCSize){
  double total = 0;
  double absV = 0;
  int numItems = pow(RaCSize, 2);
  for(int i = 0; i < numItems; i++){
    absV = fabs(mat[i]);
    total += pow(absV, 2);
  }
  return sqrt(total);
}


/*
 * Function: printMatrix
 *  This function prints out the contents of a matrix in rows and columns
 * parameters: 
 *  double* mat: the Matrix to be printed
 *  int numRaC: the number of rows and columns in the matrix
 * returns: void
 */


void printMatrix(double * mat, int numRaC){
  int counter = 0;
  int numItems = numRaC * numRaC;
  printf("| ");
  for(int i = 0; i < numItems; i++){
    printf("%.2f", mat[i]);
    counter ++;
    if(counter % numRaC == 0){
      printf(" |\n");
      if(i + 1 != numItems){
        printf("| ");
      }
    }
    else{
       printf("%20s", "");
    }
  }
  double Frob = normf(mat, numRaC);
  printf("norm: %.3f \n \n", Frob);
}


/*
 * Function: printVector
 *  This functions prints out a 1-D array/vector
 * parameters:
 *  double* vec: The vector to be printed
 *  int numItems: the total number of items in the vector
 * returns: void
 */


void printVector(double * vec, int numItems){
  printf("| ");
  for(int i = 0; i < numItems; i++){
    printf("%.2f", vec[i]);
    if(i == numItems - 1){
      printf(" |");
      continue;
    }
    else{
      printf("%20s", "");
    }
  }
  printf("\n \n");
}




int main(int argc, char **argv) {
  double start, end, time;
  double * A;
  double * B;
  double * C;
  double * V1;
  double * V2;

  if (argc != 2) {
    printf("usage: lab1 <size>\n\t<size>\t size of matrices and vectors\n");
    exit(-1);
  }

  int RaCSize = atoi(argv[1]);
  printf("Using matrix/vector size: %d \n", RaCSize);
  int numElements = RaCSize * RaCSize;

  /* Time the iInitialization of Matrices and Vectors */
  start = get_wctime(); 
  init2d(&A, RaCSize, 1);
  init2d(&B, RaCSize, 2);
  init2d(&C, RaCSize, 0);

  init1d(&V1, RaCSize, 1);
  init1d(&V2, RaCSize, 0);
  end = get_wctime();
  printf("matrix/vector initialization: %10.7f ms \n", (end - start) * 1000.0);

  /* print Matrices and vectors if the input is 5 or less */
  if(RaCSize <= 5){
    printf("initialized matrices and vectors: \n");
    printf("Matrix A: \n");    
    printMatrix(A, RaCSize); 
    printf("Matrix B: \n");
    printMatrix(B, RaCSize);
    printf("Matrix C: \n");
    printMatrix(C, RaCSize);

    printf("Vector V1: \n");
    printVector(V1, RaCSize);
    printf("Vector V2: \n");
    printVector(V2, RaCSize);
  }
  
  /* Time the matrix-matrix multiplication */
  start = get_wctime();
  mxm(A, B, &C, RaCSize);
  end = get_wctime();
  
  printf("Computing C = A * B (mxm): \n");
  printf("Matrix/Matrix multiplication: %10.7f ms \n", (end - start) * 1000.0);
  if(RaCSize <= 5){
    printf("Matrix C: \n");
    printMatrix(C, RaCSize);
  }
  
  /* Time the Matrix-Vector Multiplication */
  start = get_wctime();
  mxv(B, V1, &V2, RaCSize);
  end = get_wctime();
  printf("Computing V2 = B * V1 \n");
  printf("Matrix/Vector multiplication: %10.7f ms \n", (end - start) * 1000.0);
  if(RaCSize <= 5){
    printf("Vector V2: \n");
    printVector(V2, RaCSize);
  }

  /* Time the matrix transpose */
  start = get_wctime();
  mmT(B, RaCSize);
  end = get_wctime();
  printf("Computing B' \n");
  printf("Matrix transpose: %10.7f ms \n", (end - start) * 1000.0);
  if(RaCSize <= 5){
    printf("matrix B: \n");
    printMatrix(B, RaCSize);
  }
  
  /* Time the second matrix-matrix multiplication */
  start = get_wctime();
  mxm2(A, B, &C, RaCSize);
  end = get_wctime();
  printf("Computing C = A * B (mxm2) \n");
  printf("matrix/matrix multiplication: %10.7f ms \n", (end - start) * 1000.0);
  if(RaCSize <= 5){
    printf("Matrix C: \n");
    printMatrix(C, RaCSize);
  }

  /* Free all the allocated memory */
  free(A);
  free(B);
  free(C);
  free(V1);
  free(V2);
}











