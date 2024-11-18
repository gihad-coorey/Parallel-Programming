#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <getopt.h>
#include <string.h>

#define NUM_THREADS_DEFAULT 1
#define DENSITY_DEFAULT 0.01
#define ROWS_DEFAULT 1000
#define COLS_DEFAULT 1000

#define OPTIONS "t:d:x:y:h"

/**
 * Matrix structure.
 * Example 2x3 matrix:
 * 1 2 3
 * 4 5 6
 */
typedef struct {
  char* name;
  int rows;
  int cols;
  int density;
  int** data;
} Matrix;

/**
 * Row-compressed Matrix structure
 * Given original Matrix:
 * 1  0  2
 * 0  0  0
 * 0  4  0
 * 
 * CompressedMatrix `values`:
 * 1  2
 * 0  0
 * 4
 * CompressedMatrix `col_indices`:
 * 0  2
 * 0  0
 * 1
 */
typedef struct {
  char* name;
  int rows;
  int cols;
  int *row_sizes;    // Stores the number of non-zero values in each row
  int **values;      // B matrix: Stores the non-zero values of the matrix
  int **col_indices; // C matrix: Stores the column indices of the non-zero values
} CompressedMatrix;

void parse_options(int argc, char *argv[], int *num_threads, double *density, int *rows_X, int *cols_X, int *rows_Y, int *cols_Y);
int parse_dimensions(const char *dim_str, int *rows, int *cols);
void print_omp_schedule();
void check_allocation(void *ptr, const char *msg);

Matrix *generate_sparse_matrix(int rows, int cols, double density);
Matrix *initialise_matrix(int rows, int cols);
void fill_matrix_with_density(Matrix *matrix, double density);

Matrix *multiply_uncompressed_matrices_parallel(Matrix *matrix_X, Matrix *matrix_Y);
Matrix *multiply_uncompressed_matrices_sequential(Matrix *matrix_X, Matrix *matrix_Y);
void write_uncompressed_matrix_to_file(FILE* file, Matrix *matrix);
void free_matrix(Matrix *matrix);
void compare_matrices(Matrix *matrix1, Matrix *matrix2);

CompressedMatrix *compress_matrix(Matrix *matrix);
Matrix *multiply_compressed_matrices_sequential(CompressedMatrix *compressed_matrix_X, CompressedMatrix *compressed_matrix_Y);
Matrix *multiply_compressed_matrices_parallel(CompressedMatrix *compressed_matrix_X, CompressedMatrix *compressed_matrix_Y);
void write_compressed_matrix_to_file(FILE* fileB, FILE* fileC, CompressedMatrix *compressed_matrix);
void free_compressed_matrix(CompressedMatrix *compressed_matrix);

int num_threads = NUM_THREADS_DEFAULT,
    rows_X = ROWS_DEFAULT, cols_X = COLS_DEFAULT,
    rows_Y = ROWS_DEFAULT, cols_Y = COLS_DEFAULT;

double density = DENSITY_DEFAULT;

int main(int argc, char *argv[]){
  // Parse command line options and print current scheduling strategy
  parse_options(argc, argv, &num_threads, &density, &rows_X, &cols_X, &rows_Y, &cols_Y);
  printf("Density: %f\n", density);
  printf("Matrix X: %d x %d\n", rows_X, cols_X);
  printf("Matrix Y: %d x %d\n", rows_Y, cols_Y);
  printf("Num threads: %d\n", num_threads);
  print_omp_schedule();

  // Generate matrices X and Y
  Matrix *matrix_X = generate_sparse_matrix(rows_X, cols_X, density);
  matrix_X->name = "Matrix X";
  // write_uncompressed_matrix_to_file(stdout, matrix_X);
  printf("Matrix X generated.\n");
  Matrix *matrix_Y = generate_sparse_matrix(rows_Y, cols_Y, density);
  matrix_Y->name = "Matrix Y";
  // write_uncompressed_matrix_to_file(stdout, matrix_Y);
  printf("Matrix Y generated.\n");
  
  // Open file pointers for writing compressed matrices
  FILE *fileB = fopen("FileB.txt", "a");
  FILE *fileC = fopen("FileC.txt", "a");
  if (fileB == NULL || fileC == NULL) {
    fprintf(stderr, "Failed to open files for writing compressed matrices.\n");
    exit(EXIT_FAILURE);
  }

  // Compress matrices X and Y and write to files B and C
  CompressedMatrix *compressed_matrix_X = compress_matrix(matrix_X);
  compressed_matrix_X->name = "Compressed Matrix X";
  write_compressed_matrix_to_file(fileB, fileC, compressed_matrix_X);
  // write_compressed_matrix_to_file(stdout, stdout, compressed_matrix_X);
  printf("Matrix X compressed.\n");
  CompressedMatrix *compressed_matrix_Y = compress_matrix(matrix_Y);
  compressed_matrix_Y->name = "Compressed Matrix Y";
  write_compressed_matrix_to_file(fileB, fileC, compressed_matrix_Y);
  // write_compressed_matrix_to_file(stdout, stdout, compressed_matrix_Y);
  printf("Matrix Y compressed.\n");

  // Close file pointers
  fclose(fileB);
  fclose(fileC);

  // Multiply compressed matrices in parallel and sequentially to compare times
  printf("Starting compressed matrix multiplication...\n");
  Matrix *matrix_result_compressed = multiply_compressed_matrices_sequential(compressed_matrix_X, compressed_matrix_Y);
  free_matrix(matrix_result_compressed);
  matrix_result_compressed = multiply_compressed_matrices_parallel(compressed_matrix_X, compressed_matrix_Y);
  matrix_result_compressed->name = "Compressed Matrix Multiplication Result";
  // write_uncompressed_matrix_to_file(stdout, matrix_result_compressed);
  printf("Compressed matrix multiplication complete.\n");
  free_compressed_matrix(compressed_matrix_X);
  free_compressed_matrix(compressed_matrix_Y);

  // Multiply uncompressed matrices in parallel and sequentially to compare times
  printf("Starting regular matrix multiplication...\n");
  Matrix *matrix_result_uncompressed = multiply_uncompressed_matrices_sequential(matrix_X, matrix_Y);
  free_matrix(matrix_result_uncompressed);
  matrix_result_uncompressed = multiply_uncompressed_matrices_parallel(matrix_X, matrix_Y);
  matrix_result_uncompressed->name = "Regular Matrix Multiplication Result";
  // write_uncompressed_matrix_to_file(stdout, matrix_result_uncompressed);
  printf("Regular matrix multiplication complete.\n");
  free_matrix(matrix_X);
  free_matrix(matrix_Y);

  // Compare results
  printf("Comparing results...\n");
  compare_matrices(matrix_result_uncompressed, matrix_result_compressed);
  free_matrix(matrix_result_uncompressed);
  free_matrix(matrix_result_compressed);

  printf("Done.\n");
  return EXIT_SUCCESS;
}

void print_omp_schedule() {
  omp_sched_t schedule;
  int chunk_size;
  omp_get_schedule(&schedule, &chunk_size);

  // Mask out the monotonic flag if present
  omp_sched_t basic_schedule = schedule & ~omp_sched_monotonic;

  switch (basic_schedule) {
    case omp_sched_static:
      printf("Omp Schedule: Static, Chunk Size: %d\n", chunk_size);
      break;
    case omp_sched_dynamic:
      printf("Omp Schedule: Dynamic, Chunk Size: %d\n", chunk_size);
      break;
    case omp_sched_guided:
      printf("Omp Schedule: Guided, Chunk Size: %d\n", chunk_size);
      break;
    case omp_sched_auto:
      printf("Omp Schedule: Auto, Chunk Size: %d\n", chunk_size);
      break;
    default:
      printf("Omp Schedule: Unknown, Chunk Size: %d\n", chunk_size);
  }

  // Check if monotonic flag is set too
  if (schedule & omp_sched_monotonic) {
    printf(" (Monotonic)\n");
  }
}

void check_allocation(void *ptr, const char *msg) {
  if (ptr == NULL) {
    fprintf(stderr, "Failed to allocate memory: %s\n", msg);
    exit(EXIT_FAILURE);
  }
}

void parse_options(int argc, char *argv[], int *num_threads, double *density, int *rows_X, int *cols_X, int *rows_Y, int *cols_Y) {
  int opt;
  while ((opt = getopt(argc, argv, OPTIONS)) != -1) {
    switch (opt) {
      case 't':
        *num_threads = atoi(optarg);

        if (*num_threads <= 0) {
          fprintf(stderr, "Invalid number of threads.\n");
          exit(EXIT_FAILURE);
        } 
        else if (*num_threads > omp_get_max_threads()) {
          fprintf(stderr,
          "\033[1;33mWARNING: The number of threads (%d) exceeds the maximum available physical cores (%d).\033[0m\n",
          *num_threads, omp_get_max_threads());
        }
        omp_set_num_threads(*num_threads);
        break;
      case 'd':
        *density = strtod(optarg, NULL);
        if (*density <= 0) {
          fprintf(stderr, "Invalid density.");
          exit(EXIT_FAILURE);
        }
        break;
      case 'x':
        if (parse_dimensions(optarg, rows_X, cols_X) != 0) {
          fprintf(stderr, "Invalid dimensions for matrix X");
          exit(EXIT_FAILURE);
        }
        break;
      case 'y':
        if (parse_dimensions(optarg, rows_Y, cols_Y) != 0) {
          fprintf(stderr, "Invalid dimensions for matrix Y");
          exit(EXIT_FAILURE);
        }
        break;
      case 'h':
        fprintf(stdout, "Usage: %s [-t num_threads] [-d density] [-x rowsXcols] [-y rowsXcols] [-h]\n", argv[0]);
        fprintf(stdout, "Example: %s -t 4 -d 0.2 -x 10x20 -y 20x30\n", argv[0]);
        exit(EXIT_SUCCESS);
    }
  }
}

int parse_dimensions(const char *dim_str, int *rows, int *cols) {
  char *x_pos = strchr(dim_str, 'x');
  
  // Invalid format
  if (x_pos == NULL) {
    return -1;
  }

  // Split the string at 'x'
  *x_pos = '\0';
  *rows = atoi(dim_str);
  *cols = atoi(x_pos + 1);

  // Invalid row or column value
  if (*rows <= 0 || *cols <= 0) {
    return -1;  
  }

  return 0;
}

Matrix* initialise_matrix(int rows, int cols) {
  Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
  check_allocation(matrix, "initialise_matrix: matrix");
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->data = (int**) calloc(rows, sizeof(int*));
  check_allocation(matrix->data, "initialise_matrix: matrix->data");
  for (int i = 0; i < rows; i++) {
    matrix->data[i] = (int*) calloc(cols, sizeof(int));
    check_allocation(matrix->data[i], "initialise_matrix: matrix->data[i]");
  }
  return matrix;
}

void write_uncompressed_matrix_to_file(FILE* file, Matrix* matrix) {
  if(matrix == NULL) {
    fprintf(stderr, "Matrix is NULL.\n");
    exit(EXIT_FAILURE);
  }
  if(matrix->data == NULL) {
    fprintf(stderr, "Matrix data is NULL.\n");
    exit(EXIT_FAILURE);
  }

  fprintf(file, "%s\n", matrix->name);
  fprintf(file, "Rows: %d Cols: %d\n", matrix->rows, matrix->cols);
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      fprintf(file, "%3d ", matrix->data[i][j]);
    }
    fprintf(file, "\n");
  }
}

void write_compressed_matrix_to_file(FILE* fileB, FILE* fileC, CompressedMatrix* compressed_matrix) {
  if (compressed_matrix == NULL) {
    fprintf(stderr, "Compressed Matrix is NULL.\n");
    exit(EXIT_FAILURE);
  }
  if (compressed_matrix->values == NULL || compressed_matrix->col_indices == NULL || compressed_matrix->row_sizes == NULL) {
    fprintf(stderr, "Compressed matrix data is NULL.\n");
    exit(EXIT_FAILURE);
  }

  // Values to fileB
  fprintf(fileB, "%s - Rows: %d\n", compressed_matrix->name, compressed_matrix->rows);
  for (int i = 0; i < compressed_matrix->rows; i++) {
    for (int j = 0; j < compressed_matrix->row_sizes[i]; j++) {
      fprintf(fileB, "%3d ", compressed_matrix->values[i][j]);
    }
    fprintf(fileB, "\n");
  }

  // Column indices to fileC
  fprintf(fileC, "%s - Rows: %d\n", compressed_matrix->name, compressed_matrix->rows);
  for (int i = 0; i < compressed_matrix->rows; i++) {
    for (int j = 0; j < compressed_matrix->row_sizes[i]; j++) {
      fprintf(fileC, "%3d ", compressed_matrix->col_indices[i][j]);
    }
    fprintf(fileC, "\n");
  }
}

void free_matrix(Matrix* matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->data[i]);
  }
  free(matrix->data);
  free(matrix);
}

void free_compressed_matrix(CompressedMatrix* compressed_matrix) {
  for (int i = 0; i < compressed_matrix->rows; i++) {
    free(compressed_matrix->values[i]);
    free(compressed_matrix->col_indices[i]);
  }
  free(compressed_matrix->values);
  free(compressed_matrix->col_indices);
  free(compressed_matrix);
}

void compare_matrices(Matrix* matrix1, Matrix* matrix2) {
  if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
    fprintf(stderr, "Matrices have different dimensions.\n");
    return;
  }

  for (int i = 0; i < matrix1->rows; i++) {
    for (int j = 0; j < matrix1->cols; j++) {
      if (matrix1->data[i][j] != matrix2->data[i][j]) {
        printf("Mismatch at (%d, %d): %d != %d\n", i, j, matrix1->data[i][j], matrix2->data[i][j]);
        return;
      }
    }
  }
  printf("Matrices are equal.\n");
  return;
}

void fill_matrix_with_density(Matrix* matrix, double density) {
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      double prob = (double) rand() / RAND_MAX;
      if (prob <= density) {
        matrix->data[i][j] = rand() % 10 + 1;
      } else {
        matrix->data[i][j] = 0;
      }
    }
  }

  matrix->density = density;
}

Matrix* generate_sparse_matrix(int rows, int cols, double density) {
  Matrix* matrix = initialise_matrix(rows, cols);
  fill_matrix_with_density(matrix, density);
  return matrix;
}

Matrix* multiply_uncompressed_matrices_sequential(Matrix* matrix_X, Matrix* matrix_Y) {
  if (matrix_X->cols != matrix_Y->rows) {
    fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix* matrix_result = initialise_matrix(matrix_X->rows, matrix_Y->cols);

  for (int i = 0; i < matrix_X->rows; i++) {
    for (int j = 0; j < matrix_Y->cols; j++) {
      int sum = 0;
      for (int k = 0; k < matrix_X->cols; k++) {
        sum += matrix_X->data[i][k] * matrix_Y->data[k][j];
      }
      matrix_result->data[i][j] = sum;
    }
  }
  end_time = omp_get_wtime();
  printf("Sequential uncompressed matrix multiplication wall time: %fs\n", end_time - start_time);

  return matrix_result;
}

Matrix* multiply_uncompressed_matrices_parallel(Matrix* matrix_X, Matrix* matrix_Y) {
  if (matrix_X->cols != matrix_Y->rows) {
    fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix* matrix_result = initialise_matrix(matrix_X->rows, matrix_Y->cols);

  #pragma omp parallel for collapse(2) default(shared) schedule(runtime)
  for (int i = 0; i < matrix_X->rows; i++) {
    for (int j = 0; j < matrix_Y->cols; j++) {
      int sum = 0;
      for (int k = 0; k < matrix_X->cols; k++) {
        sum += matrix_X->data[i][k] * matrix_Y->data[k][j];
      }
      matrix_result->data[i][j] = sum;
    }
  }
  end_time = omp_get_wtime();
  printf("Parallel uncompressed matrix multiplication wall time: %fs\n", end_time - start_time);

  return matrix_result;
}

CompressedMatrix* compress_matrix(Matrix* matrix) {
  CompressedMatrix* compressed_matrix = (CompressedMatrix*) malloc(sizeof(CompressedMatrix));
  check_allocation(compressed_matrix, "compress_matrix: compressed_matrix");
  compressed_matrix->rows = matrix->rows;
  compressed_matrix->cols = matrix->cols;
  compressed_matrix->values = (int**) calloc(matrix->rows, sizeof(int*));
  check_allocation(compressed_matrix->values, "compress_matrix: compressed_matrix->values");
  compressed_matrix->col_indices = (int**) calloc(matrix->rows, sizeof(int*));
  check_allocation(compressed_matrix->col_indices, "compress_matrix: compressed_matrix->col_indices");
  compressed_matrix->row_sizes = (int*) calloc(matrix->rows, sizeof(int));
  check_allocation(compressed_matrix->row_sizes, "compress_matrix: compressed_matrix->row_sizes");
  
  for (int i = 0; i < matrix->rows; i++) {
    int non_zeroes_in_row = 0;

    // Initially allocate space for the maximum possible elements in the row (i.e., matrix->cols)
    compressed_matrix->values[i] = (int*) calloc(matrix->cols, sizeof(int));
    check_allocation(compressed_matrix->values[i], "compress_matrix: compressed_matrix->values[i]");
    compressed_matrix->col_indices[i] = (int*) calloc(matrix->cols, sizeof(int));
    check_allocation(compressed_matrix->col_indices[i], "compress_matrix: compressed_matrix->col_indices[i]");

    for (int j = 0; j < matrix->cols; j++) {
      if (matrix->data[i][j] != 0) {
        // Store the non-zero value and its column index
        compressed_matrix->values[i][non_zeroes_in_row] = matrix->data[i][j];
        compressed_matrix->col_indices[i][non_zeroes_in_row] = j;
        non_zeroes_in_row++;
      }
    }

    // If there are no non-zero elements, initialize the first two elements to 0
    if (non_zeroes_in_row == 0) {
      compressed_matrix->values[i][0] = 0;
      compressed_matrix->col_indices[i][0] = 0;
      compressed_matrix->values[i][1] = 0;
      compressed_matrix->col_indices[i][1] = 0;
      non_zeroes_in_row = 2;  // Set non_zeroes_in_row to 2 to avoid reallocation
    } else if (non_zeroes_in_row < matrix->cols) {
      // Reallocate to fit exactly the number of non-zero elements
      compressed_matrix->values[i] = (int*) realloc(compressed_matrix->values[i], non_zeroes_in_row * sizeof(int));
      check_allocation(compressed_matrix->values[i], "compress_matrix: compressed_matrix->values[i] realloc");
      compressed_matrix->col_indices[i] = (int*) realloc(compressed_matrix->col_indices[i], non_zeroes_in_row * sizeof(int));
      check_allocation(compressed_matrix->col_indices[i], "compress_matrix: compressed_matrix->col_indices[i] realloc");
    }

    compressed_matrix->row_sizes[i] = non_zeroes_in_row;  // Track non-zero elements in the row
  }
  return compressed_matrix;
}

Matrix* multiply_compressed_matrices_sequential(CompressedMatrix* compressed_matrix_X, CompressedMatrix* compressed_matrix_Y) {
  if (compressed_matrix_X->cols != compressed_matrix_Y->rows) {
    fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix* result = initialise_matrix(compressed_matrix_X->rows, compressed_matrix_Y->cols);
  
  for (int i = 0; i < compressed_matrix_X->rows; i++) {
    // For each row in compressed X
    for (int k = 0; k < compressed_matrix_X->row_sizes[i]; k++) {
      int col_X = compressed_matrix_X->col_indices[i][k];
      int value_X = compressed_matrix_X->values[i][k]; // Value from X at (i, col_X)

      // Check elements in `col_X`-th row of Y
      for (int l = 0; l < compressed_matrix_Y->row_sizes[col_X]; l++) {
        int col_Y = compressed_matrix_Y->col_indices[col_X][l];
        int value_Y = compressed_matrix_Y->values[col_X][l]; // Value from Y at (col_X, col_Y)

        // Update the dot product at position (i, col_Y) in the result matrix
        result->data[i][col_Y] += value_X * value_Y;
      }
    }
  }
  end_time = omp_get_wtime();
  printf("Sequential compressed matrix multiplication wall time: %fs\n", end_time - start_time);

  return result;
}

Matrix* multiply_compressed_matrices_parallel(CompressedMatrix* compressed_matrix_X, CompressedMatrix* compressed_matrix_Y) {
  if (compressed_matrix_X->cols != compressed_matrix_Y->rows) {
    fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix* result = initialise_matrix(compressed_matrix_X->rows, compressed_matrix_Y->cols);

  #pragma omp parallel for default(shared) schedule(runtime)
  for (int i = 0; i < compressed_matrix_X->rows; i++) {
    // For each row in compressed X
    for (int k = 0; k < compressed_matrix_X->row_sizes[i]; k++) {
      int col_X = compressed_matrix_X->col_indices[i][k];
      int value_X = compressed_matrix_X->values[i][k]; // Value from X at (i, col_X)

      // Check elements in `col_X`-th row of Y
      for (int l = 0; l < compressed_matrix_Y->row_sizes[col_X]; l++) {
        int col_Y = compressed_matrix_Y->col_indices[col_X][l];
        int value_Y = compressed_matrix_Y->values[col_X][l]; // Value from Y at (col_X, col_Y)

        // Update the dot product at position (i, col_Y) in the result matrix
        #pragma omp atomic
        result->data[i][col_Y] += value_X * value_Y;
      }
    }
  }
  end_time = omp_get_wtime();
  printf("Parallel compressed matrix multiplication wall time: %fs\n", end_time - start_time);

  return result;
}
